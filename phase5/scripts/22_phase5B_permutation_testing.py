"""
22_phase5B_permutation_testing.py
---------------------------------
TRACEBIND Phase 5B: Monte Carlo Spatial Permutation Testing Engine (N=13)

Pipeline Architecture:
  Measurement Layer (Phase 5A) 
  --> Integration Layer (Phase 5 Feature Matrix)
  --> Inference Layer (This Script: Phase 5B Permutation Testing)

Methodology:
1. Loads the frozen phase5_integrated_feature_matrix.csv.
2. For each storm, loads the 3D ERA5 pressure cube (T, Lat, Lon).
3. Evaluates the observed spatial Gradient Energy Density (GE_density).
4. Executes N_PERMUTATIONS (1,000) Monte Carlo iterations:
   - Randomly shuffles spatial pressure pixels within each time frame (destroying coherent spatial gradients).
   - Re-computes spherical spatial gradient energy density on the permuted field.
5. Computes statistical significance metrics:
   - Empirical p-value: (sum(GE_perm >= GE_obs) + 1) / (N_PERMUTATIONS + 1)
   - Standardized Z-Score: (GE_obs - mean(GE_perm)) / std(GE_perm)
6. Appends explicit 'GE_density_rank' and exports inference outputs (CSV & JSON).
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import xarray as xr
import pandas as pd

# Base Directories
BASE_DIR = Path(r"C:\TRACEBIND-Atmosphere\phase5")
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EARTH_RADIUS_M = 6371000.0  # Mean Earth radius in meters
N_PERMUTATIONS = 1000       # Monte Carlo shuffle iterations per storm
RANDOM_SEED = 42            # Strict reproducibility seed

def compute_spherical_ge_density(msl_field: np.ndarray, lats: np.ndarray, lons: np.ndarray) -> float:
    """
    Computes spatial mean gradient energy density (Pa^2 / m^2) for a 2D field.
    """
    dlat_deg = abs(float(lats[1] - lats[0])) if len(lats) > 1 else 0.25
    dy = np.radians(dlat_deg) * EARTH_RADIUS_M
    
    gy = np.gradient(msl_field, dy, axis=0)
    
    dlon_deg = abs(float(lons[1] - lons[0])) if len(lons) > 1 else 0.25
    dlon_rad = np.radians(dlon_deg)
    
    gx = np.zeros_like(msl_field)
    for i, lat_deg in enumerate(lats):
        lat_rad = np.radians(lat_deg)
        dx_meters = dlon_rad * EARTH_RADIUS_M * np.cos(lat_rad)
        gx[i, :] = np.gradient(msl_field[i, :], dx_meters)
        
    grad_sq = gx**2 + gy**2
    return float(np.mean(grad_sq))

def run_phase5B():
    print("=========================================================================================")
    print("        TRACEBIND PHASE 5B: MONTE CARLO SPATIAL PERMUTATION TESTING (N=13)               ")
    print("=========================================================================================\n")

    matrix_csv = RESULTS_DIR / "phase5_integrated_feature_matrix.csv"
    if not matrix_csv.exists():
        print(f"[-] Missing integrated feature matrix at {matrix_csv}")
        return

    df_matrix = pd.read_csv(matrix_csv)
    
    # Sort descending and assign explicit rank
    df_matrix = df_matrix.sort_values(by="GE_density", ascending=False).reset_index(drop=True)
    df_matrix["GE_density_rank"] = df_matrix.index + 1

    np.random.seed(RANDOM_SEED)
    permutation_results = []

    print(f"[*] Starting {N_PERMUTATIONS} spatial shuffle permutations per storm...\n")

    for idx, row in df_matrix.iterrows():
        storm_id = row["storm_id"]
        rank = row["GE_density_rank"]
        ge_obs = row["GE_density"]
        nc_file = DATA_DIR / f"era5_{storm_id}_72h.nc"

        if not nc_file.exists():
            print(f"[-] [{rank:02d}/13] Missing NetCDF file for storm {storm_id}. Skipping.")
            continue

        try:
            ds = xr.open_dataset(nc_file)
            lat_coord = "latitude" if "latitude" in ds.coords else "lat"
            lon_coord = "longitude" if "longitude" in ds.coords else "lon"
            msl_var = "msl" if "msl" in ds.data_vars else list(ds.data_vars.keys())[0]

            lats = ds[lat_coord].values
            lons = ds[lon_coord].values
            msl_data = ds[msl_var].values  # Shape: (T, Lat, Lon)
            t_steps = msl_data.shape[0]

            # Monte Carlo Shuffle Loop
            null_ge_densities = []

            for perm_i in range(N_PERMUTATIONS):
                perm_frame_means = []
                for t in range(t_steps):
                    frame = msl_data[t].copy()
                    shape = frame.shape
                    
                    # Flatten, shuffle spatially, reshape
                    flat_frame = frame.flatten()
                    np.random.shuffle(flat_frame)
                    shuffled_frame = flat_frame.reshape(shape)
                    
                    # Compute GE density on spatially scrambled field
                    ge_m = compute_spherical_ge_density(shuffled_frame, lats, lons)
                    perm_frame_means.append(ge_m)
                
                # Window integrated spatial mean density for this permutation
                null_ge_densities.append(np.mean(perm_frame_means))

            null_ge_densities = np.array(null_ge_densities)

            # Calculate Inference Metrics
            null_mean = float(np.mean(null_ge_densities))
            null_std = float(np.std(null_ge_densities))
            
            # Empirical p-value
            p_val = float(np.sum(null_ge_densities >= ge_obs) + 1) / (N_PERMUTATIONS + 1)
            
            # Z-Score
            z_score = float((ge_obs - null_mean) / null_std) if null_std > 0 else np.nan

            perm_record = row.to_dict()
            perm_record.update({
                "N_permutations": N_PERMUTATIONS,
                "null_GE_density_mean": null_mean,
                "null_GE_density_std": null_std,
                "perm_z_score": round(z_score, 2),
                "perm_p_value": p_val,
                "significant_p001": p_val < 0.001
            })

            permutation_results.append(perm_record)

            print(f"[{rank:02d}/13] Storm: {storm_id.upper():<10} | GE_obs: {ge_obs:.4e} | Null_mean: {null_mean:.4e} | Z: {z_score:6.2f} | p: {p_val:.4f}")

            ds.close()

        except Exception as e:
            print(f"[-] Error processing storm {storm_id}: {e}")

    df_perm = pd.DataFrame(permutation_results)

    # Re-order columns for clarity
    front_cols = [
        "GE_density_rank", "storm_id", "basin", "GE_density", 
        "null_GE_density_mean", "perm_z_score", "perm_p_value", 
        "significant_p001", "GE_peak_ratio", "GE_cv"
    ]
    remaining_cols = [c for c in df_perm.columns if c not in front_cols]
    df_perm = df_perm[front_cols + remaining_cols]

    # Save Outputs
    csv_out = RESULTS_DIR / "phase5B_permutation_test_results.csv"
    json_out = RESULTS_DIR / "phase5B_permutation_test_results.json"

    df_perm.to_csv(csv_out, index=False)

    meta_wrapper = {
        "pipeline_version": "TRACEBIND-P5B-v1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": RANDOM_SEED,
        "n_permutations": N_PERMUTATIONS,
        "cohort_size": len(df_perm),
        "data": df_perm.to_dict(orient="records")
    }
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(meta_wrapper, f, indent=2)

    print("\n=========================================================================================")
    print("                  PHASE 5B PERMUTATION TEST RESULTS SUMMARY                              ")
    print("=========================================================================================")
    summary_cols = ["GE_density_rank", "storm_id", "basin", "GE_density", "null_GE_density_mean", "perm_z_score", "perm_p_value"]
    print(df_perm[summary_cols].to_string(index=False))

    print(f"\n[+] Permutation Test Results exported successfully:")
    print(f"    - CSV:  {csv_out}")
    print(f"    - JSON: {json_out}\n")

if __name__ == "__main__":
    run_phase5B()