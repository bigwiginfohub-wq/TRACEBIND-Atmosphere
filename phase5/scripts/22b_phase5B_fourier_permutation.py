"""
22b_phase5B_fourier_permutation.py
-----------------------------------
TRACEBIND Phase 5B: Fourier Phase Surrogate Permutation Engine (N=13 Cohort)
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import xarray as xr
import pandas as pd

BASE_DIR = Path(r"C:\TRACEBIND-Atmosphere\phase5")
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EARTH_RADIUS_M = 6371000.0
N_PERMUTATIONS = 1000
RANDOM_SEED = 42

def compute_spherical_ge_density(msl_field: np.ndarray, lats: np.ndarray, lons: np.ndarray) -> float:
    """Computes spatial mean gradient energy density (Pa^2 / m^2)."""
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

def generate_hermitian_fourier_surrogate_2d(field: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Generates a 2D Fourier phase surrogate using Real FFT."""
    ny, nx = field.shape
    orig_mean = np.mean(field)
    orig_std = np.std(field)

    fft_half = np.fft.rfft2(field)
    amplitude = np.abs(fft_half)
    
    random_phase = rng.uniform(-np.pi, np.pi, size=fft_half.shape)
    random_phase[0, 0] = 0.0
    
    if ny % 2 == 0:
        random_phase[ny // 2, 0] = 0.0
    if nx % 2 == 0:
        random_phase[0, nx // 2] = 0.0
    if ny % 2 == 0 and nx % 2 == 0:
        random_phase[ny // 2, nx // 2] = 0.0

    surrogate_fft = amplitude * np.exp(1j * random_phase)
    surrogate_field = np.fft.irfft2(surrogate_fft, s=(ny, nx))
    
    surr_std = np.std(surrogate_field)
    if surr_std > 0:
        surrogate_field = (surrogate_field - np.mean(surrogate_field)) / surr_std * orig_std + orig_mean

    return surrogate_field

def run_fourier_phase_permutation():
    print("=========================================================================================")
    print("      TRACEBIND PHASE 5B: FOURIER PHASE SURROGATE INFERENCE ENGINE (N=13)               ")
    print("=========================================================================================\n")

    matrix_csv = RESULTS_DIR / "phase5_integrated_feature_matrix.csv"
    if not matrix_csv.exists():
        print(f"[-] Missing integrated feature matrix at {matrix_csv}")
        return

    df_matrix = pd.read_csv(matrix_csv)
    df_matrix = df_matrix.sort_values(by="GE_density", ascending=False).reset_index(drop=True)
    df_matrix["GE_density_rank"] = df_matrix.index + 1

    rng = np.random.default_rng(RANDOM_SEED)
    fourier_results = []

    for idx, row in df_matrix.iterrows():
        storm_id = row["storm_id"]
        rank = row["GE_density_rank"]
        ge_obs = row["GE_density"]
        nc_file = DATA_DIR / f"era5_{storm_id}_72h.nc"

        if not nc_file.exists():
            continue

        try:
            ds = xr.open_dataset(nc_file)
            lat_coord = "latitude" if "latitude" in ds.coords else "lat"
            lon_coord = "longitude" if "longitude" in ds.coords else "lon"
            msl_var = "msl" if "msl" in ds.data_vars else list(ds.data_vars.keys())[0]

            lats = ds[lat_coord].values
            lons = ds[lon_coord].values
            msl_data = ds[msl_var].values
            t_steps = msl_data.shape[0]

            null_ge_densities = []

            for perm_i in range(N_PERMUTATIONS):
                perm_frame_means = []
                for t in range(t_steps):
                    frame = msl_data[t].copy()
                    surrogate_frame = generate_hermitian_fourier_surrogate_2d(frame, rng)
                    ge_m = compute_spherical_ge_density(surrogate_frame, lats, lons)
                    perm_frame_means.append(ge_m)
                
                null_ge_densities.append(np.mean(perm_frame_means))

            null_ge_densities = np.array(null_ge_densities)
            null_mean = float(np.mean(null_ge_densities))
            null_std = float(np.std(null_ge_densities))
            
            p05 = float(np.percentile(null_ge_densities, 5))
            p50 = float(np.percentile(null_ge_densities, 50))
            p95 = float(np.percentile(null_ge_densities, 95))

            p_upper = float(np.sum(null_ge_densities >= ge_obs) + 1) / (N_PERMUTATIONS + 1)
            p_lower = float(np.sum(null_ge_densities <= ge_obs) + 1) / (N_PERMUTATIONS + 1)
            p_twosided = float(2 * min(p_upper, p_lower))
            
            z_score = float((ge_obs - null_mean) / null_std) if null_std > 0 else np.nan

            perm_record = row.to_dict()
            perm_record.update({
                "null_model": "Fourier_Phase_Surrogate",
                "N_permutations": N_PERMUTATIONS,
                "null_GE_density_mean": null_mean,
                "null_GE_density_std": null_std,
                "null_p05": p05,
                "null_p50": p50,
                "null_p95": p95,
                "fourier_z_score": round(z_score, 2),
                "p_value_upper": p_upper,
                "p_value_lower": p_lower,
                "p_value_twosided": p_twosided
            })

            fourier_results.append(perm_record)

            print(f"[{rank:02d}/13] Storm: {storm_id.upper():<10} | GE_obs: {ge_obs:.4e} | Null_p50: {p50:.4e} | Z: {z_score:6.2f} | p_upper: {p_upper:.4f}")

            ds.close()

        except Exception as e:
            print(f"[-] Error processing storm {storm_id}: {e}")

    df_fourier = pd.DataFrame(fourier_results)
    
    csv_out = RESULTS_DIR / "phase5B_fourier_permutation_results.csv"
    json_out = RESULTS_DIR / "phase5B_fourier_permutation_results.json"

    df_fourier.to_csv(csv_out, index=False)

    meta_wrapper = {
        "pipeline_version": "TRACEBIND-P5B-FOURIER-v1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": RANDOM_SEED,
        "n_permutations": N_PERMUTATIONS,
        "cohort_size": len(df_fourier),
        "data": df_fourier.to_dict(orient="records")
    }
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(meta_wrapper, f, indent=2)

    print("\n=========================================================================================")
    print("            FOURIER PHASE SURROGATE INFERENCE SUMMARY (N=13)                            ")
    print("=========================================================================================")
    summary_cols = ["GE_density_rank", "storm_id", "GE_density", "null_p50", "null_p95", "fourier_z_score", "p_value_upper"]
    print(df_fourier[summary_cols].to_string(index=False))

if __name__ == "__main__":
    run_fourier_phase_permutation()