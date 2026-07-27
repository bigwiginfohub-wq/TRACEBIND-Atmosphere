"""
14_phase5a_msl_qc.py
--------------------
TRACEBIND Phase 5A: Spatial Organization & Quality Control Validation (MSL)
Validates that MSL input fields represent smooth, dynamic spatial distributions
rather than static topography or spatially uncorrelated white noise.
"""

import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

RAW_DATA_DIR = Path(r"C:\TRACEBIND-Atmosphere\phase4\data\raw")
OUTPUT_DIR = Path(r"C:\TRACEBIND-Atmosphere\phase5\results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STORMS = ["Amphan", "Fani", "Mocha", "Yaas", "Sidr", "Nargis"]
TARGET_FIELD = "msl"
N_SHUFFLES = 1000

def resolve_nc_path(storm_name: str) -> Path | None:
    name_low = storm_name.lower()
    candidates = [
        RAW_DATA_DIR / f"era5_{name_low}_72h.nc",
        RAW_DATA_DIR / f"era5_{name_low}.nc",
        RAW_DATA_DIR / f"{name_low}_era5.nc",
        RAW_DATA_DIR / f"{storm_name}_era5.nc",
    ]
    for p in candidates:
        if p.exists(): return p
    matches = list(RAW_DATA_DIR.glob(f"*{name_low}*.nc"))
    return matches[0] if matches else None

def compute_gradient_energy(array_2d: np.ndarray) -> float:
    gy, gx = np.gradient(array_2d)
    return float(np.sum(gx**2 + gy**2))

def run_phase5a_qc():
    print("=========================================================================================")
    print(f"       TRACEBIND PHASE 5A: SPATIAL ORGANIZATION & QC (FIELD: {TARGET_FIELD.upper()})    ")
    print("=========================================================================================\n")

    results = []
    rng = np.random.default_rng(seed=42)

    for storm in STORMS:
        nc_path = resolve_nc_path(storm)
        if not nc_path or not nc_path.exists():
            print(f"[-] {storm:10s} | File missing. Skipping.")
            continue

        ds = xr.open_dataset(nc_path)
        if TARGET_FIELD not in ds:
            print(f"[-] {storm:10s} | Field '{TARGET_FIELD}' missing.")
            ds.close()
            continue

        da = ds[TARGET_FIELD]
        time_dim = next((d for d in ["valid_time", "time", "date"] if d in da.dims), None)
        
        if not time_dim:
            print(f"[-] {storm:10s} | Missing temporal dimension.")
            ds.close()
            continue

        # TEMPORAL SAFETY GUARD
        first_frame = np.squeeze(da.isel({time_dim: 0}).values).astype(np.float64)
        last_frame  = np.squeeze(da.isel({time_dim: -1}).values).astype(np.float64)
        max_delta   = float(np.max(np.abs(last_frame - first_frame)))

        if np.isclose(max_delta, 0.0, atol=1e-12):
            ds.close()
            raise RuntimeError(
                f"SAFETY GUARD TRIGGERED: Field '{TARGET_FIELD}' in {storm} is temporally static. "
                "Cannot compute dynamic metrics."
            )

        # Compute Observed GE
        obs_ge = compute_gradient_energy(first_frame)
        frame_hash = hashlib.md5(first_frame.tobytes()).hexdigest()[:8]

        # Spatial Permutation Null Distribution
        shuffled_ges = []
        flat_frame = first_frame.flatten()
        
        for _ in range(N_SHUFFLES):
            shuffled_2d = rng.permutation(flat_frame).reshape(first_frame.shape)
            shuffled_ges.append(compute_gradient_energy(shuffled_2d))

        shuffled_ges = np.array(shuffled_ges)
        null_mean = float(np.mean(shuffled_ges))
        null_std  = float(np.std(shuffled_ges))

        # Standardized Effect Size (Cohen's d) & Empirical Left-Tailed P-Value
        cohen_d = (obs_ge - null_mean) / null_std
        p_val_left = float(np.sum(shuffled_ges <= obs_ge) + 1) / (N_SHUFFLES + 1)

        results.append({
            "Storm": storm,
            "Field_Used": TARGET_FIELD,
            "GRIB_Level": da.attrs.get("GRIB_typeOfLevel", da.attrs.get("long_name", "surface")),
            "MD5_Frame0": frame_hash,
            "Temporal_Delta": round(max_delta, 2),
            "Observed_GE": f"{obs_ge:.4e}",
            "Null_GE_Mean": f"{null_mean:.4e}",
            "Null_GE_Std": f"{null_std:.4e}",
            "Cohens_d": round(cohen_d, 2),
            "p_value_left": round(p_val_left, 4),
            "QC_Passed": p_val_left < 0.05
        })
        
        print(
            f"[✓] {storm:8s} | MD5: {frame_hash} | Obs GE: {obs_ge:.3e} | "
            f"Null Mean: {null_mean:.3e} | Cohen's d: {cohen_d:6.2f} SD | Left p-val: {p_val_left:.4f}"
        )
        ds.close()

    results_df = pd.DataFrame(results)
    output_path = OUTPUT_DIR / "phase5a_msl_qc_summary.csv"
    results_df.to_csv(output_path, index=False)
    
    print("\n=========================================================================================")
    print(f"                   PHASE 5A QC RESULTS SAVED TO: {output_path.name}")
    print("=========================================================================================\n")

if __name__ == "__main__":
    run_phase5a_qc()