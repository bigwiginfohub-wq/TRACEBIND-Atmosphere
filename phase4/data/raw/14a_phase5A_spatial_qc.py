"""
14a_phase5A_spatial_qc.py
-------------------------
Phase 5A: Spatial Organization Quality Control & Baseline Validation
Computes Gradient Energy (GE) on t=0 against a spatial permutation null.
Logs Cohen's d effect sizes, left-tailed empirical p-values, and provenance.
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
N_PERMUTATIONS = 1000

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

def run_phase5a_spatial_qc():
    print("=========================================================================================")
    print(f"       TRACEBIND PHASE 5A: SPATIAL QC & PERMUTATION NULL (FIELD: {TARGET_FIELD.upper()}) ")
    print("=========================================================================================\n")

    results = []
    rng = np.random.default_rng(42)

    for storm in STORMS:
        nc_path = resolve_nc_path(storm)
        if not nc_path or not nc_path.exists():
            print(f"[-] {storm:10s} | Path not found. Skipping.")
            continue

        ds = xr.open_dataset(nc_path)
        if TARGET_FIELD not in ds:
            ds.close()
            continue

        da = ds[TARGET_FIELD]
        time_dim = next((d for d in ["valid_time", "time", "date"] if d in da.dims), None)
        if not time_dim:
            ds.close()
            continue

        # TEMPORAL SAFETY GUARD
        first_frame = np.squeeze(da.isel({time_dim: 0}).values).astype(np.float64)
        last_frame  = np.squeeze(da.isel({time_dim: -1}).values).astype(np.float64)
        max_delta   = float(np.max(np.abs(last_frame - first_frame)))

        if np.isclose(max_delta, 0.0, atol=1e-12):
            ds.close()
            raise RuntimeError(f"TEMPORAL GUARD TRIGGERED: Static field detected in {storm}.")

        # OBSERVED METRIC
        obs_ge = compute_gradient_energy(first_frame)
        frame_hash = hashlib.md5(first_frame.tobytes()).hexdigest()[:8]

        # FAST SPATIAL PERMUTATION NULL
        flat_frame = first_frame.flatten()
        null_ges = np.empty(N_PERMUTATIONS, dtype=np.float64)
        
        for i in range(N_PERMUTATIONS):
            permuted_2d = rng.permutation(flat_frame).reshape(first_frame.shape)
            null_ges[i] = compute_gradient_energy(permuted_2d)

        null_mean = float(np.mean(null_ges))
        null_std  = float(np.std(null_ges))
        
        # STANDARDIZED EFFECT SIZE (Cohen's d) & EMPIRICAL P-VALUE
        cohen_d = (obs_ge - null_mean) / null_std if null_std > 0 else np.nan
        p_val_left = float(np.sum(null_ges <= obs_ge) + 1) / (N_PERMUTATIONS + 1)

        records = {
            "Storm": storm,
            "Field_Used": TARGET_FIELD,
            "GRIB_Level": da.attrs.get("GRIB_typeOfLevel", da.attrs.get("long_name", "surface")),
            "MD5_Frame0": frame_hash,
            "Temporal_Delta": round(max_delta, 2),
            "Observed_GE": f"{obs_ge:.4e}",
            "Null_GE_Mean": f"{null_mean:.4e}",
            "Cohens_d": round(cohen_d, 2),
            "p_value_left": round(p_val_left, 4),
            "Spatial_QC_Passed": p_val_left < 0.05
        }
        results.append(records)
        print(f"[✓] {storm:8s} | MD5: {frame_hash} | Obs GE: {obs_ge:.3e} | Cohen's d: {cohen_d:6.2f} SD | Left p-val: {p_val_left:.4f}")
        ds.close()

    results_df = pd.DataFrame(results)
    output_path = OUTPUT_DIR / "phase5a_spatial_qc_summary.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\n[+] Saved Phase 5A QC results to: {output_path}\n")
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    run_phase5a_spatial_qc()