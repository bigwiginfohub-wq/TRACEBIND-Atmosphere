"""
35_eval_fourier_mahalanobis.py
-----------------------------------------------------
TRACEBIND Phase 5: Shrinkage-Stabilized Fourier Evaluation & Audit
"""

import xarray as xr
import numpy as np
import pandas as pd
import json
import sys
import datetime
import hashlib
from pathlib import Path
from sklearn.covariance import LedoitWolf
from tracebind_core import compute_row_wise_grid_spacing, compute_reduced_vector, generate_exact_fourier_surrogate

SCRIPTS_DIR = Path(__file__).parent
PHASE5_DIR = SCRIPTS_DIR.parent
DATA_DIR = PHASE5_DIR / "data"
RESULTS_DIR = PHASE5_DIR / "results"

DESCRIPTORS_IN = RESULTS_DIR / "storm_descriptors_5d.csv"
MAHALANOBIS_OUT = RESULTS_DIR / "mahalanobis_fourier_null.csv"
AUDIT_OUT = RESULTS_DIR / "audit.json"

N_SURROGATES = 100

def generate_storm_seed(storm_id: str) -> int:
    return int(hashlib.md5(storm_id.encode('utf-8')).hexdigest(), 16) % (2**32 - 1)

def evaluate_surrogate_mahalanobis(n_surrogates: int = N_SURROGATES):
    df_desc = pd.read_csv(DESCRIPTORS_IN)
    feature_cols = ["GE", "LE", "C_orient", "A_radial", "S_orient"]
    
    results = []
    print(f"[*] Evaluating Stabilized Fourier Null (N={n_surrogates} surrogates/storm)...\n")
    
    for _, row in df_desc.iterrows():
        storm_id = str(row['storm_id']).strip()
        v_obs = row[feature_cols].values.astype(float)
        
        nc_file = DATA_DIR / f"era5_{storm_id}_72h.nc"
        if not nc_file.exists():
            nc_file = DATA_DIR / f"era5_{storm_id}.nc"
            
        ds = xr.open_dataset(nc_file)
        dx_rows, dy = compute_row_wise_grid_spacing(ds)
        var_name = 'msl' if 'msl' in ds else 'mean_sea_level_pressure'
        field = ds[var_name].values
        field_2d = field[field.shape[0] // 2, :, :] if field.ndim >= 3 else field
        ds.close()

        base_seed = generate_storm_seed(storm_id)
        surr_vectors = []
        for s_idx in range(n_surrogates):
            surr_field = generate_exact_fourier_surrogate(field_2d, seed=base_seed + s_idx)
            v_surr = compute_reduced_vector(surr_field, dx_rows=dx_rows, dy=dy)
            surr_vectors.append(v_surr)
            
        surr_vectors = np.array(surr_vectors)
        
        # Diagnostics: Feature-level standard deviations
        feat_stds = np.std(surr_vectors, axis=0)
        
        # Ledoit-Wolf Shrinkage for robust covariance estimation
        lw = LedoitWolf(assume_centered=False).fit(surr_vectors)
        cov_lw = lw.covariance_
        cond_num = float(np.linalg.cond(cov_lw))
        
        # Robust Mahalanobis calculation using fitted shrinkage estimator
        d_m = float(lw.mahalanobis(v_obs.reshape(1, -1))[0])
        
        # Calculate surrogate distances under the same covariance structure
        surr_dms = lw.mahalanobis(surr_vectors)
        surrogate_percentile = float(np.mean(d_m > surr_dms) * 100.0)

        results.append({
            "storm_id": storm_id,
            "cohort_type": row["cohort_type"],
            "min_pressure_hpa": row["min_pressure_hpa"],
            "max_wind_kt": row["max_wind_kt"],
            "D_M_unscaled": d_m,
            "surrogate_percentile": surrogate_percentile,
            "null_cov_condition_num": cond_num,
            "n_surrogates": n_surrogates,
            "mean_surrogate_DM": float(np.mean(surr_dms)),
            "std_surrogate_DM": float(np.std(surr_dms)),
            "GE_std": feat_stds[0],
            "LE_std": feat_stds[1],
            "C_std": feat_stds[2],
            "A_std": feat_stds[3],
            "S_std": feat_stds[4]
        })
        print(f"  [✓] {storm_id} ({row['cohort_type']}): Shrinkage D_M = {d_m:.2f} | Percentile = {surrogate_percentile:.1f}% | Cond# = {cond_num:.2e}")

    df_out = pd.DataFrame(results)
    df_out.to_csv(MAHALANOBIS_OUT, index=False)

    audit_data = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "xarray_version": xr.__version__,
        "n_surrogates": n_surrogates,
        "framework_version": "TRACEBIND v1.1 Shrinkage Stabilized",
        "descriptors_computed": feature_cols
    }
    with open(AUDIT_OUT, "w") as f:
        json.dump(audit_data, f, indent=2)

    print(f"\n[*] Mahalanobis distances saved to: {MAHALANOBIS_OUT}")

if __name__ == "__main__":
    evaluate_surrogate_mahalanobis()