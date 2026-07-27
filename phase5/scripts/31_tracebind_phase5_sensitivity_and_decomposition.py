"""
31_tracebind_phase5_sensitivity_and_decomposition.py
-----------------------------------------------------
TRACEBIND Phase 5 Final Validation:
1. LOFO Analysis with Mean, Median, and 95% CI for Delta D_M.
2. Signed & Absolute Mahalanobis Quadratic Decomposition.
3. Ensemble Size Sensitivity Analysis (N = 100, 200, 500).
"""

import importlib.util
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from scipy.spatial.distance import mahalanobis

SCRIPT_DIR = Path(__file__).parent
NULL_SUITE_PATH = SCRIPT_DIR / "22c_tracebind_exact_null_suite.py"

spec = importlib.util.spec_from_file_location("tracebind_exact_null", NULL_SUITE_PATH)
tracebind_null_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tracebind_null_mod)
generate_exact_fourier_surrogate = tracebind_null_mod.generate_exact_fourier_surrogate

import_29 = SCRIPT_DIR / "29_tracebind_mahalanobis_validation.py"
spec29 = importlib.util.spec_from_file_location("script_29", import_29)
mod29 = importlib.util.module_from_spec(spec29)
spec29.loader.exec_module(mod29)
compute_reduced_vector = mod29.compute_reduced_vector

BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

FEATURE_NAMES = ["GE", "LE", "C_orient", "A_radial", "S_orient"]

def run_phase5_final_checks():
    nc_files = sorted(list(DATA_DIR.glob("*.nc")))
    global_rng = np.random.default_rng(42)
    
    # --- Experiment 1: N Sensitivity Check ---
    print("=========================================================================================")
    print("                     EXPERIMENT 1: SURROGATE ENSEMBLE SENSITIVITY                        ")
    print("=========================================================================================")
    
    sens_results = []
    for N in [100, 200, 500]:
        dm_list = []
        cond_list = []
        for fpath in nc_files:
            ds = xr.open_dataset(fpath)
            msl_var = "msl" if "msl" in ds.data_vars else list(ds.data_vars.keys())[0]
            f_obs = ds[msl_var].values[0]
            ds.close()

            m_obs_raw = compute_reduced_vector(f_obs)
            surr_matrix_raw = np.zeros((N, 5), dtype=np.float64)
            for i in range(N):
                f_surr = generate_exact_fourier_surrogate(f_obs, rng=global_rng)
                surr_matrix_raw[i, :] = compute_reduced_vector(f_surr)

            mu_null = np.mean(surr_matrix_raw, axis=0)
            std_null = np.std(surr_matrix_raw, axis=0, ddof=1) + 1e-12

            m_obs_z = (m_obs_raw - mu_null) / std_null
            surr_z = (surr_matrix_raw - mu_null) / std_null

            cov_z = np.cov(surr_z, rowvar=False)
            eps = 1e-8 * (np.trace(cov_z) / 5.0)
            inv_cov = np.linalg.pinv(cov_z + eps * np.eye(5))
            
            cond_num = np.linalg.cond(cov_z + eps * np.eye(5))
            dm = np.sqrt(float(m_obs_z @ inv_cov @ m_obs_z))
            
            dm_list.append(dm)
            cond_list.append(cond_num)

        mean_dm = np.mean(dm_list)
        std_dm = np.std(dm_list)
        mean_cond = np.mean(cond_list)
        print(f"  N = {N:3d} | Cohort Mean D_M: {mean_dm:6.2f} +/- {std_dm:5.2f} | Mean Cov Condition No: {mean_cond:4.2f}")
        sens_results.append({"N": N, "Mean_DM": mean_dm, "Std_DM": std_dm, "Mean_Cond": mean_cond})

    # --- Experiment 2: Decomposition & LOFO (at N=200) ---
    print("\n=========================================================================================")
    print("                     EXPERIMENT 2: LOFO ABLATION & SIGNED DECOMPOSITION                   ")
    print("=========================================================================================")
    
    lofo_deltas = {name: [] for name in FEATURE_NAMES}
    abs_contribs = {name: [] for name in FEATURE_NAMES}

    for fpath in nc_files:
        ds = xr.open_dataset(fpath)
        msl_var = "msl" if "msl" in ds.data_vars else list(ds.data_vars.keys())[0]
        f_obs = ds[msl_var].values[0]
        ds.close()

        m_obs_raw = compute_reduced_vector(f_obs)
        surr_matrix_raw = np.zeros((200, 5), dtype=np.float64)
        for i in range(200):
            f_surr = generate_exact_fourier_surrogate(f_obs, rng=global_rng)
            surr_matrix_raw[i, :] = compute_reduced_vector(f_surr)

        mu_null = np.mean(surr_matrix_raw, axis=0)
        std_null = np.std(surr_matrix_raw, axis=0, ddof=1) + 1e-12

        m_obs_z = (m_obs_raw - mu_null) / std_null
        surr_z = (surr_matrix_raw - mu_null) / std_null

        cov_z = np.cov(surr_z, rowvar=False)
        eps = 1e-8 * (np.trace(cov_z) / 5.0)
        inv_cov = np.linalg.pinv(cov_z + eps * np.eye(5))

        diff = m_obs_z
        dm2_full = float(diff @ inv_cov @ diff)
        dm_full = np.sqrt(dm2_full)

        # Signed quadratic terms
        row_contribs = diff * (inv_cov @ diff)
        abs_sum = np.sum(np.abs(row_contribs))
        
        for idx, name in enumerate(FEATURE_NAMES):
            abs_contribs[name].append((abs(row_contribs[idx]) / abs_sum) * 100.0)

        # LOFO Delta D_M
        for drop_idx, drop_name in enumerate(FEATURE_NAMES):
            keep_indices = [i for i in range(5) if i != drop_idx]
            m_obs_sub = m_obs_z[keep_indices]
            surr_sub = surr_z[:, keep_indices]
            
            cov_sub = np.cov(surr_sub, rowvar=False)
            eps_sub = 1e-8 * (np.trace(cov_sub) / 4.0)
            inv_cov_sub = np.linalg.pinv(cov_sub + eps_sub * np.eye(4))

            dm_ablated = mahalanobis(m_obs_sub, np.zeros(4), inv_cov_sub)
            delta_dm = dm_full - dm_ablated
            lofo_deltas[drop_name].append(delta_dm)

    print("\nLOFO Delta D_M Summary (Full D_M minus Ablated D_M across cohort):")
    for name in FEATURE_NAMES:
        arr = np.array(lofo_deltas[name])
        mean_v = np.mean(arr)
        med_v = np.median(arr)
        ci_low = np.percentile(arr, 2.5)
        ci_high = np.percentile(arr, 97.5)
        print(f"  - Excluding {name:<8}: Mean Delta = {mean_v:5.2f} | Median = {med_v:5.2f} | 95% CI = [{ci_low:5.2f}, {ci_high:5.2f}]")

    print("\nRelative Absolute Contribution (|C_k| / sum|C_j|):")
    for name in FEATURE_NAMES:
        mean_abs_pct = np.mean(abs_contribs[name])
        print(f"  - {name:<10}: {mean_abs_pct:5.2f}%")

    print("=========================================================================================\n")

if __name__ == "__main__":
    run_phase5_final_checks()