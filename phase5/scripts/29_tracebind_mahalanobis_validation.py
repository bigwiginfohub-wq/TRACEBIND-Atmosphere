"""
29_tracebind_mahalanobis_validation.py
---------------------------------------
TRACEBIND Phase 5 Final Closure: Multivariate Departure Analysis

1. Dynamically imports Canonical Fourier Generator from 22c_tracebind_exact_null_suite.py.
2. Computes Standardized Covariance (Correlation Matrix Transformation) for Numerical Stability.
3. Evaluates Multivariate Empirical Mahalanobis Departure D_M on Reduced 5D Space:
   M = (GE, LE, C_orient, A_radial, S_orient) [TV excluded for r > 0.91].
4. Computes Empirical Rank p-values, Null Percentiles (95th/99th), Chi-Square Degrees-of-Freedom tail probabilities, 
   and Condition Number Diagnostics with Singularity Warnings.
5. Saves Cohort Summary, Exceedance Fractions, and Raw D_M Distributions for Manuscript Plotting.
"""

import importlib.util
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from scipy.ndimage import gaussian_filter, laplace
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2

# ==============================================================================
# 1. DYNAMICALLY LOAD EXACT FOURIER GENERATOR FROM 22c SCRIPT
# ==============================================================================

SCRIPT_DIR = Path(__file__).parent
NULL_SUITE_PATH = SCRIPT_DIR / "22c_tracebind_exact_null_suite.py"

if not NULL_SUITE_PATH.exists():
    raise FileNotFoundError(f"[-] Target null generator file not found at: {NULL_SUITE_PATH}")

spec = importlib.util.spec_from_file_location("tracebind_exact_null", NULL_SUITE_PATH)
tracebind_null_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tracebind_null_mod)

# Retrieve validated function from 22c
generate_exact_fourier_surrogate = tracebind_null_mod.generate_exact_fourier_surrogate

# Resolve Base and Output Directories
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 2. REDUCED 5D DESCRIPTOR COMPUTATION WITH NORMALIZED SHANNON ENTROPY
# ==============================================================================

def compute_reduced_vector(f: np.ndarray, sigma_tensor: float = 2.0) -> np.ndarray:
    """Computes M = (GE, LE, C_orient, A_radial, S_orient) [TV omitted for r > 0.91]."""
    ny, nx = f.shape
    gy, gx = np.gradient(f)
    grad_mag = np.hypot(gx, gy) + 1e-12

    ge = np.sum(gx**2 + gy**2)
    le = np.sum(laplace(f)**2)

    # Structure Tensor Alignment
    Jxx = gaussian_filter(gx**2, sigma=sigma_tensor)
    Jyy = gaussian_filter(gy**2, sigma=sigma_tensor)
    Jxy = gaussian_filter(gx * gy, sigma=sigma_tensor)
    
    trace = Jxx + Jyy
    det = Jxx * Jyy - Jxy**2
    sqrt_disc = np.sqrt(np.maximum(0.0, trace**2 / 4.0 - det))
    
    l1, l2 = trace / 2.0 + sqrt_disc, trace / 2.0 - sqrt_disc
    c_orient = np.mean(((l1 - l2) / (l1 + l2 + 1e-12))**2)

    # Radial Pressure Symmetry
    cutoff = np.percentile(f, 10.0)
    mask = f <= cutoff
    w = cutoff - f[mask]
    cy, cx = (np.average(np.where(mask)[0], weights=w), np.average(np.where(mask)[1], weights=w)) if np.sum(w) > 0 else (ny/2, nx/2)
    
    y_g, x_g = np.ogrid[:ny, :nx]
    dy, dx = y_g - cy, x_g - cx
    dist = np.hypot(dx, dy) + 1e-12
    a_radial = np.mean(np.abs((gx * (dx/dist) + gy * (dy/dist)) / grad_mag))

    # Normalized Discrete Shannon Entropy (sum p_i = 1.0)
    angles = np.arctan2(gy, gx)
    hist, _ = np.histogram(angles, bins=36, range=(-np.pi, np.pi))
    p = hist.astype(np.float64)
    p_sum = np.sum(p)
    if p_sum > 0:
        p /= p_sum
        p = p[p > 0]
        s_orient = -np.sum(p * np.log2(p))
    else:
        s_orient = 0.0

    return np.array([ge, le, c_orient, a_radial, s_orient], dtype=np.float64)

# ==============================================================================
# 3. MULTIVARIATE MAHALANOBIS DISCREPANCY EVALUATION
# ==============================================================================

def run_mahalanobis_cohort_validation(n_surrogates: int = 200):
    print("=========================================================================================================")
    print("      TRACEBIND PHASE 5 CLOSURE: MULTIVARIATE MAHALANOBIS DEPARTURE ANALYSIS                            ")
    print("=========================================================================================================\n")

    nc_files = sorted(list(DATA_DIR.glob("*.nc")))
    if not nc_files:
        print(f"[-] No NetCDF cohort datasets found in {DATA_DIR}.")
        return

    # Single global RNG stream across all storms
    global_rng = np.random.default_rng(42)
    summary_rows = []
    surrogate_distributions = {}

    for idx, fpath in enumerate(nc_files, 1):
        storm_name = fpath.stem.replace("era5_", "").upper()
        ds = xr.open_dataset(fpath)
        msl_var = "msl" if "msl" in ds.data_vars else list(ds.data_vars.keys())[0]
        f_obs = ds[msl_var].values[0]
        ds.close()

        # 1. Observed 5D Descriptor Vector
        m_obs_raw = compute_reduced_vector(f_obs)

        # 2. Generate N=200 Exact Fourier Surrogates via 22c generator
        surr_matrix_raw = np.zeros((n_surrogates, 5), dtype=np.float64)
        for i in range(n_surrogates):
            f_surr = generate_exact_fourier_surrogate(f_obs, rng=global_rng)
            surr_matrix_raw[i, :] = compute_reduced_vector(f_surr)

        # 3. Standardization (Standardize features to transform Covariance -> Correlation Matrix)
        mu_null_raw = np.mean(surr_matrix_raw, axis=0)
        std_null_raw = np.std(surr_matrix_raw, axis=0, ddof=1) + 1e-12
        
        m_obs_z = (m_obs_raw - mu_null_raw) / std_null_raw
        surr_matrix_z = (surr_matrix_raw - mu_null_raw) / std_null_raw

        # 4. Correlation Matrix and Scale-Invariant Regularized Inversion
        cov_corr = np.cov(surr_matrix_z, rowvar=False)
        cond_num = np.linalg.cond(cov_corr)
        is_singular = cond_num > 1e10

        eps = 1e-8 * (np.trace(cov_corr) / cov_corr.shape[0])
        inv_cov_corr = np.linalg.pinv(cov_corr + eps * np.eye(5))

        # 5. Compute Mahalanobis Distance (D_M) in Standardized Space
        mu_z = np.zeros(5)
        dm_obs = mahalanobis(m_obs_z, mu_z, inv_cov_corr)

        dm_surrogates = np.array([mahalanobis(surr_matrix_z[i, :], mu_z, inv_cov_corr) for i in range(n_surrogates)])
        surrogate_distributions[storm_name] = dm_surrogates

        # Null Distribution Metrics
        mu_dm_null = np.mean(dm_surrogates)
        p95_null = np.percentile(dm_surrogates, 95.0)
        p99_null = np.percentile(dm_surrogates, 99.0)
        exceeds_99 = bool(dm_obs > p99_null)

        # Inference: Empirical Rank p-value and Chi-Square p-value (df=5)
        empirical_p = (np.sum(dm_surrogates >= dm_obs) + 1) / (n_surrogates + 1)
        chi2_p = float(chi2.sf(dm_obs**2, df=5))

        summary_rows.append({
            "Storm": storm_name,
            "DM_Observed": dm_obs,
            "DM_Null_Mean": mu_dm_null,
            "DM_Null_95th": p95_null,
            "DM_Null_99th": p99_null,
            "Exceeds_99th": exceeds_99,
            "Empirical_Rank_p": empirical_p,
            "Chi2_p_val": chi2_p,
            "Cov_Cond_Num": cond_num,
            "Singular_Warn": is_singular
        })

    df_res = pd.DataFrame(summary_rows)

    print(f"{'Storm Name':<12} | {'D_M (Obs)':<10} | {'Null Mean':<10} | {'Null 95%':<10} | {'Null 99%':<10} | {'Exceed 99%':<10} | {'Rank p-val':<11} | {'Chi2 p-val':<11} | {'Cond Num'}")
    print("-" * 120)
    for _, r in df_res.iterrows():
        p_str = f"< {r['Empirical_Rank_p']:.4f}" if r['Empirical_Rank_p'] <= 0.005 else f"{r['Empirical_Rank_p']:.4f}"
        chi2_str = f"{r['Chi2_p_val']:.2e}" if r['Chi2_p_val'] < 1e-4 else f"{r['Chi2_p_val']:.4f}"
        warn_str = f"{r['Cov_Cond_Num']:.1e} (!)" if r['Singular_Warn'] else f"{r['Cov_Cond_Num']:.1e}"
        exceed_str = "YES" if r["Exceeds_99th"] else "NO"
        
        print(f"{r['Storm']:<12} | {r['DM_Observed']:<10.2f} | {r['DM_Null_Mean']:<10.2f} | {r['DM_Null_95th']:<10.2f} | {r['DM_Null_99th']:<10.2f} | {exceed_str:<10} | {p_str:<11} | {chi2_str:<11} | {warn_str}")
    
    print("-" * 120)
    total_storms = len(df_res)
    exceed_count = df_res['Exceeds_99th'].sum()
    print(f"  Cohort Mean Observed D_M : {df_res['DM_Observed'].mean():.2f}")
    print(f"  Cohort Mean Null D_M     : {df_res['DM_Null_Mean'].mean():.2f}")
    print(f"  Exceedance Summary       : {exceed_count} / {total_storms} storms ({exceed_count/total_storms*100:.1f}%) exceeded the 99th percentile of the Fourier null.")
    print("=========================================================================================================\n")

    # Persist Results and Full Distributions
    df_res.to_csv(RESULTS_DIR / "phase5_final_mahalanobis_cohort.csv", index=False)
    np.savez(RESULTS_DIR / "phase5_surrogate_dm_distributions.npz", **surrogate_distributions)
    print(f"[+] Saved cohort summary to '{RESULTS_DIR / 'phase5_final_mahalanobis_cohort.csv'}'")
    print(f"[+] Saved full surrogate D_M distributions to '{RESULTS_DIR / 'phase5_surrogate_dm_distributions.npz'}'")

if __name__ == "__main__":
    run_mahalanobis_cohort_validation()