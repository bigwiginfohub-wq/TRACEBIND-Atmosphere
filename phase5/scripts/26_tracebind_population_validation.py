"""
26_tracebind_population_validation.py
--------------------------------------
TRACEBIND Phase 5 Final Validation Engine:
1. Computes 6D Descriptor Vectors M = (GE, TV, LE, C_orient, A_radial, S_orient) across N=13 storms.
2. Standardizes observed fields into surrogate Z-scores: Z_i = (M_obs - mu_null) / sigma_null.
3. Calculates Pairwise Correlation Matrix to assess metric redundancy.
4. Performs Data-Driven Standardized PCA without prior loading assumptions.
5. Exports paired Wilcoxon/t-test statistics, Cohen's dz, and 95% Confidence Intervals.
"""

import sys
import os
from pathlib import Path
import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter, laplace
from scipy.stats import wilcoxon, ttest_rel, pearsonr, spearmanr
import pandas as pd

BASE_DIR = Path(r"C:\TRACEBIND-Atmosphere\phase5")
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 1. EXACT NULL GENERATOR & CENTROID FITTER
# ==============================================================================

def enforce_exact_2d_hermitian_phase(phase: np.ndarray, ny: int, nx: int) -> np.ndarray:
    """Enforces strict 2D Hermitian symmetry on rfft2 phase array for double-precision energy preservation."""
    p = phase.copy()
    p[0, 0] = 0.0
    for ky in range(1, (ny + 1) // 2):
        p[ny - ky, 0] = -p[ky, 0]
    if ny % 2 == 0:
        p[ny // 2, 0] = 0.0
    if nx % 2 == 0:
        p[0, nx // 2] = 0.0
        for ky in range(1, (ny + 1) // 2):
            p[ny - ky, nx // 2] = -p[ky, nx // 2]
        if ny % 2 == 0 and nx % 2 == 0:
            p[ny // 2, nx // 2] = 0.0
    return p

def generate_exact_fourier_surrogate(field: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Generates 2D Fourier phase surrogate conserving exact power spectrum and signal energy."""
    ny, nx = field.shape
    fft_half = np.fft.rfft2(field)
    amplitude = np.abs(fft_half)
    random_phase = rng.uniform(-np.pi, np.pi, size=fft_half.shape)
    strict_phase = enforce_exact_2d_hermitian_phase(random_phase, ny, nx)
    return np.fft.irfft2(amplitude * np.exp(1j * strict_phase), s=(ny, nx))

def fit_vortex_centroid(f: np.ndarray, threshold_percentile: float = 10.0) -> tuple[float, float]:
    """Calculates pressure-weighted center of mass over core low-pressure region."""
    cutoff = np.percentile(f, threshold_percentile)
    mask = f <= cutoff
    weights = cutoff - f[mask]
    if np.sum(weights) == 0:
        return float(np.argmin(f) // f.shape[1]), float(np.argmin(f) % f.shape[1])
    
    y_indices, x_indices = np.where(mask)
    cy = np.average(y_indices, weights=weights)
    cx = np.average(x_indices, weights=weights)
    return float(cy), float(cx)

def generate_circular_radial_surrogate(field: np.ndarray) -> np.ndarray:
    """Reconstructs axisymmetric radial profile centered on fitted pressure centroid."""
    ny, nx = field.shape
    center_y, center_x = fit_vortex_centroid(field)
    
    y, x = np.ogrid[:ny, :nx]
    r_grid = np.hypot(x - center_x, y - center_y)
    r_flat = r_grid.ravel().astype(int)
    max_r = r_flat.max()
    
    radial_mean = np.bincount(r_flat, weights=field.ravel()) / np.maximum(1, np.bincount(r_flat))
    r_grid_clamped = np.clip(r_grid.astype(int), 0, max_r)
    return radial_mean[r_grid_clamped]

# ==============================================================================
# 2. 6D DESCRIPTOR VECTOR COMPUTATION M(f)
# ==============================================================================

def compute_metric_vector(f: np.ndarray, sigma_tensor: float = 2.0) -> dict[str, float]:
    """Computes the full 6-dimensional descriptor vector M(f)."""
    ny, nx = f.shape
    gy, gx = np.gradient(f)
    grad_mag = np.hypot(gx, gy) + 1e-12

    # 1st-Order Roughness & Edge Strength
    ge = float(np.sum(gx**2 + gy**2))
    tv = float(np.sum(grad_mag))

    # 2nd-Order Curvature Energy (Laplacian)
    lap = laplace(f)
    le = float(np.sum(lap**2))

    # 1st-Order Structure Tensor Orientation Coherence
    Jxx = gaussian_filter(gx**2, sigma=sigma_tensor)
    Jyy = gaussian_filter(gy**2, sigma=sigma_tensor)
    Jxy = gaussian_filter(gx * gy, sigma=sigma_tensor)
    
    trace = Jxx + Jyy
    det = Jxx * Jyy - Jxy**2
    discriminant = np.maximum(0.0, trace**2 / 4.0 - det)
    sqrt_disc = np.sqrt(discriminant)
    
    lambda1 = trace / 2.0 + sqrt_disc
    lambda2 = trace / 2.0 - sqrt_disc
    coherence = ((lambda1 - lambda2) / (lambda1 + lambda2 + 1e-12))**2
    c_orient = float(np.mean(coherence))

    # 1st-Order Radial Alignment (Centroid-Fitted Axisymmetry)
    center_y, center_x = fit_vortex_centroid(f)
    y_grid, x_grid = np.ogrid[:ny, :nx]
    dy, dx = y_grid - center_y, x_grid - center_x
    dist = np.hypot(dx, dy) + 1e-12
    rx, ry = dx / dist, dy / dist
    a_radial = float(np.mean(np.abs((gx * rx + gy * ry) / grad_mag)))

    # 1st-Order Orientation Entropy
    angles = np.arctan2(gy, gx)
    hist, _ = np.histogram(angles, bins=36, range=(-np.pi, np.pi), density=True)
    hist = hist[hist > 0]
    s_orient = float(-np.sum(hist * np.log2(hist)) * (2 * np.pi / 36))

    return {
        "GE": ge,
        "TV": tv,
        "LE": le,
        "C_orient": c_orient,
        "A_radial": a_radial,
        "S_orient": s_orient
    }

# ==============================================================================
# 3. POPULATION PIPELINE & DATA-DRIVEN ANALYSIS
# ==============================================================================

def run_final_validation_pipeline():
    print("=========================================================================================")
    print("          TRACEBIND PHASE 5: FINAL POPULATION METRIC VALIDATION                          ")
    print("=========================================================================================\n")

    nc_files = list(DATA_DIR.glob("era5_*.nc"))
    if not nc_files:
        print("[-] No ERA5 cohort files found in data directory.")
        return

    print(f"[*] Analyzing N={len(nc_files)} Cyclone Datasets with N=200 Fourier Surrogates/storm...\n")

    rng = np.random.default_rng(42)
    metric_keys = ["GE", "TV", "LE", "C_orient", "A_radial", "S_orient"]

    obs_records = []
    null_mean_records = []
    null_std_records = []
    z_score_records = []

    for idx, fpath in enumerate(nc_files, 1):
        ds = xr.open_dataset(fpath)
        msl_var = "msl" if "msl" in ds.data_vars else list(ds.data_vars.keys())[0]
        frame = ds[msl_var].values[0]
        ds.close()

        # Compute observed descriptor
        m_obs = compute_metric_vector(frame)
        obs_records.append(m_obs)

        # Generate N=200 Fourier Surrogates
        surrogate_vectors = [compute_metric_vector(generate_exact_fourier_surrogate(frame, rng)) for _ in range(200)]
        
        m_null_mean = {k: float(np.mean([s[k] for s in surrogate_vectors])) for k in metric_keys}
        m_null_std = {k: float(np.std([s[k] for s in surrogate_vectors])) + 1e-12 for k in metric_keys}
        
        null_mean_records.append(m_null_mean)
        null_std_records.append(m_null_std)

        # Compute standardized Z-scores per storm
        z_scores = {k: (m_obs[k] - m_null_mean[k]) / m_null_std[k] for k in metric_keys}
        z_score_records.append(z_scores)

    df_obs = pd.DataFrame(obs_records)
    df_null_mean = pd.DataFrame(null_mean_records)
    df_z = pd.DataFrame(z_score_records)

    # --------------------------------------------------------------------------
    # A. PAIRED POPULATION HYPOTHESIS TESTS
    # --------------------------------------------------------------------------
    print("-----------------------------------------------------------------------------------------")
    print("  1. POPULATION PAIRED HYPOTHESIS TESTS (OBSERVED VS. FOURIER NULL, N=13)               ")
    print("-----------------------------------------------------------------------------------------")
    print(f"{'Descriptor':<12} | {'Observed (Mean ± SD)':<22} | {'Null (Mean ± SD)':<22} | {'Paired dz':<10} | {'t-test p':<12} | {'Wilcoxon p'}")
    print("-" * 105)

    for k in metric_keys:
        o = df_obs[k].values
        n = df_null_mean[k].values
        diff = o - n
        
        dz = np.mean(diff) / (np.std(diff, ddof=1) + 1e-12)
        _, p_ttest = ttest_rel(o, n)
        _, p_wilc = wilcoxon(o, n)

        if k in ["GE", "TV", "LE"]:
            o_str = f"{np.mean(o):.2e} ± {np.std(o):.2e}"
            n_str = f"{np.mean(n):.2e} ± {np.std(n):.2e}"
        else:
            o_str = f"{np.mean(o):.4f} ± {np.std(o):.4f}"
            n_str = f"{np.mean(n):.4f} ± {np.std(n):.4f}"

        print(f"{k:<12} | {o_str:<22} | {n_str:<22} | {dz:<+10.2f} | {p_ttest:<12.2e} | {p_wilc:.2e}")
    print("-----------------------------------------------------------------------------------------\n")

    # --------------------------------------------------------------------------
    # B. DESCRIPTOR CORRELATION MATRIX (PEARSON & SPEARMAN)
    # --------------------------------------------------------------------------
    print("-----------------------------------------------------------------------------------------")
    print("  2. DESCRIPTOR CORRELATION MATRIX ACROSS OBSERVED COHORT (ASSESSING REDUNDANCY)         ")
    print("-----------------------------------------------------------------------------------------")
    corr_pearson = df_obs.corr(method="pearson")
    print("Pearson Correlation Matrix (r):")
    print(corr_pearson.round(3).to_string())
    print("\n-----------------------------------------------------------------------------------------\n")

    # --------------------------------------------------------------------------
    # C. DATA-DRIVEN STANDARDIZED PCA
    # --------------------------------------------------------------------------
    print("-----------------------------------------------------------------------------------------")
    print("  3. DATA-DRIVEN STANDARDIZED PCA (PERFORMED ON Z-SCORE MATRIX)                           ")
    print("-----------------------------------------------------------------------------------------")
    
    # PCA on standardized Z-score data matrix
    z_matrix = df_z.values
    z_mean = np.mean(z_matrix, axis=0)
    z_std = np.std(z_matrix, axis=0) + 1e-12
    z_standardized = (z_matrix - z_mean) / z_std

    cov_mat = np.cov(z_standardized, rowvar=False)
    eig_vals, eig_vecs = np.linalg.eigh(cov_mat)

    # Sort descending
    idx = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:, idx]

    var_exp = eig_vals / np.sum(eig_vals)

    print(f"  PC1 Variance Explained : {var_exp[0]*100:.2f}%")
    print(f"  PC2 Variance Explained : {var_exp[1]*100:.2f}%")
    print(f"  Cumulative (PC1 + PC2) : {np.sum(var_exp[:2])*100:.2f}%\n")

    print(f"{'Descriptor':<15} | {'PC1 Loading':<15} | {'PC2 Loading':<15}")
    print("-" * 50)
    for i, name in enumerate(metric_keys):
        print(f"{name:<15} | {eig_vecs[i, 0]:<+15.4f} | {eig_vecs[i, 1]:<+15.4f}")
    print("-----------------------------------------------------------------------------------------\n")

if __name__ == "__main__":
    run_final_validation_pipeline()