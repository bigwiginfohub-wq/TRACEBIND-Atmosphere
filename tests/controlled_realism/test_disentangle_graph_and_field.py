"""
TRACEBIND Field Statistics & Spatial Autocorrelation Benchmark

Key Diagnostics:
1. Spatial Autocorrelation Metric: Computes Moran's I to isolate spatial continuity.
2. Accurate Gradient Boundary Handling: Prevents NaN bleed during field gradient estimation.
3. Method-Clustered Correlation Analysis: Calculates within-method Pearson r & Spearman rho 
   to eliminate method-clustering artifacts.
4. Statistical Uncertainty: Computes 95% Confidence Intervals (mean ± 1.96 * SEM).
"""

import sys
from pathlib import Path
import warnings
import numpy as np
from scipy.ndimage import generic_filter, binary_erosion
from scipy.stats import skew, kurtosis, pearsonr, spearmanr
from typing import Dict, Any, Tuple, List, NamedTuple

# Set project root on sys.path dynamically
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.test_domain_validation import run_pipeline


def generate_synthetic_grf(shape: Tuple[int, int] = (128, 128), 
                           correlation_length: float = 16.0, 
                           seed: int = 42) -> np.ndarray:
    """Generates a reproducible 2D exponential Gaussian Random Field."""
    rng = np.random.default_rng(seed)
    nx, ny = shape
    x, y = np.arange(nx), np.arange(ny)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    
    dist = np.sqrt((xx - nx // 2)**2 + (yy - ny // 2)**2)
    cov = np.exp(-dist / correlation_length)
    
    fft_cov = np.fft.fft2(np.fft.ifftshift(cov))
    white_noise = rng.normal(size=shape)
    fft_noise = np.fft.fft2(white_noise)
    field = np.real(np.fft.ifft2(fft_noise * np.sqrt(np.maximum(0, fft_cov))))
    
    return (field - np.mean(field)) / np.std(field, ddof=1)


def apply_circle_mask(field: np.ndarray, target_frac: float = 0.50) -> np.ndarray:
    """Applies a centered circular occlusion mask (50% missingness)."""
    masked = field.copy()
    nx, ny = field.shape
    cx, cy = nx / 2.0, ny / 2.0
    target_area = (nx * ny) * target_frac
    r = np.sqrt(target_area / np.pi)
    
    xx, yy = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    mask = np.sqrt((xx - cx)**2 + (yy - cy)**2) <= r
    masked[mask] = np.nan
    return masked


def compute_morans_i(field: np.ndarray) -> float:
    """Computes Moran's I spatial autocorrelation for 2D regular grids (ignoring NaNs)."""
    valid_mask = np.isfinite(field)
    if not np.any(valid_mask):
        return 0.0
    
    mean_val = np.nanmean(field)
    z = field - mean_val
    z[~valid_mask] = 0.0
    
    # 4-neighborhood spatial weight kernel
    z_pad = np.pad(z, 1, mode='constant', constant_values=0)
    valid_pad = np.pad(valid_mask, 1, mode='constant', constant_values=False)
    
    neighbors_sum = (
        (z_pad[:-2, 1:-1] * valid_pad[:-2, 1:-1]) +
        (z_pad[2:, 1:-1] * valid_pad[2:, 1:-1]) +
        (z_pad[1:-1, :-2] * valid_pad[1:-1, :-2]) +
        (z_pad[1:-1, 2:] * valid_pad[1:-1, 2:])
    )
    
    numerator = np.sum(z[valid_mask] * neighbors_sum[valid_mask])
    denominator = np.sum(z[valid_mask]**2)
    
    w_sum = (
        valid_pad[:-2, 1:-1].astype(int) +
        valid_pad[2:, 1:-1].astype(int) +
        valid_pad[1:-1, :-2].astype(int) +
        valid_pad[1:-1, 2:].astype(int)
    )
    s0 = np.sum(w_sum[valid_mask])
    n_valid = np.sum(valid_mask)
    
    if denominator == 0 or s0 == 0:
        return 0.0
    
    return float((n_valid / s0) * (numerator / denominator))


def compute_robust_field_moments(field: np.ndarray) -> Tuple[float, float, float, float, float]:
    """
    Computes sample variance (ddof=1), NaN-eroded gradient magnitude, 
    skewness, kurtosis, and Moran's I spatial autocorrelation.
    """
    valid = field[np.isfinite(field)]
    var_val = float(np.var(valid, ddof=1))
    skew_val = float(skew(valid))
    kurt_val = float(kurtosis(valid))
    moran_val = compute_morans_i(field)
    
    # Gradient calculation with NaN erosion to eliminate edge propagation artifacts
    valid_mask = np.isfinite(field)
    eroded_mask = binary_erosion(valid_mask, structure=np.ones((3, 3)))
    
    field_filled = field.copy()
    field_filled[~valid_mask] = np.nanmean(valid)
    gy, gx = np.gradient(field_filled)
    grad_mag_arr = np.sqrt(gx**2 + gy**2)
    
    grad_val = float(np.mean(grad_mag_arr[eroded_mask])) if np.any(eroded_mask) else float(np.nanmean(grad_mag_arr))
    
    return var_val, grad_val, skew_val, kurt_val, moran_val


def run_disentanglement_benchmark(n_realizations: int = 15, grid_shape: Tuple[int, int] = (128, 128)):
    print("\n" + "=" * 135)
    print("      TRACEBIND FIELD STATISTICS & SPATIAL AUTOCORRELATION BENCHMARK")
    print(f"          Grid Resolution: {grid_shape[0]}x{grid_shape[1]} | Independent Realizations: {n_realizations}")
    print("=" * 135)

    fill_stages = [
        "Original", 
        "Masked (Rebuilt)", 
        "Ground-Truth", 
        "Local Window Mean", 
        "Global Mean", 
        "Matched Noise", 
        "Linear Interp"
    ]
    fill_results = {s: {"R": [], "var": [], "grad": [], "skew": [], "kurt": [], "moran": []} for s in fill_stages}

    for i in range(n_realizations):
        seed = 4000 + i
        rng = np.random.default_rng(seed)
        
        # 1. Base Realization
        orig = generate_synthetic_grf(shape=grid_shape, correlation_length=16.0, seed=seed)
        res_orig, model_orig, _, _ = run_pipeline(orig, k=4, n_permutations=20, seed=seed, drop_nan=True)
        
        # 2. Masked (50% Occlusion)
        masked = apply_circle_mask(orig, target_frac=0.50)
        nan_mask = np.isnan(masked)
        res_mask, _, _, _ = run_pipeline(masked, k=4, n_permutations=20, seed=seed, drop_nan=True)

        # 3. Ground-Truth Fill
        f_gt = masked.copy()
        f_gt[nan_mask] = orig[nan_mask]
        res_gt, _, _, _ = run_pipeline(f_gt, k=4, n_permutations=20, seed=seed, drop_nan=True)

        # 4. Local Window Mean Fill
        f_loc = masked.copy()
        valid_mean = np.nanmean(masked)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
            local_avg = generic_filter(masked, np.nanmean, size=5, mode='constant', cval=np.nan)

        local_avg[np.isnan(local_avg)] = valid_mean
        f_loc[nan_mask] = local_avg[nan_mask]
        res_loc, _, _, _ = run_pipeline(f_loc, k=4, n_permutations=20, seed=seed, drop_nan=True)
        
        # 5. Global Mean Fill
        f_glob = masked.copy()
        f_glob[nan_mask] = valid_mean
        res_glob, _, _, _ = run_pipeline(f_glob, k=4, n_permutations=20, seed=seed, drop_nan=True)
        
        # 6. Matched Variance Gaussian Noise Fill
        f_noise = masked.copy()
        valid_std = np.nanstd(masked, ddof=1)
        f_noise[nan_mask] = rng.normal(loc=valid_mean, scale=valid_std, size=np.sum(nan_mask))
        res_noise, _, _, _ = run_pipeline(f_noise, k=4, n_permutations=20, seed=seed, drop_nan=True)
        
        # 7. Linear Interpolation
        from scipy.interpolate import griddata
        nx, ny = grid_shape
        gx_grid, gy_grid = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
        pts = np.column_stack((gx_grid[~nan_mask], gy_grid[~nan_mask]))
        vals = masked[~nan_mask]
        f_lin = griddata(pts, vals, (gx_grid, gy_grid), method="linear")
        if np.isnan(f_lin).any():
            f_lin_nn = griddata(pts, vals, (gx_grid, gy_grid), method="nearest")
            f_lin[np.isnan(f_lin)] = f_lin_nn[np.isnan(f_lin)]
        res_lin, _, _, _ = run_pipeline(f_lin, k=4, n_permutations=20, seed=seed, drop_nan=True)

        # Log Stage Results
        stage_map = [
            ("Original", res_orig.r_observed, orig),
            ("Masked (Rebuilt)", res_mask.r_observed, masked),
            ("Ground-Truth", res_gt.r_observed, f_gt),
            ("Local Window Mean", res_loc.r_observed, f_loc),
            ("Global Mean", res_glob.r_observed, f_glob),
            ("Matched Noise", res_noise.r_observed, f_noise),
            ("Linear Interp", res_lin.r_observed, f_lin)
        ]

        for name, r_val, field_arr in stage_map:
            v, g, s, k, m = compute_robust_field_moments(field_arr)
            fill_results[name]["R"].append(r_val)
            fill_results[name]["var"].append(v)
            fill_results[name]["grad"].append(g)
            fill_results[name]["skew"].append(s)
            fill_results[name]["kurt"].append(k)
            fill_results[name]["moran"].append(m)

    # Print Summary Diagnostics Table with 95% CIs
    print(f"\n{'Fill Strategy':<20} | {'Mean R (±95% CI)':<18} | {'ΔR vs Orig':<10} | {'Variance (σ²)':<13} | {'Mean Grad (<|∇f|>)':<18} | {'Moran\'s I':<10}")
    print("-" * 135)
    
    R_0_mean = np.mean(fill_results["Original"]["R"])
    
    for stage in fill_stages:
        r_arr = np.array(fill_results[stage]["R"])
        r_m = np.mean(r_arr)
        r_ci = 1.96 * (np.std(r_arr, ddof=1) / np.sqrt(n_realizations))
        
        delta_r = r_m - R_0_mean
        v_m = np.mean(fill_results[stage]["var"])
        g_m = np.mean(fill_results[stage]["grad"])
        m_m = np.mean(fill_results[stage]["moran"])
        
        print(f"{stage:<20} | {r_m:.4f} ± {r_ci:.4f}     | {delta_r:<+10.4f} | {v_m:<13.4f} | {g_m:<18.4f} | {m_m:<+10.4f}")

    print("=" * 135)

    # Within-Method Clustered Correlation Diagnostics
    print("\n--- CLUSTERED WITHIN-METHOD CORRELATION DIAGNOSTICS (MEAN ± SD ACROSS METHODS) ---")
    
    pearson_r_var, spearman_p_var = [], []
    pearson_r_grad, spearman_p_grad = [], []
    pearson_r_moran, spearman_p_moran = [], []

    eval_stages = ["Masked (Rebuilt)", "Local Window Mean", "Global Mean", "Matched Noise", "Linear Interp"]
    
    for stage in eval_stages:
        r_diff = np.array(fill_results[stage]["R"]) - np.array(fill_results["Original"]["R"])
        var_diff = np.array(fill_results[stage]["var"]) - np.array(fill_results["Original"]["var"])
        grad_diff = np.array(fill_results[stage]["grad"]) - np.array(fill_results["Original"]["grad"])
        moran_diff = np.array(fill_results[stage]["moran"]) - np.array(fill_results["Original"]["moran"])
        
        if np.std(var_diff) > 1e-8:
            pearson_r_var.append(pearsonr(var_diff, r_diff)[0])
            spearman_p_var.append(spearmanr(var_diff, r_diff)[0])
        if np.std(grad_diff) > 1e-8:
            pearson_r_grad.append(pearsonr(grad_diff, r_diff)[0])
            spearman_p_grad.append(spearmanr(grad_diff, r_diff)[0])
        if np.std(moran_diff) > 1e-8:
            pearson_r_moran.append(pearsonr(moran_diff, r_diff)[0])
            spearman_p_moran.append(spearmanr(moran_diff, r_diff)[0])

    print(f"  1. ΔR vs ΔVariance         : Pearson r = {np.mean(pearson_r_var):+.4f} ± {np.std(pearson_r_var):.4f} | Spearman ρ = {np.mean(spearman_p_var):+.4f} ± {np.std(spearman_p_var):.4f}")
    print(f"  2. ΔR vs ΔGradient         : Pearson r = {np.mean(pearson_r_grad):+.4f} ± {np.std(pearson_r_grad):.4f} | Spearman ρ = {np.mean(spearman_p_grad):+.4f} ± {np.std(spearman_p_grad):.4f}")
    print(f"  3. ΔR vs ΔMoran's I (Spatial Autocorrelation) : Pearson r = {np.mean(pearson_r_moran):+.4f} ± {np.std(pearson_r_moran):.4f} | Spearman ρ = {np.mean(spearman_p_moran):+.4f} ± {np.std(spearman_p_moran):.4f}")
    print("=" * 135 + "\n")


if __name__ == "__main__":
    run_disentanglement_benchmark(n_realizations=15, grid_shape=(128, 128))