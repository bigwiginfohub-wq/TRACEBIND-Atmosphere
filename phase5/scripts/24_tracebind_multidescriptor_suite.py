"""
24_tracebind_multidescriptor_suite.py
------------------------------------
TRACEBIND Phase 5D: Multi-Descriptor Vector Framework
Computes the vector metric M = (GE, TV, C_orient, A_radial, S_orient) across null models.

Statistical Effect Size:
Cliff's Delta (delta = -1.0 to +1.0) measures exact distributional separation.
"""

import sys
import os
from pathlib import Path
import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter

BASE_DIR = Path(r"C:\TRACEBIND-Atmosphere\phase5")
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 1. NULL MODEL GENERATORS (EXACT NUMERICS)
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
# 2. MULTI-DESCRIPTOR METRIC SPACE M(f)
# ==============================================================================

def compute_metric_vector(f: np.ndarray, sigma_tensor: float = 2.0) -> dict[str, float]:
    """Computes the full 5-dimensional descriptor vector M(f)."""
    ny, nx = f.shape
    gy, gx = np.gradient(f)
    grad_mag = np.hypot(gx, gy) + 1e-12

    # 1. Gradient Energy (Roughness)
    ge = float(np.sum(gx**2 + gy**2))

    # 2. Total Variation (Edge Energy)
    tv = float(np.sum(grad_mag))

    # 3. Structure Tensor Orientation Coherence
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

    # 4. Radial Alignment (Relative to Fitted Centroid)
    center_y, center_x = fit_vortex_centroid(f)
    y_grid, x_grid = np.ogrid[:ny, :nx]
    dy, dx = y_grid - center_y, x_grid - center_x
    dist = np.hypot(dx, dy) + 1e-12
    rx, ry = dx / dist, dy / dist
    a_radial = float(np.mean(np.abs((gx * rx + gy * ry) / grad_mag)))

    # 5. Orientation Entropy (Disorder)
    angles = np.arctan2(gy, gx)  # Range [-pi, pi]
    hist, _ = np.histogram(angles, bins=36, range=(-np.pi, np.pi), density=True)
    hist = hist[hist > 0]
    s_orient = float(-np.sum(hist * np.log2(hist)) * (2 * np.pi / 36))  # Scaled Shannon entropy

    return {
        "GE": ge,
        "TV": tv,
        "C_orient": c_orient,
        "A_radial": a_radial,
        "S_orient": s_orient
    }

# ==============================================================================
# 3. STATISTICAL EFFECT SIZE ENGINE
# ==============================================================================

def compute_cliffs_delta(obs_val: float, null_vals: np.ndarray) -> float:
    """Computes Cliff's Delta effect size between an observed value and a null distribution.
    
    delta = (+1.0: Observed strictly greater, -1.0: Observed strictly smaller, 0.0: Identical)
    """
    n = len(null_vals)
    greater = np.sum(obs_val > null_vals)
    smaller = np.sum(obs_val < null_vals)
    return float((greater - smaller) / n)

# ==============================================================================
# 4. EXECUTION PIPELINE
# ==============================================================================

def run_multidescriptor_pipeline():
    print("=========================================================================================")
    print("          TRACEBIND PHASE 5D: MULTI-DESCRIPTOR VECTOR METRIC SUITE                       ")
    print("=========================================================================================\n")

    nc_files = list(DATA_DIR.glob("era5_*.nc"))
    if not nc_files:
        print("[-] No ERA5 dataset found in data directory.")
        return

    test_file = nc_files[0]
    print(f"[*] Field Source: {test_file.name}")

    ds = xr.open_dataset(test_file)
    msl_var = "msl" if "msl" in ds.data_vars else list(ds.data_vars.keys())[0]
    frame = ds[msl_var].values[0]
    ds.close()

    rng = np.random.default_rng(42)

    # Compute observed vector
    m_obs = compute_metric_vector(frame)

    # Compute axisymmetric radial null vector
    circular_null = generate_circular_radial_surrogate(frame)
    m_circ = compute_metric_vector(circular_null)

    # Compute Fourier null ensemble (N=200)
    N_samples = 200
    print(f"[*] Generating N={N_samples} Fourier Phase Surrogates...")
    fourier_ensemble = [compute_metric_vector(generate_exact_fourier_surrogate(frame, rng)) for _ in range(N_samples)]

    # Aggregate Fourier metrics
    fourier_keys = m_obs.keys()
    m_four_mean = {k: np.mean([f[k] for f in fourier_ensemble]) for k in fourier_keys}
    m_four_std = {k: np.std([f[k] for f in fourier_ensemble]) for k in fourier_keys}

    print("\n-----------------------------------------------------------------------------------------")
    print("  MULTI-DESCRIPTOR VECTOR SUMMARY: M = (GE, TV, C_orient, A_radial, S_orient)           ")
    print("-----------------------------------------------------------------------------------------")
    headers = f"{'Metric':<18} | {'Pure Radial':<15} | {'Observed (ERA5)':<16} | {'Fourier Null (Mean ± Std)':<28} | {'Cliff\'s Delta (d)'}"
    print(headers)
    print("-" * len(headers))

    metric_names = {
        "GE": "Gradient Energy",
        "TV": "Total Variation",
        "C_orient": "Orientation Coh.",
        "A_radial": "Radial Alignment",
        "S_orient": "Orientation Entropy"
    }

    for k in fourier_keys:
        null_arr = np.array([f[k] for f in fourier_ensemble])
        delta = compute_cliffs_delta(m_obs[k], null_arr)
        four_str = f"{m_four_mean[k]:.2e} ± {m_four_std[k]:.2e}" if "GE" in k or "TV" in k else f"{m_four_mean[k]:.4f} ± {m_four_std[k]:.4f}"
        
        obs_str = f"{m_obs[k]:.2e}" if "GE" in k or "TV" in k else f"{m_obs[k]:.4f}"
        circ_str = f"{m_circ[k]:.2e}" if "GE" in k or "TV" in k else f"{m_circ[k]:.4f}"

        print(f"{metric_names[k]:<18} | {circ_str:<15} | {obs_str:<16} | {four_str:<28} | d = {delta:+.2f}")

    print("-----------------------------------------------------------------------------------------\n")

if __name__ == "__main__":
    run_multidescriptor_pipeline()