"""
28_tracebind_mahalanobis_distance_suite.py
-------------------------------------------
TRACEBIND Phase 5/6 Migration:
1. Reduced 5D Descriptor Vector M = (GE, LE, C_orient, A_radial, S_orient) [TV excluded for r > 0.91].
2. Computes Mahalanobis Distance D_M across N=13 ERA5 storms against N=200 Fourier surrogates/storm.
3. Quantifies separation in units of the null covariance matrix.
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from scipy.ndimage import gaussian_filter, laplace
from scipy.spatial.distance import mahalanobis

BASE_DIR = Path(r"C:\TRACEBIND-Atmosphere\phase5")
DATA_DIR = BASE_DIR / "data"

# ==============================================================================
# REDUCED 5D DESCRIPTOR COMPUTATION
# ==============================================================================

def compute_reduced_metric_vector(f: np.ndarray, sigma_tensor: float = 2.0) -> dict[str, float]:
    """Computes reduced 5D descriptor vector (excluding TV due to redundancy with GE/LE)."""
    ny, nx = f.shape
    gy, gx = np.gradient(f)
    grad_mag = np.hypot(gx, gy) + 1e-12

    # Derivative Energy (1st and 2nd Order)
    ge = float(np.sum(gx**2 + gy**2))
    le = float(np.sum(laplace(f)**2))

    # Geometric Organization Family
    Jxx = gaussian_filter(gx**2, sigma=sigma_tensor)
    Jyy = gaussian_filter(gy**2, sigma=sigma_tensor)
    Jxy = gaussian_filter(gx * gy, sigma=sigma_tensor)
    
    trace = Jxx + Jyy
    det = Jxx * Jyy - Jxy**2
    sqrt_disc = np.sqrt(np.maximum(0.0, trace**2 / 4.0 - det))
    
    lambda1 = trace / 2.0 + sqrt_disc
    lambda2 = trace / 2.0 - sqrt_disc
    c_orient = float(np.mean(((lambda1 - lambda2) / (lambda1 + lambda2 + 1e-12))**2))

    # Radial Alignment
    cutoff = np.percentile(f, 10.0)
    mask = f <= cutoff
    w = cutoff - f[mask]
    cy, cx = (np.average(np.where(mask)[0], weights=w), np.average(np.where(mask)[1], weights=w)) if np.sum(w) > 0 else (ny/2, nx/2)
    
    y_grid, x_grid = np.ogrid[:ny, :nx]
    dy, dx = y_grid - cy, x_grid - cx
    dist = np.hypot(dx, dy) + 1e-12
    a_radial = float(np.mean(np.abs((gx * (dx/dist) + gy * (dy/dist)) / grad_mag)))

    # Orientation Entropy
    angles = np.arctan2(gy, gx)
    hist, _ = np.histogram(angles, bins=36, range=(-np.pi, np.pi), density=True)
    hist = hist[hist > 0]
    s_orient = float(-np.sum(hist * np.log2(hist)) * (2 * np.pi / 36))

    return {"GE": ge, "LE": le, "C_orient": c_orient, "A_radial": a_radial, "S_orient": s_orient}

# ==============================================================================
# MAHALANOBIS DISTANCE COMPUTATION
# ==============================================================================

def compute_storm_mahalanobis_distance(f_obs: np.ndarray, n_surrogates: int = 200) -> float:
    """Calculates Mahalanobis distance D_M of observed cyclone from its surrogate cloud."""
    rng = np.random.default_rng(42)
    keys = ["GE", "LE", "C_orient", "A_radial", "S_orient"]
    
    m_obs = np.array([compute_reduced_metric_vector(f_obs)[k] for k in keys])
    
    # Generate Surrogates
    surr_vectors = []
    ny, nx = f_obs.shape
    fft_half = np.fft.rfft2(f_obs)
    amp = np.abs(fft_half)

    for _ in range(n_surrogates):
        rand_p = rng.uniform(-np.pi, np.pi, size=fft_half.shape)
        # Enforce Hermitian Symmetry
        rand_p[0, 0] = 0.0
        f_surr = np.fft.irfft2(amp * np.exp(1j * rand_p), s=(ny, nx))
        m_s = np.array([compute_reduced_metric_vector(f_surr)[k] for k in keys])
        surr_vectors.append(m_s)

    surr_matrix = np.array(surr_vectors) # Shape: (200, 5)
    mu_null = np.mean(surr_matrix, axis=0)
    cov_null = np.cov(surr_matrix, rowvar=False)
    
    # Regularized Inverse Covariance Matrix
    inv_cov_null = np.linalg.pinv(cov_null + 1e-8 * np.eye(len(keys)))
    
    d_m = mahalanobis(m_obs, mu_null, inv_cov_null)
    return float(d_m)

if __name__ == "__main__":
    print("=========================================================================================")
    print("  TRACEBIND: MAHALANOBIS DISTANCE EVALUATION (OBSERVED VS. NULL COVARIANCE)               ")
    print("=========================================================================================")
    print("  Feature Vector M = (GE, LE, C_orient, A_radial, S_orient) [Reduced 5D Space]\n")
    print("  Status: Ready to execute over N=13 ERA5 Cohort.")