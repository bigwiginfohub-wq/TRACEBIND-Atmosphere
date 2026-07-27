"""
22c_tracebind_exact_null_suite.py
---------------------------------
TRACEBIND Phase 5B: Rigorous Multi-Layer Null Testing Suite
- Enforces Machine-Precision 2D Hermitian Phase Symmetry (Parseval Error ~ 1e-16)
- Evaluates Spectral Complexity Profiling (90%, 95%, 99% Mass)
- Decouples Empirical Rank p-Values from Minimum Attainable Resolution
"""

import sys
import os
from pathlib import Path
import numpy as np
import xarray as xr

BASE_DIR = Path(r"C:\TRACEBIND-Atmosphere\phase5")
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# LAYER 1: NUMERICAL VALIDATION ENGINE (MACHINE-PRECISION HERMITIAN)
# ==============================================================================

def enforce_exact_2d_hermitian_phase(phase: np.ndarray, ny: int, nx: int) -> np.ndarray:
    """Enforces strict 2D Hermitian symmetry on rfft2 phase array for exact double-precision Parseval conservation."""
    p = phase.copy()
    
    # DC mode at (0,0) must have 0 phase for positive amplitude
    p[0, 0] = 0.0
    
    # Column kx = 0: Enforce 1D Hermitian reflection along ky
    for ky in range(1, (ny + 1) // 2):
        p[ny - ky, 0] = -p[ky, 0]
    if ny % 2 == 0:
        p[ny // 2, 0] = 0.0
        
    # Column kx = nx // 2 (if nx is even)
    if nx % 2 == 0:
        p[0, nx // 2] = 0.0
        for ky in range(1, (ny + 1) // 2):
            p[ny - ky, nx // 2] = -p[ky, nx // 2]
        if ny % 2 == 0:
            p[ny // 2, nx // 2] = 0.0

    return p

def generate_exact_fourier_surrogate(field: np.ndarray, rng: np.random.Generator):
    """Generates 2D Fourier phase surrogate with exact Parseval energy conservation (~1e-16 relative error)."""
    ny, nx = field.shape
    fft_half = np.fft.rfft2(field)
    amplitude = np.abs(fft_half)
    
    # Generate random phase and apply strict symmetry rules
    random_phase = rng.uniform(-np.pi, np.pi, size=fft_half.shape)
    strict_phase = enforce_exact_2d_hermitian_phase(random_phase, ny, nx)

    surrogate_fft = amplitude * np.exp(1j * strict_phase)
    surrogate_field = np.fft.irfft2(surrogate_fft, s=(ny, nx))

    return surrogate_field

def validate_layer1_numerics(field: np.ndarray, rng: np.random.Generator):
    """Validates Layer 1: Direct Parseval Signal Energy, DC Component, and Spectral Complexity Profile."""
    ny, nx = field.shape
    surrogate_field = generate_exact_fourier_surrogate(field, rng)

    # 1. Direct Parseval Signal Energy Conservation (Sum of Squares)
    energy_orig = np.sum(field**2)
    energy_surr = np.sum(surrogate_field**2)
    parseval_rel_err = abs(energy_orig - energy_surr) / energy_orig

    # 2. DC Component Preservation
    dc_orig = float(np.mean(field))
    dc_surr = float(np.mean(surrogate_field))
    dc_diff = abs(dc_orig - dc_surr)

    # 3. Spectral Complexity Profiling
    fft_half = np.fft.rfft2(field)
    orig_power = np.abs(fft_half)**2
    fluct_power = orig_power.copy()
    fluct_power[0, 0] = 0.0  # Exclude DC
    
    flat_fluct = fluct_power.flatten()
    sorted_fluct = np.sort(flat_fluct)[::-1]
    cum_energy = np.cumsum(sorted_fluct) / np.sum(sorted_fluct)

    modes_90 = int(np.searchsorted(cum_energy, 0.90) + 1)
    modes_95 = int(np.searchsorted(cum_energy, 0.95) + 1)
    modes_99 = int(np.searchsorted(cum_energy, 0.99) + 1)
    total_modes = len(flat_fluct)

    return {
        "parseval_rel_err": parseval_rel_err,
        "dc_orig": dc_orig,
        "dc_surr": dc_surr,
        "dc_diff": dc_diff,
        "modes_90": modes_90,
        "modes_95": modes_95,
        "modes_99": modes_99,
        "total_modes": total_modes,
        "surrogate_field": surrogate_field
    }

# ==============================================================================
# LAYER 2: HYPOTHESIS-SPECIFIC NULL MODELS
# ==============================================================================

def generate_circular_radial_surrogate(field: np.ndarray, center_y: int = None, center_x: int = None):
    """Constructs a pure radial profile P(r) = <P(r, theta)>_theta centered on vortex eye."""
    ny, nx = field.shape
    if center_y is None or center_x is None:
        center_y, center_x = np.unravel_index(np.argmin(field), field.shape)

    y, x = np.ogrid[:ny, :nx]
    r_grid = np.hypot(x - cx, y - cy) if 'cx' in locals() else np.hypot(x - center_x, y - center_y)

    r_flat = r_grid.ravel().astype(int)
    f_flat = field.ravel()
    max_r = r_flat.max()

    radial_mean = np.bincount(r_flat, weights=f_flat) / np.maximum(1, np.bincount(r_flat))
    r_grid_clamped = np.clip(r_grid.astype(int), 0, max_r)
    return radial_mean[r_grid_clamped]

def generate_coherent_band_azimuthal_surrogate(field: np.ndarray, band_width_px: int = 4, 
                                               center_y: int = None, center_x: int = None, 
                                               rng: np.random.Generator = None):
    """Rotates coherent concentric annuli by delta_theta ~ U(0, 2pi) to preserve local radial structure."""
    if rng is None:
        rng = np.random.default_rng()

    ny, nx = field.shape
    if center_y is None or center_x is None:
        center_y, center_x = np.unravel_index(np.argmin(field), field.shape)

    y, x = np.ogrid[:ny, :nx]
    dy, dx = y - center_y, x - center_x
    r_grid = np.hypot(dx, dy)
    theta_grid = np.arctan2(dy, dx)

    band_indices = (r_grid // band_width_px).astype(int)
    max_band = band_indices.max()
    surrogate_field = np.zeros_like(field)

    for b in range(max_band + 1):
        mask = (band_indices == b)
        if not np.any(mask):
            continue

        dtheta = rng.uniform(0, 2 * np.pi)
        theta_rot = (theta_grid[mask] - dtheta) % (2 * np.pi) - np.pi
        r_vals = r_grid[mask]

        x_rot = np.clip(np.round(center_x + r_vals * np.cos(theta_rot)).astype(int), 0, nx - 1)
        y_rot = np.clip(np.round(center_y + r_vals * np.sin(theta_rot)).astype(int), 0, ny - 1)

        surrogate_field[mask] = field[y_rot, x_rot]

    return surrogate_field

# ==============================================================================
# LAYER 3: STATISTICAL INFERENCE ENGINE
# ==============================================================================

def compute_empirical_p_statistics(obs_metric: float, null_metrics: np.ndarray):
    """Computes exact non-parametric empirical p-value and minimum attainable resolution."""
    N = len(null_metrics)
    count = np.sum(null_metrics >= obs_metric)
    
    p_val = (1.0 + count) / (N + 1.0)
    min_attainable_p = 1.0 / (N + 1.0)
    
    return p_val, min_attainable_p, count, N

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def run_pipeline():
    print("=========================================================================================")
    print("          TRACEBIND PHASE 5B: EXACT NUMERICAL & INFERENTIAL NULL SUITE                  ")
    print("=========================================================================================\n")

    nc_files = list(DATA_DIR.glob("era5_*.nc"))
    if not nc_files:
        print("[-] No ERA5 files found in data directory.")
        return

    test_file = nc_files[0]
    print(f"[*] Field Source: {test_file.name}")

    ds = xr.open_dataset(test_file)
    msl_var = "msl" if "msl" in ds.data_vars else list(ds.data_vars.keys())[0]
    frame = ds[msl_var].values[0]
    ds.close()

    rng = np.random.default_rng(42)

    print("\n-----------------------------------------------------------------------------------------")
    print("  LAYER 1: NUMERICAL CORRECTNESS & SPECTRAL COMPLEXITY                                   ")
    print("-----------------------------------------------------------------------------------------")
    diag = validate_layer1_numerics(frame, rng)
    print(f"  Field Dimensions            : {frame.shape}")
    print(f"  Direct Parseval Rel. Error : {diag['parseval_rel_err']:.4e}  [Status: {'PASS (Machine Precision)' if diag['parseval_rel_err'] < 1e-12 else 'FAIL'}]")
    print(f"  DC Amplitude Preserved      : {diag['dc_diff']:.4e} Pa [Status: {'PASS (Machine Precision)' if diag['dc_diff'] < 1e-10 else 'FAIL'}]")
    print("  Spectral Complexity Profile (Cumulative Fluctuating Mass):")
    print(f"    - 90% Energy Mass         : {diag['modes_90']:5d} / {diag['total_modes']} modes ({diag['modes_90']/diag['total_modes']*100:.2f}%)")
    print(f"    - 95% Energy Mass         : {diag['modes_95']:5d} / {diag['total_modes']} modes ({diag['modes_95']/diag['total_modes']*100:.2f}%)")
    print(f"    - 99% Energy Mass         : {diag['modes_99']:5d} / {diag['total_modes']} modes ({diag['modes_99']/diag['total_modes']*100:.2f}%)")

    print("\n-----------------------------------------------------------------------------------------")
    print("  LAYER 2 & 3: MULTI-NULL EVALUATION ON GRADIENT ENERGY                                  ")
    print("-----------------------------------------------------------------------------------------")

    def gradient_energy(f):
        gy, gx = np.gradient(f)
        return np.sum(gx**2 + gy**2)

    obs_ge = gradient_energy(frame)
    print(f"  Observed Field Gradient Energy: {obs_ge:.4e}\n")

    # 1. Fourier Phase Surrogate Null (N=500)
    N_surr = 500
    fourier_ges = np.array([gradient_energy(generate_exact_fourier_surrogate(frame, rng)) for _ in range(N_surr)])
    p_fourier, min_p, count, N = compute_empirical_p_statistics(obs_ge, fourier_ges)

    print(f"  1. Fourier Phase Surrogate Null (N={N}):")
    print(f"     - Null Mean ± StdDev     : {np.mean(fourier_ges):.4e} ± {np.std(fourier_ges):.4e}")
    print(f"     - Empirical p-value      : p = {p_fourier:.4f}  (Exceedance Count: {count}/{N})")
    print(f"     - Minimum Attainable p   : p_min = {min_p:.4f}")

    # 2. Circular Radial Profile Null
    circ_frame = generate_circular_radial_surrogate(frame)
    circ_ge = gradient_energy(circ_frame)
    print(f"\n  2. Circular Radial Profile Null:")
    print(f"     - Gradient Energy        : {circ_ge:.4e}")
    print(f"     - Relative Shift to Obs  : {(circ_ge - obs_ge) / obs_ge * 100:+.2f}%")

    # 3. Coherent Radial Band Azimuthal Rotation Null (N=500)
    band_ges = np.array([gradient_energy(generate_coherent_band_azimuthal_surrogate(frame, band_width_px=4, rng=rng)) for _ in range(N_surr)])
    p_band, _, count_b, _ = compute_empirical_p_statistics(obs_ge, band_ges)

    print(f"\n  3. Coherent Radial Band Azimuthal Null (N={N}):")
    print(f"     - Null Mean ± StdDev     : {np.mean(band_ges):.4e} ± {np.std(band_ges):.4e}")
    print(f"     - Empirical p-value      : p = {p_band:.4f}  (Exceedance Count: {count_b}/{N})")
    print(f"     - Minimum Attainable p   : p_min = {min_p:.4f}")

    print("-----------------------------------------------------------------------------------------\n")

if __name__ == "__main__":
    run_pipeline()