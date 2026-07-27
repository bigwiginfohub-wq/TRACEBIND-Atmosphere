"""
22b_tracebind_null_suite.py
---------------------------
TRACEBIND Phase 5B: Multi-Layer Null Suite & Generator Engine
- Layer 1: Numerical Correctness (Direct Parseval, DC preservation, Spectral Complexity)
- Layer 2: Domain Null Generators (Fourier Phase, Coherent Radial Band, Circular Profile)
- Layer 3: Empirical Rank p-Value Estimator
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

EARTH_RADIUS_M = 6371000.0

# ==============================================================================
# LAYER 1: NUMERICAL VALIDATION & DIAGNOSTICS
# ==============================================================================

def validate_fourier_generator_numerics(field: np.ndarray, rng: np.random.Generator):
    """Validates Layer 1: Direct Parseval Energy, DC Mode Preservation, and Spectral Complexity."""
    ny, nx = field.shape
    fft_half = np.fft.rfft2(field)
    amplitude = np.abs(fft_half)
    
    # Randomize phases while respecting Hermitian symmetry
    random_phase = rng.uniform(-np.pi, np.pi, size=fft_half.shape)
    random_phase[0, 0] = 0.0  # Preserve DC mode
    if ny % 2 == 0:
        random_phase[ny // 2, 0] = 0.0
    if nx % 2 == 0:
        random_phase[0, nx // 2] = 0.0
    if ny % 2 == 0 and nx % 2 == 0:
        random_phase[ny // 2, nx // 2] = 0.0

    surrogate_fft = amplitude * np.exp(1j * random_phase)
    surrogate_field = np.fft.irfft2(surrogate_fft, s=(ny, nx))

    # 1. Direct Parseval Verification: Sum of squares of field
    energy_orig = np.sum(field**2)
    energy_surr = np.sum(surrogate_field**2)
    parseval_rel_err = abs(energy_orig - energy_surr) / energy_orig

    # 2. DC Component Check
    dc_orig = float(np.mean(field))
    dc_surr = float(np.mean(surrogate_field))
    dc_diff = abs(dc_orig - dc_surr)

    # 3. Spectral Complexity Profiling (Modes needed for 90%, 95%, 99% energy)
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
# LAYER 2: SPECIALIZED NULL GENERATORS
# ==============================================================================

def generate_circular_radial_surrogate(field: np.ndarray, center_y: int = None, center_x: int = None):
    """Constructs a pure radial profile P(r) = <P(r, theta)>_theta.

    Strips all angular asymmetries/spiral features while keeping storm scale & core depth.
    """
    ny, nx = field.shape
    if center_y is None or center_x is None:
        # Default to field minimum (vortex eye)
        center_y, center_x = np.unravel_index(np.argmin(field), field.shape)

    y, x = np.ogrid[:ny, :nx]
    r_grid = np.hypot(x - center_x, y - center_y)

    # Bin radial distances
    r_flat = r_grid.ravel()
    f_flat = field.ravel()
    
    r_int = r_flat.astype(int)
    max_r = r_int.max()
    
    radial_mean = np.bincount(r_int, weights=f_flat) / np.maximum(1, np.bincount(r_int))
    
    # Reconstruct 2D circular field
    r_grid_clamped = np.clip(r_grid.astype(int), 0, max_r)
    circular_field = radial_mean[r_grid_clamped]
    
    return circular_field

def generate_coherent_band_azimuthal_surrogate(field: np.ndarray, band_width_px: int = 4, 
                                               center_y: int = None, center_x: int = None, 
                                               rng: np.random.Generator = None):
    """Applies uniform angular rotation delta_theta per radial annulus band.

    Preserves radial profile and local radial gradients while destroying spiral phase alignment.
    """
    if rng is None:
        rng = np.random.default_rng()

    ny, nx = field.shape
    if center_y is None or center_x is None:
        center_y, center_x = np.unravel_index(np.argmin(field), field.shape)

    y, x = np.ogrid[:ny, :nx]
    dy = y - center_y
    dx = x - center_x
    r_grid = np.hypot(dx, dy)
    theta_grid = np.arctan2(dy, dx)

    band_indices = (r_grid // band_width_px).astype(int)
    max_band = band_indices.max()

    surrogate_field = np.zeros_like(field)

    for b in range(max_band + 1):
        mask = (band_indices == b)
        if not np.any(mask):
            continue

        # Single random rotation offset for this entire radial band
        dtheta = rng.uniform(0, 2 * np.pi)
        
        # Rotated coordinates for sampling
        theta_rot = (theta_grid[mask] - dtheta) % (2 * np.pi) - np.pi
        r_vals = r_grid[mask]

        # Convert back to Cartesian sampling positions
        x_rot = center_x + r_vals * np.cos(theta_rot)
        y_rot = center_y + r_vals * np.sin(theta_rot)

        # Nearest-neighbor / Bilinear sampling
        x_rot_clamped = np.clip(np.round(x_rot).astype(int), 0, nx - 1)
        y_rot_clamped = np.clip(np.round(y_rot).astype(int), 0, ny - 1)

        surrogate_field[mask] = field[y_rot_clamped, x_rot_clamped]

    return surrogate_field

# ==============================================================================
# LAYER 3: STATISTICAL ESTIMATION
# ==============================================================================

def compute_exact_empirical_p_value(obs_metric: float, null_metrics: np.ndarray, two_tailed: bool = False):
    """Computes exact non-parametric empirical p-value bound: p = (1 + count) / (N + 1)."""
    N = len(null_metrics)
    if two_tailed:
        obs_dev = abs(obs_metric - np.mean(null_metrics))
        null_devs = np.abs(null_metrics - np.mean(null_metrics))
        count = np.sum(null_devs >= obs_dev)
    else:
        count = np.sum(null_metrics >= obs_metric)
        
    p_val = (1.0 + count) / (N + 1.0)
    upper_bound = 1.0 / (N + 1.0)
    
    return p_val, upper_bound

# ==============================================================================
# PIPELINE EXECUTION
# ==============================================================================

def run_suite():
    print("=========================================================================================")
    print("          TRACEBIND PHASE 5B: THREE-LAYERED NULL TESTING SUITE                           ")
    print("=========================================================================================\n")

    nc_files = list(DATA_DIR.glob("era5_*.nc"))
    if not nc_files:
        print("[-] No ERA5 files found. Aborting test.")
        return

    test_file = nc_files[0]
    print(f"[*] Processing ERA5 Frame: {test_file.name}")

    ds = xr.open_dataset(test_file)
    msl_var = "msl" if "msl" in ds.data_vars else list(ds.data_vars.keys())[0]
    frame = ds[msl_var].values[0]
    ds.close()

    rng = np.random.default_rng(42)

    print("\n-----------------------------------------------------------------------------------------")
    print("  LAYER 1: NUMERICAL VALIDATION DIAGNOSTICS                                             ")
    print("-----------------------------------------------------------------------------------------")
    diag = validate_fourier_generator_numerics(frame, rng)
    print(f"  Field Dimensions            : {frame.shape}")
    print(f"  Direct Parseval Rel. Error : {diag['parseval_rel_err']:.4e}  [Status: {'PASS' if diag['parseval_rel_err'] < 1e-12 else 'FAIL'}]")
    print(f"  DC Amplitude Preserved      : {diag['dc_diff']:.4e} Pa [Status: {'PASS' if diag['dc_diff'] < 1e-6 else 'FAIL'}]")
    print("  Spectral Complexity Profile (Cumulative Energy Modes):")
    print(f"    - 90% Spectral Mass       : {diag['modes_90']:5d} / {diag['total_modes']} modes ({diag['modes_90']/diag['total_modes']*100:.2f}%)")
    print(f"    - 95% Spectral Mass       : {diag['modes_95']:5d} / {diag['total_modes']} modes ({diag['modes_95']/diag['total_modes']*100:.2f}%)")
    print(f"    - 99% Spectral Mass       : {diag['modes_99']:5d} / {diag['total_modes']} modes ({diag['modes_99']/diag['total_modes']*100:.2f}%)")

    print("\n-----------------------------------------------------------------------------------------")
    print("  LAYER 2: GENERATING SPECIALIZED NULL MODELS                                            ")
    print("-----------------------------------------------------------------------------------------")
    
    circ_surrogate = generate_circular_radial_surrogate(frame)
    band_surrogate = generate_coherent_band_azimuthal_surrogate(frame, band_width_px=4, rng=rng)

    print("  [✓] Circular Radial Profile Null Model built.")
    print("  [✓] Coherent Radial Band Azimuthal Null Model built (4-px band width).")

    print("\n-----------------------------------------------------------------------------------------")
    print("  LAYER 3: INFERENTIAL METRIC & EMPIRICAL P-VALUE EVALUATION                             ")
    print("-----------------------------------------------------------------------------------------")
    
    # Compute Gradient Energy (Integral of |grad P|^2) as example diagnostic metric
    def compute_gradient_energy(f):
        gy, gx = np.gradient(f)
        return np.sum(gx**2 + gy**2)

    obs_metric = compute_gradient_energy(frame)
    
    # Run Monte Carlo Fourier Phase Null Distribution (N=100)
    null_metrics = []
    for _ in range(100):
        surr = validate_fourier_generator_numerics(frame, rng)["surrogate_field"]
        null_metrics.append(compute_gradient_energy(surr))
    
    null_metrics = np.array(null_metrics)
    p_val, upper_bound = compute_exact_empirical_p_value(obs_metric, null_metrics)

    print(f"  Observed Gradient Energy     : {obs_metric:.4e}")
    print(f"  Null Mean (Fourier Phase)    : {np.mean(null_metrics):.4e} ± {np.std(null_metrics):.4e}")
    print(f"  Empirical Rank p-Value       : p = {p_val:.4f} (Monte Carlo Resolution Limit: p <= {upper_bound:.4f})")
    print("-----------------------------------------------------------------------------------------\n")

if __name__ == "__main__":
    run_suite()