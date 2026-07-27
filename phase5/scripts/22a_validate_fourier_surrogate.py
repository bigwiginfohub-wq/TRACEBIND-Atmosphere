"""
22a_validate_fourier_surrogate.py
---------------------------------
TRACEBIND Phase 5B: Fourier Surrogate Generator Validation Engine
Corrects per-mode spectral diagnostic mask to prevent low-amplitude noise explosion.
"""

import sys
import os
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

BASE_DIR = Path(r"C:\TRACEBIND-Atmosphere\phase5")
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EARTH_RADIUS_M = 6371000.0

def compute_spatial_gradients(field: np.ndarray, lats: np.ndarray, lons: np.ndarray):
    """Computes spatial gradient components (Pa/m) and magnitude."""
    dlat_deg = abs(float(lats[1] - lats[0])) if len(lats) > 1 else 0.25
    dy = np.radians(dlat_deg) * EARTH_RADIUS_M
    gy = np.gradient(field, dy, axis=0)
    
    dlon_deg = abs(float(lons[1] - lons[0])) if len(lons) > 1 else 0.25
    dlon_rad = np.radians(dlon_deg)
    
    gx = np.zeros_like(field)
    for i, lat_deg in enumerate(lats):
        lat_rad = np.radians(lat_deg)
        dx_meters = dlon_rad * EARTH_RADIUS_M * np.cos(lat_rad)
        gx[i, :] = np.gradient(field[i, :], dx_meters)
        
    grad_mag = np.sqrt(gx**2 + gy**2)
    return gx, gy, grad_mag

def generate_hermitian_fourier_surrogate_2d(field: np.ndarray, rng: np.random.Generator):
    """Generates 2D Fourier phase surrogate and exposes pre/post IFFT spectra."""
    ny, nx = field.shape
    orig_mean = np.mean(field)
    orig_std = np.std(field)

    fft_half = np.fft.rfft2(field)
    amplitude = np.abs(fft_half)
    
    random_phase = rng.uniform(-np.pi, np.pi, size=fft_half.shape)
    random_phase[0, 0] = 0.0
    
    if ny % 2 == 0:
        random_phase[ny // 2, 0] = 0.0
    if nx % 2 == 0:
        random_phase[0, nx // 2] = 0.0
    if ny % 2 == 0 and nx % 2 == 0:
        random_phase[ny // 2, nx // 2] = 0.0

    surrogate_fft = amplitude * np.exp(1j * random_phase)
    
    # Pre-IFFT spectrum
    pre_ifft_power = np.abs(surrogate_fft)**2
    
    # Transform to physical domain
    surrogate_field = np.fft.irfft2(surrogate_fft, s=(ny, nx))
    
    # Spatial Variance Conservation Scaling
    surr_std = np.std(surrogate_field)
    if surr_std > 0:
        surrogate_field = (surrogate_field - np.mean(surrogate_field)) / surr_std * orig_std + orig_mean

    # Post-IFFT spectrum
    post_ifft_fft = np.fft.rfft2(surrogate_field)
    post_ifft_power = np.abs(post_ifft_fft)**2

    return surrogate_field, pre_ifft_power, post_ifft_power

def validate_and_plot():
    print("=========================================================================================")
    print("          TRACEBIND PHASE 5B: FOURIER SURROGATE GENERATOR VALIDATION                     ")
    print("=========================================================================================\n")

    nc_files = list(DATA_DIR.glob("era5_*.nc"))
    if not nc_files:
        print("[-] No ERA5 NetCDF files found for validation.")
        return

    test_file = nc_files[0]
    print(f"[*] Validating on test file: {test_file.name}")

    ds = xr.open_dataset(test_file)
    lat_coord = "latitude" if "latitude" in ds.coords else "lat"
    lon_coord = "longitude" if "longitude" in ds.coords else "lon"
    msl_var = "msl" if "msl" in ds.data_vars else list(ds.data_vars.keys())[0]

    lats = ds[lat_coord].values
    lons = ds[lon_coord].values
    frame = ds[msl_var].values[0]
    ds.close()

    rng = np.random.default_rng(42)

    # Compute original metrics
    orig_mean = float(np.mean(frame))
    orig_var = float(np.var(frame))
    orig_power = np.abs(np.fft.rfft2(frame))**2

    # Generate Surrogate
    surr_frame, pre_power, post_power = generate_hermitian_fourier_surrogate_2d(frame, rng)

    # Compute surrogate spatial metrics
    surr_mean = float(np.mean(surr_frame))
    surr_var = float(np.var(surr_frame))

    # Global Relative RMSE metrics
    pre_rel_rmse = float(np.linalg.norm(orig_power - pre_power) / np.linalg.norm(orig_power))
    post_rel_rmse = float(np.linalg.norm(orig_power - post_power) / np.linalg.norm(orig_power))

    # Corrected Per-Mode Error Distribution (Significant Energy Modes: > 1e-4 of max power)
    max_power = np.max(orig_power)
    significant_mask = orig_power > (1e-4 * max_power)
    
    sig_orig = orig_power[significant_mask]
    sig_post = post_power[significant_mask]
    
    rel_errors = np.abs(sig_orig - sig_post) / sig_orig
    
    mean_rel_err = float(np.mean(rel_errors))
    median_rel_err = float(np.median(rel_errors))
    p95_rel_err = float(np.percentile(rel_errors, 95))
    max_rel_err = float(np.max(rel_errors))

    # Energy-Weighted Relative Error Metric across ALL modes
    weighted_rel_err = float(np.sum(np.abs(orig_power - post_power)) / np.sum(orig_power))

    mean_diff = abs(orig_mean - surr_mean)
    var_diff = abs(orig_var - surr_var)

    print("\n-----------------------------------------------------------------------------------------")
    print("                      MATHEMATICAL DIAGNOSTIC REPORT                                     ")
    print("-----------------------------------------------------------------------------------------")
    print(f"  Field Shape                : {frame.shape}")
    print(f"  Original Mean Pressure     : {orig_mean:.6f} Pa")
    print(f"  Surrogate Mean Pressure    : {surr_mean:.6f} Pa")
    print(f"  Mean Difference            : {mean_diff:.4e} Pa (Tolerance: < 1e-4)")
    print("-----------------------------------------------------------------------------------------")
    print(f"  Original Variance          : {orig_var:.6f} Pa^2")
    print(f"  Surrogate Variance         : {surr_var:.6f} Pa^2")
    print(f"  Variance Difference        : {var_diff:.4e} Pa^2 (Tolerance: < 1e-2)")
    print("-----------------------------------------------------------------------------------------")
    print(f"  Pre-IFFT Spectrum RMSE     : {pre_rel_rmse:.4e} (Tolerance: < 1e-8)")
    print(f"  Post-IFFT Spectrum RMSE    : {post_rel_rmse:.4e} (Tolerance: < 1e-5)")
    print("-----------------------------------------------------------------------------------------")
    print(f"  Energy-Weighted Relative Error : {weighted_rel_err:.4e}")
    print(f"  Mode Error Distribution (Power > 0.01% of Max):")
    print(f"    - Evaluated Modes        : {np.sum(significant_mask)} / {orig_power.size}")
    print(f"    - Mean Relative Error    : {mean_rel_err:.4e}")
    print(f"    - Median Relative Error  : {median_rel_err:.4e}")
    print(f"    - 95th Percentile Error  : {p95_rel_err:.4e}")
    print(f"    - Max Relative Error     : {max_rel_err:.4e}")
    print("-----------------------------------------------------------------------------------------\n")

    # Assertions
    assert mean_diff < 1e-4, f"Mean conservation failed: {mean_diff}"
    assert var_diff < 1e-2, f"Variance conservation failed: {var_diff}"
    assert pre_rel_rmse < 1e-8, f"Pre-IFFT spectral generation altered amplitudes: {pre_rel_rmse}"
    assert post_rel_rmse < 1e-5, f"Post-IFFT spectral drift exceeded threshold: {post_rel_rmse}"

    print("[✓] PASSED: Fourier Surrogate Generator verified under robust error diagnostics!")

if __name__ == "__main__":
    validate_and_plot()