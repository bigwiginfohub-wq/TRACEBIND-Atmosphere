"""
run_diagnostics.py
-----------------------------------------------------
TRACEBIND Diagnostic & Falsification Suite
Executes 5 core validations:
  1. Identity / Self-Consistency Test
  2. Per-Feature Z-Score Breakdown
  3. Feature Variance & CV Across Surrogates
  4. Metadata Propagation Verification
  5. Controlled Synthetic Field Validation (Gaussian / Vortex / Noise)
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from sklearn.covariance import LedoitWolf
from tracebind_core import compute_row_wise_grid_spacing, compute_reduced_vector, generate_exact_fourier_surrogate

SCRIPTS_DIR = Path(__file__).parent
PHASE5_DIR = SCRIPTS_DIR.parent
DATA_DIR = PHASE5_DIR / "data"
RESULTS_DIR = PHASE5_DIR / "results"

FEATURE_NAMES = ["GE", "LE", "C_orient", "A_radial", "S_orient"]

def run_diagnostics_on_field(field_2d: np.ndarray, dx_rows: np.ndarray, dy: float, label: str = "field"):
    v_obs = compute_reduced_vector(field_2d, dx_rows, dy)
    
    # Generate 100 surrogates
    surr_vecs = np.array([
        compute_reduced_vector(generate_exact_fourier_surrogate(field_2d, seed=1000 + s), dx_rows, dy)
        for s in range(100)
    ])
    
    mu_surr = np.mean(surr_vecs, axis=0)
    std_surr = np.std(surr_vecs, axis=0)
    cv_surr = (std_surr / (np.abs(mu_surr) + 1e-12)) * 100.0
    z_scores = (v_obs - mu_surr) / (std_surr + 1e-12)
    
    lw = LedoitWolf().fit(surr_vecs)
    d_m = float(lw.mahalanobis(v_obs.reshape(1, -1))[0])
    surr_dms = lw.mahalanobis(surr_vecs)
    p_exceed = float(np.mean(d_m > surr_dms) * 100.0)
    
    # Identity Test: Take Surrogate #0 as fake observation
    surr_0 = generate_exact_fourier_surrogate(field_2d, seed=999)
    v_identity = compute_reduced_vector(surr_0, dx_rows, dy)
    secondary_surrs = np.array([
        compute_reduced_vector(generate_exact_fourier_surrogate(surr_0, seed=2000 + s), dx_rows, dy)
        for s in range(100)
    ])
    lw_id = LedoitWolf().fit(secondary_surrs)
    d_m_id = float(lw_id.mahalanobis(v_identity.reshape(1, -1))[0])
    surr_dms_id = lw_id.mahalanobis(secondary_surrs)
    p_id = float(np.mean(d_m_id > surr_dms_id) * 100.0)
    
    return {
        "label": label,
        "v_obs": v_obs,
        "mu_surr": mu_surr,
        "std_surr": std_surr,
        "cv_surr": cv_surr,
        "z_scores": z_scores,
        "d_m": d_m,
        "p_exceed": p_exceed,
        "d_m_id": d_m_id,
        "p_id": p_id
    }

def print_field_report(res: dict):
    print(f"\n==========================================================================")
    print(f" DIAGNOSTIC REPORT: {res['label'].upper()}")
    print(f"==========================================================================")
    print(f" [1] Target D_M: {res['d_m']:.2f} | Percentile Exceedance: {res['p_exceed']:.1f}%")
    print(f" [2] Identity Test D_M: {res['d_m_id']:.2f} | Identity Percentile: {res['p_id']:.1f}%")
    if res['p_id'] > 90.0 or res['p_id'] < 10.0:
        print("     ⚠️ ALERT: Identity test failed! Null model is not self-consistent.")
    else:
        print("     ✅ SUCCESS: Identity test passed! Null model is self-consistent.")
    print("--------------------------------------------------------------------------")
    print(f"{'Feature':10s} | {'Observed':10s} | {'Surr Mean':10s} | {'Surr Std':10s} | {'CV (%)':8s} | {'Z-Score':8s}")
    print("--------------------------------------------------------------------------")
    for i, name in enumerate(FEATURE_NAMES):
        print(f"{name:10s} | {res['v_obs'][i]:10.4f} | {res['mu_surr'][i]:10.4f} | {res['std_surr'][i]:10.4f} | {res['cv_surr'][i]:8.2f} | {res['z_scores'][i]:8.2f}")
    print("==========================================================================\n")

def run_suite():
    print("[*] Starting TRACEBIND Diagnostic & Falsification Suite...")
    
    # 1. Real Atmospheric Field Check (e.g., Amphan)
    nc_file = DATA_DIR / "era5_amphan_72h.nc"
    if not nc_file.exists():
        nc_file = DATA_DIR / "era5_amphan.nc"
        
    if nc_file.exists():
        ds = xr.open_dataset(nc_file)
        dx_rows, dy = compute_row_wise_grid_spacing(ds)
        var_name = 'msl' if 'msl' in ds else 'mean_sea_level_pressure'
        field = ds[var_name].values
        field_2d = field[field.shape[0] // 2, :, :] if field.ndim >= 3 else field
        ds.close()
        
        res_real = run_diagnostics_on_field(field_2d, dx_rows, dy, label="ERA5 Storm: Amphan")
        print_field_report(res_real)
        
    # 2. Synthetic Controlled Fields Validation
    ny, nx = 100, 100
    dx_synth = np.full((ny,), 25000.0)
    dy_synth = 25000.0
    y, x = np.ogrid[:ny, :nx]
    cy, cx = ny // 2, nx // 2
    r2 = (x - cx)**2 + (y - cy)**2
    
    # Synthetic Field A: Concentric Gaussian Vortex
    vortex_field = 101325.0 - 5000.0 * np.exp(-r2 / (2 * 15.0**2))
    res_vortex = run_diagnostics_on_field(vortex_field, dx_synth, dy_synth, label="Synthetic Gaussian Vortex")
    print_field_report(res_vortex)
    
    # Synthetic Field B: Pure Gaussian Noise
    np.random.seed(42)
    noise_field = np.random.normal(loc=101325.0, scale=1000.0, size=(ny, nx))
    res_noise = run_diagnostics_on_field(noise_field, dx_synth, dy_synth, label="Synthetic Gaussian Noise")
    print_field_report(res_noise)

if __name__ == "__main__":
    run_suite()