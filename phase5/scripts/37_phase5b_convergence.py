"""
37_phase5b_convergence.py
-----------------------------------------------------
TRACEBIND Phase 5B: Convergence & Statistical Diagnostics Suite (Fixed)
"""

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr, median_abs_deviation
from tracebind_core import compute_row_wise_grid_spacing, compute_reduced_vector, generate_exact_fourier_surrogate
from sklearn.covariance import LedoitWolf

SCRIPTS_DIR = Path(__file__).parent
PHASE5_DIR = SCRIPTS_DIR.parent
DATA_DIR = PHASE5_DIR / "data"
RESULTS_DIR = PHASE5_DIR / "results"

MAHALANOBIS_IN = RESULTS_DIR / "mahalanobis_fourier_null.csv"
CONVERGENCE_OUT = RESULTS_DIR / "running_convergence.csv"
SENSITIVITY_OUT = RESULTS_DIR / "surrogate_sensitivity_study.csv"

def run_surrogate_sensitivity_study(storm_id: str = "amphan", surrogate_counts = [100, 250, 500]):
    print(f"\n[*] Running Surrogate Sample Size Sensitivity Study for storm: '{storm_id}'...")
    
    nc_file = DATA_DIR / f"era5_{storm_id}_72h.nc"
    if not nc_file.exists():
        nc_file = DATA_DIR / f"era5_{storm_id}.nc"
        
    ds = xr.open_dataset(nc_file)
    dx_rows, dy = compute_row_wise_grid_spacing(ds)
    var_name = 'msl' if 'msl' in ds else 'mean_sea_level_pressure'
    field = ds[var_name].values
    field_2d = field[field.shape[0] // 2, :, :] if field.ndim >= 3 else field
    ds.close()

    v_obs = compute_reduced_vector(field_2d, dx_rows=dx_rows, dy=dy)
    records = []

    for n_surr in surrogate_counts:
        surr_vectors = []
        for s_idx in range(n_surr):
            surr_field = generate_exact_fourier_surrogate(field_2d, seed=1000 + s_idx)
            v_surr = compute_reduced_vector(surr_field, dx_rows=dx_rows, dy=dy)
            surr_vectors.append(v_surr)
            
        surr_vectors = np.array(surr_vectors)
        lw = LedoitWolf().fit(surr_vectors)
        d_m = float(lw.mahalanobis(v_obs.reshape(1, -1))[0])
        records.append({"storm_id": storm_id, "N_surrogates": n_surr, "D_M": d_m})
        print(f"  [✓] N={n_surr:3d} Surrogates -> Shrinkage D_M = {d_m:.4f}")

    df_sens = pd.DataFrame(records)
    df_sens.to_csv(SENSITIVITY_OUT, index=False)
    print(f"[*] Sensitivity study exported to: {SENSITIVITY_OUT}")

def run_phase5b_diagnostics():
    if not MAHALANOBIS_IN.exists():
        raise FileNotFoundError("Run 35_eval_fourier_mahalanobis.py first.")

    df = pd.read_csv(MAHALANOBIS_IN)
    tc_df = df[df['cohort_type'].isin(['Core', 'Expanded'])].copy()
    ctrl_df = df[df['cohort_type'] == 'NegativeControl'].copy()
    
    tc_dms = tc_df['D_M_unscaled'].values
    ctrl_dms = ctrl_df['D_M_unscaled'].values

    # 1. Running Convergence Matrix with 95% Bootstrap CIs
    conv_records = []
    np.random.seed(42)
    min_k = min(13, len(tc_dms))
    for i in range(min_k, len(tc_dms) + 1):
        sub_sample = tc_dms[:i]
        
        boot_means = [np.mean(np.random.choice(sub_sample, size=len(sub_sample), replace=True)) for _ in range(1000)]
        ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
        
        conv_records.append({
            "N": i,
            "mean_DM": float(np.mean(sub_sample)),
            "std_DM": float(np.std(sub_sample, ddof=1)) if len(sub_sample) > 1 else 0.0,
            "median_DM": float(np.median(sub_sample)),
            "mad_DM": float(median_abs_deviation(sub_sample)),
            "ci_95_low": float(ci_low),
            "ci_95_high": float(ci_high)
        })
    df_conv = pd.DataFrame(conv_records)
    df_conv.to_csv(CONVERGENCE_OUT, index=False)

    # 2. Defensive Pearson & Spearman Correlations
    valid_p = tc_df.dropna(subset=['min_pressure_hpa', 'D_M_unscaled'])
    valid_v = tc_df.dropna(subset=['max_wind_kt', 'D_M_unscaled'])
    
    if len(valid_p) >= 3:
        r_p, p_p = pearsonr(valid_p['min_pressure_hpa'], valid_p['D_M_unscaled'])
        rho_p, p_sp_p = spearmanr(valid_p['min_pressure_hpa'], valid_p['D_M_unscaled'])
        press_str = f"Pearson r = {r_p:.3f} (p = {p_p:.4f}) | Spearman ρ = {rho_p:.3f} (p = {p_sp_p:.4f})"
    else:
        press_str = "[!] Insufficient pressure metadata observations (N < 3)."

    if len(valid_v) >= 3:
        r_v, p_v = pearsonr(valid_v['max_wind_kt'], valid_v['D_M_unscaled'])
        rho_v, p_sp_v = spearmanr(valid_v['max_wind_kt'], valid_v['D_M_unscaled'])
        wind_str = f"Pearson r = {r_v:.3f} (p = {p_v:.4f}) | Spearman ρ = {rho_v:.3f} (p = {p_sp_v:.4f})"
    else:
        wind_str = "[!] Insufficient wind metadata observations (N < 3)."

    print("\n==========================================================================")
    print("      TRACEBIND PHASE 5B: PUBLICATION-GRADE STATISTICAL MATRIX            ")
    print("==========================================================================")
    print(f" 1. TC Cohort (N={len(tc_dms)}):        Mean D_M = {np.mean(tc_dms):.2f}")
    print(f" 2. Non-TC Controls (N={len(ctrl_dms)}): Mean D_M = {np.mean(ctrl_dms):.2f}" if len(ctrl_dms)>0 else " 2. Non-TC Controls: None")
    print("--------------------------------------------------------------------------")
    print(" 3. Correlation Diagnostics (Physical Intensity Independence):")
    print(f"    - D_M vs Min Pressure: {press_str}")
    print(f"    - D_M vs Max Wind Speed: {wind_str}")
    print("--------------------------------------------------------------------------")
    print(f" 4. Full Convergence Log Saved To: {CONVERGENCE_OUT}")
    print("==========================================================================\n")

if __name__ == "__main__":
    run_surrogate_sensitivity_study()
    run_phase5b_diagnostics()