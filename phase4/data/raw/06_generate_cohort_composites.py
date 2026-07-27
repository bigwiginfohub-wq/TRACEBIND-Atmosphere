"""
06_generate_cohort_composites.py
---------------------------------
Stage C Validation Suite:
1. Dynamic MSLP column detection (including 'min_mslp').
2. Centering on Minimum MSLP (tau = 0 h).
3. Primary Z-score Standardization (z = (x - mean)/std) & Secondary Min-Max Sensitivity.
4. Window completeness guard (minimum 40 valid frames required).
5. Dynamic sample size n(tau) tracking and composite envelope generation.
6. Non-parametric hypothesis testing (Wilcoxon Signed-Rank) for lead/lag significance.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

OUTPUT_COHORT_DIR = Path(r"C:\TRACEBIND-Atmosphere\phase4\data\raw\output_cohort")
SUMMARY_DIR = OUTPUT_COHORT_DIR / "summary"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

STORMS = ["Amphan", "Fani", "Mocha", "Yaas", "Sidr", "Nargis"]
METRICS_TO_ANALYZE = [
    'gradient_energy', 
    'tb_v2_intensity', 
    'tb_v2_cosine', 
    'tb_v1', 
    'morans_i', 
    'spatial_entropy'
]

TAU_RANGE = np.arange(-36, 37, 1)  # Window: -36h to +36h around MSLP min

def find_mslp_column(df: pd.DataFrame) -> str:
    """Robustly detects MSLP column name across variable convention variations."""
    candidates = ['min_mslp', 'msl', 'mslp', 'min_mslp_hpa', 'mean_sea_level_pressure']
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"No MSLP column found in DataFrame. Columns present: {list(df.columns)}")

def compute_zscore(series: pd.Series) -> pd.Series:
    """Computes standard z-score (x - mu) / sigma."""
    std = series.std()
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std

def compute_minmax(series: pd.Series) -> pd.Series:
    """Computes min-max scaling [0, 1] for sensitivity analysis."""
    s_min, s_max = series.min(), series.max()
    if s_max == s_min:
        return pd.Series(0.0, index=series.index)
    return (series - s_min) / (s_max - s_min)

def process_cohort_stage_c():
    aligned_zscore = {m: [] for m in METRICS_TO_ANALYZE}
    aligned_minmax = {m: [] for m in METRICS_TO_ANALYZE}
    peak_leads_h = {m: [] for m in METRICS_TO_ANALYZE}

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Data ingestion & alignment loop
    for storm in STORMS:
        csv_path = OUTPUT_COHORT_DIR / storm / "metrics.csv"
        if not csv_path.exists():
            print(f"[!] Warning: {csv_path} not found. Skipping.")
            continue
        
        df = pd.read_csv(csv_path)
        df['time'] = pd.to_datetime(df['time'])
        
        # 1. Dynamic MSLP column detection
        mslp_col = find_mslp_column(df)
        
        # 2. Window completeness guard
        if len(df) < 40:
            print(f"[!] Skipping {storm}: insufficient window depth ({len(df)} frames < 40 minimum).")
            continue

        # 3. Align time to Minimum MSLP (tau = 0)
        min_mslp_idx = df[mslp_col].idxmin()
        t_mslp = df.loc[min_mslp_idx, 'time']
        df['tau'] = (df['time'] - t_mslp).dt.total_seconds() / 3600.0

        # Filter to tau window [-36, +36]
        df_win = df[(df['tau'] >= -36) & (df['tau'] <= 36)].sort_values('tau').copy()

        for metric in METRICS_TO_ANALYZE:
            # Standardize & Scale
            df_win[f'{metric}_z'] = compute_zscore(df_win[metric])
            df_win[f'{metric}_mm'] = compute_minmax(df_win[metric])

            # Interpolate to uniform TAU_RANGE
            z_interp = np.interp(TAU_RANGE, df_win['tau'], df_win[f'{metric}_z'], left=np.nan, right=np.nan)
            mm_interp = np.interp(TAU_RANGE, df_win['tau'], df_win[f'{metric}_mm'], left=np.nan, right=np.nan)
            
            aligned_zscore[metric].append(z_interp)
            aligned_minmax[metric].append(mm_interp)

            # Record exact peak lead/lag time relative to MSLP min (tau_peak)
            peak_tau = df_win.loc[df_win[metric].idxmax(), 'tau']
            peak_leads_h[metric].append(peak_tau)

    # -------------------------------------------------------------------------
    # OUTPUT 1: Z-Score Storm Overlays
    # -------------------------------------------------------------------------
    fig_overlay, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
    axes_flat = axes.flatten()

    for idx, metric in enumerate(METRICS_TO_ANALYZE):
        ax = axes_flat[idx]
        matrix = np.array(aligned_zscore[metric])
        for s_idx, storm_name in enumerate(STORMS[:matrix.shape[0]]):
            ax.plot(TAU_RANGE, matrix[s_idx], alpha=0.6, linewidth=1.5, label=storm_name)
        
        ax.axvline(0, color='black', linestyle='--', alpha=0.8, label='Min MSLP (\u03c4=0)')
        ax.set_title(f"Z-Score Trajectories: {metric.upper()}", fontsize=11, fontweight='bold')
        ax.set_ylabel("Standardized Anomaly (z)")
        ax.set_xlim(-36, 36)
        if idx >= 4:
            ax.set_xlabel("Hours Relative to Min MSLP (\u03c4)")
        ax.legend(loc='upper left', fontsize=8, frameon=True)

    plt.tight_layout()
    overlay_path = SUMMARY_DIR / "stage_c_zscore_overlays.png"
    plt.savefig(overlay_path, dpi=300)
    plt.close()
    print(f"[✓] Output 1 Saved: {overlay_path}")

    # -------------------------------------------------------------------------
    # OUTPUT 2: Composite Ensembles (Mean ± 1σ & Dynamic n(τ))
    # -------------------------------------------------------------------------
    fig_comp, axes_comp = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
    axes_comp_flat = axes_comp.flatten()
    
    comp_df = pd.DataFrame({'tau_h': TAU_RANGE})

    for idx, metric in enumerate(METRICS_TO_ANALYZE):
        ax = axes_comp_flat[idx]
        matrix = np.array(aligned_zscore[metric])
        
        # Calculate dynamic valid count, mean, and std
        n_valid = np.sum(~np.isnan(matrix), axis=0)
        mean_curve = np.nanmean(matrix, axis=0)
        std_curve = np.nanstd(matrix, axis=0)

        comp_df[f'{metric}_n'] = n_valid
        comp_df[f'{metric}_mean_z'] = mean_curve
        comp_df[f'{metric}_std_z'] = std_curve

        # Plot Ensemble
        ax.plot(TAU_RANGE, mean_curve, color='navy', linewidth=2.5, label='Composite Mean (z)')
        ax.fill_between(TAU_RANGE, mean_curve - std_curve, mean_curve + std_curve, color='blue', alpha=0.2, label='\u00b11 Std Dev')
        ax.axvline(0, color='red', linestyle='--', alpha=0.8, label='Min MSLP (\u03c4=0)')
        
        ax.set_title(f"Composite Ensemble: {metric.upper()}", fontsize=11, fontweight='bold')
        ax.set_ylabel("Composite z-score")
        ax.set_xlim(-36, 36)
        if idx >= 4:
            ax.set_xlabel("Hours Relative to Min MSLP (\u03c4)")
        ax.legend(loc='upper left', fontsize=8, frameon=True)

    plt.tight_layout()
    comp_path = SUMMARY_DIR / "stage_c_composite_ensembles.png"
    plt.savefig(comp_path, dpi=300)
    plt.close()
    print(f"[✓] Output 2 Saved: {comp_path}")
    
    # Save composite curves CSV
    comp_csv = SUMMARY_DIR / "stage_c_composite_data.csv"
    comp_df.to_csv(comp_csv, index=False)
    print(f"[✓] Composite CSV Saved: {comp_csv}")

    # -------------------------------------------------------------------------
    # OUTPUT 3: Lead/Lag Non-Parametric Statistical Hypothesis Testing
    # -------------------------------------------------------------------------
    stat_records = []
    for metric in METRICS_TO_ANALYZE:
        leads = np.array(peak_leads_h[metric])
        mean_lead = np.mean(leads)
        median_lead = np.median(leads)
        q25, q75 = np.percentile(leads, [25, 75])
        
        # Non-parametric Wilcoxon Signed-Rank Test (H0: Median Lead = 0)
        try:
            if np.all(leads == 0):
                p_val = 1.0
            else:
                res = stats.wilcoxon(leads, alternative='two-sided')
                p_val = res.pvalue
        except Exception:
            p_val = np.nan

        stat_records.append({
            'Metric': metric,
            'Mean_Lead_h': round(mean_lead, 2),
            'Median_Lead_h': round(median_lead, 2),
            'IQR_25_75_h': f"[{round(q25, 1)}, {round(q75, 1)}]",
            'Wilcoxon_p_value': round(p_val, 4) if not np.isnan(p_val) else "N/A",
            'Significant_p05': "Yes" if p_val < 0.05 else "No"
        })

    stat_df = pd.DataFrame(stat_records)
    stat_csv = SUMMARY_DIR / "stage_c_lead_lag_statistics.csv"
    stat_df.to_csv(stat_csv, index=False)
    print(f"[✓] Output 3 Saved: {stat_csv}")
    
    print("\n================================================================================")
    print("                      STAGE C LEAD/LAG STATISTICAL SUMMARY                      ")
    print("================================================================================")
    print(stat_df.to_string(index=False))

if __name__ == "__main__":
    process_cohort_stage_c()