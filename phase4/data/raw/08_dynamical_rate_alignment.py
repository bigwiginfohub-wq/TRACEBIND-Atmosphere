"""
08_dynamical_rate_alignment.py
--------------------------------
Stage D-2 Dynamical Rate & Feature Alignment Suite:
1. Calculates rate of change d(Metric)/dt using centered finite differences.
2. Unclipped windowing with boundary-censoring detection (flags hitting domain edge).
3. Relaxed RI Onset: dP/dt <= -0.75 hPa/h (-18 hPa/day) over 3h persistence.
4. Chronological UTC event reporting for manual visual sanity check.
5. Statistical summary with Min/Max ranges alongside Mean/Median/Std.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np

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

# Relaxed RI Threshold to capture -18 hPa/day in regional domain windows
RI_THRESHOLD_HPA_PER_H = -0.75 
RI_PERSISTENCE_HOURS = 3

def find_mslp_column(df: pd.DataFrame) -> str:
    candidates = ['min_mslp', 'msl', 'mslp', 'min_mslp_hpa', 'mean_sea_level_pressure']
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"No MSLP column found in DataFrame. Columns present: {list(df.columns)}")

def run_dynamical_alignment():
    chronology_records = []
    lag_records = []

    for storm in STORMS:
        csv_path = OUTPUT_COHORT_DIR / storm / "metrics.csv"
        if not csv_path.exists():
            print(f"[!] Warning: {csv_path} not found. Skipping.")
            continue
        
        df = pd.read_csv(csv_path)
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time').reset_index(drop=True)
        
        mslp_col = find_mslp_column(df)
        
        # 1. Physical Pressure Derivatives (dP/dt)
        df['dp_dt'] = np.gradient(df[mslp_col], 1.0)  # hPa / hour
        
        # Key Atmospheric Events
        idx_min_mslp = df[mslp_col].idxmin()
        t_min_mslp = df.loc[idx_min_mslp, 'time']
        min_mslp_val = df.loc[idx_min_mslp, mslp_col]
        
        idx_max_deep = df['dp_dt'].idxmin()
        t_max_deep = df.loc[idx_max_deep, 'time']
        max_deep_val = df.loc[idx_max_deep, 'dp_dt']
        
        # RI Onset Detection
        is_deepening = (df['dp_dt'] <= RI_THRESHOLD_HPA_PER_H).astype(int)
        persistent_deep = is_deepening.rolling(window=RI_PERSISTENCE_HOURS).sum()
        ri_matches = df[persistent_deep >= RI_PERSISTENCE_HOURS]
        
        if len(ri_matches) > 0:
            first_ri_end_idx = ri_matches.index[0]
            first_ri_start_idx = max(0, first_ri_end_idx - (RI_PERSISTENCE_HOURS - 1))
            t_ri_onset = df.loc[first_ri_start_idx, 'time']
        else:
            t_ri_onset = pd.NaT

        # 2. Metric Rate of Change Derivatives d(Metric)/dt
        chron_entry = {
            'Storm': storm,
            'Min_MSLP_hPa': round(min_mslp_val, 1),
            'Min_MSLP_UTC': t_min_mslp.strftime('%m-%d %H:%M'),
            'Max_Deepening_hPa_h': round(max_deep_val, 2),
            'Max_Deepening_UTC': t_max_deep.strftime('%m-%d %H:%M'),
            'RI_Onset_UTC': t_ri_onset.strftime('%m-%d %H:%M') if pd.notna(t_ri_onset) else 'N/A'
        }

        for metric in METRICS_TO_ANALYZE:
            df[f'd_{metric}_dt'] = np.gradient(df[metric], 1.0)
            
            # Find maximum growth rate (inflection point)
            idx_max_rate = df[f'd_{metric}_dt'].idxmax()
            t_max_rate = df.loc[idx_max_rate, 'time']
            
            chron_entry[f'{metric}_MaxGrowth_UTC'] = t_max_rate.strftime('%m-%d %H:%M')
            
            # Calculate Lead Times relative to Max Deepening Rate (t_max_deep)
            # Lead = t_max_deep - t_max_growth (Positive = Metric Growth led Max Deepening)
            lead_vs_deepening_h = (t_max_deep - t_max_rate).total_seconds() / 3600.0
            
            # Censoring Check: Did max growth occur on boundary frames (first or last frame)?
            is_censored = (idx_max_rate == 0) or (idx_max_rate == len(df) - 1)
            
            lag_records.append({
                'Storm': storm,
                'Metric': metric,
                'Max_Growth_UTC': t_max_rate,
                'Max_Deepening_UTC': t_max_deep,
                'Lead_vs_MaxDeepening_h': lead_vs_deepening_h,
                'Boundary_Censored': is_censored
            })

        chronology_records.append(chron_entry)

    # -------------------------------------------------------------------------
    # OUTPUT 1: Chronological Event Audit Table
    # -------------------------------------------------------------------------
    chron_df = pd.DataFrame(chronology_records)
    chron_csv = SUMMARY_DIR / "stage_d2_chronology_audit.csv"
    chron_df.to_csv(chron_csv, index=False)
    
    print("\n=========================================================================================")
    print("                    STAGE D-2 CHRONOLOGICAL EVENT AUDIT TABLE                            ")
    print("=========================================================================================")
    print(chron_df[['Storm', 'Min_MSLP_UTC', 'Max_Deepening_UTC', 'RI_Onset_UTC', 'gradient_energy_MaxGrowth_UTC', 'tb_v2_intensity_MaxGrowth_UTC']].to_string(index=False))

    # -------------------------------------------------------------------------
    # OUTPUT 2: Rate-of-Change Lead/Lag Analysis & Censoring Report
    # -------------------------------------------------------------------------
    lag_df = pd.DataFrame(lag_records)
    lag_csv = SUMMARY_DIR / "stage_d2_rate_alignment_lags.csv"
    lag_df.to_csv(lag_csv, index=False)

    print("\n=========================================================================================")
    print("           DYNAMICAL ALIGNMENT: MAX d(METRIC)/dt vs. MAX DEEPENING RATE (-dP/dt)          ")
    print("=========================================================================================")
    
    stat_summary = []
    for metric in METRICS_TO_ANALYZE:
        sub = lag_df[lag_df['Metric'] == metric]
        leads = sub['Lead_vs_MaxDeepening_h'].values
        censored_count = sub['Boundary_Censored'].sum()
        
        stat_summary.append({
            'Metric': metric,
            'Mean_Lead_h': round(np.mean(leads), 1),
            'Median_Lead_h': round(np.median(leads), 1),
            'StdDev_h': round(np.std(leads), 1),
            'Min_Lead_h': round(np.min(leads), 1),
            'Max_Lead_h': round(np.max(leads), 1),
            'Censored_Frames': f"{censored_count}/{len(sub)}"
        })

    summary_df = pd.DataFrame(stat_summary)
    summary_csv = SUMMARY_DIR / "stage_d2_rate_alignment_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    
    print(summary_df.to_string(index=False))
    print("=========================================================================================\n")
    print("[*] Note: Positive Lead (h) means metric growth rate peaked BEFORE maximum deepening rate.")

if __name__ == "__main__":
    run_dynamical_alignment()