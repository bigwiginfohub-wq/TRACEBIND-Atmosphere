"""
07_multi_reference_alignment.py
--------------------------------
Stage D-1 Multi-Reference Alignment Suite:
1. Dynamic detection of 'min_mslp'.
2. Centered difference derivative dP/dt via np.gradient for hourly data.
3. Event Definitions:
   - REF 1: Min MSLP (tau_mslp = 0 h)
   - REF 2: Max Deepening Rate (min dP/dt) (tau_deep = 0 h)
   - REF 3: RI Onset (first frame of >= 3 consecutive hours with dP/dt <= -1.25 hPa/h)
4. Evaluation of peak lead/lag distributions for all metrics across all 3 references.
5. Saves comparative summary CSV and console report.
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

# RI Threshold: -1.25 hPa/hour (~30 hPa / 24 hour standard RI definition)
RI_THRESHOLD_HPA_PER_H = -1.25 
RI_PERSISTENCE_HOURS = 3

def find_mslp_column(df: pd.DataFrame) -> str:
    """Detects MSLP column across variable convention variations."""
    candidates = ['min_mslp', 'msl', 'mslp', 'min_mslp_hpa', 'mean_sea_level_pressure']
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"No MSLP column found in DataFrame. Columns present: {list(df.columns)}")

def extract_reference_events(df: pd.DataFrame, mslp_col: str):
    """
    Computes timestamps for:
    1. Min MSLP
    2. Max Deepening Rate (min dP/dt)
    3. RI Onset (First 3-hour persistent dP/dt <= threshold)
    """
    df = df.sort_values('time').reset_index(drop=True)
    
    # Hourly centered-difference pressure tendency
    df['dp_dt'] = np.gradient(df[mslp_col], 1.0) # hPa / hour
    
    # Event 1: Minimum MSLP
    idx_min_mslp = df[mslp_col].idxmin()
    t_min_mslp = df.loc[idx_min_mslp, 'time']
    
    # Event 2: Max Deepening Rate (most negative dP/dt)
    idx_max_deep = df['dp_dt'].idxmin()
    t_max_deep = df.loc[idx_max_deep, 'time']
    
    # Event 3: RI Onset with 3-hour persistence guard
    is_deepening = (df['dp_dt'] <= RI_THRESHOLD_HPA_PER_H).astype(int)
    # Rolling sum over 3 consecutive hours
    persistent_deep = is_deepening.rolling(window=RI_PERSISTENCE_HOURS).sum()
    
    ri_matches = df[persistent_deep >= RI_PERSISTENCE_HOURS]
    if len(ri_matches) > 0:
        # First timestamp fulfilling 3-hour persistence (adjust to start of 3h window)
        first_ri_end_idx = ri_matches.index[0]
        first_ri_start_idx = max(0, first_ri_end_idx - (RI_PERSISTENCE_HOURS - 1))
        t_ri_onset = df.loc[first_ri_start_idx, 'time']
    else:
        # Fallback if storm does not hit persistent -1.25 hPa/h threshold
        t_ri_onset = np.nan

    return t_min_mslp, t_max_deep, t_ri_onset, df

def run_multi_reference_analysis():
    records = []

    for storm in STORMS:
        csv_path = OUTPUT_COHORT_DIR / storm / "metrics.csv"
        if not csv_path.exists():
            print(f"[!] Warning: {csv_path} not found. Skipping.")
            continue
        
        df = pd.read_csv(csv_path)
        df['time'] = pd.to_datetime(df['time'])
        
        mslp_col = find_mslp_column(df)
        
        if len(df) < 40:
            print(f"[!] Skipping {storm}: window depth too shallow ({len(df)} frames).")
            continue

        t_min_mslp, t_max_deep, t_ri_onset, df_proc = extract_reference_events(df, mslp_col)

        references = {
            'Min MSLP': t_min_mslp,
            'Max Deepening': t_max_deep,
            'RI Onset': t_ri_onset
        }

        for ref_name, t_ref in references.items():
            if pd.isna(t_ref):
                row = {'Storm': storm, 'Reference_Event': ref_name, 'Status': 'No RI Threshold Met'}
                for m in METRICS_TO_ANALYZE:
                    row[f'{m}_lead_h'] = np.nan
                records.append(row)
                continue

            df_proc['tau'] = (df_proc['time'] - t_ref).dt.total_seconds() / 3600.0
            df_win = df_proc[(df_proc['tau'] >= -36) & (df_proc['tau'] <= 36)].copy()

            row = {'Storm': storm, 'Reference_Event': ref_name, 'Status': 'Valid'}
            for metric in METRICS_TO_ANALYZE:
                if len(df_win) > 0 and metric in df_win.columns:
                    # Tau relative to reference where metric achieves maximum
                    peak_tau = df_win.loc[df_win[metric].idxmax(), 'tau']
                    row[f'{metric}_lead_h'] = round(peak_tau, 1)
                else:
                    row[f'{metric}_lead_h'] = np.nan

            records.append(row)

    res_df = pd.DataFrame(records)
    out_csv = SUMMARY_DIR / "stage_d1_multi_reference_summary.csv"
    res_df.to_csv(out_csv, index=False)
    print(f"[✓] Multi-Reference Summary CSV Saved: {out_csv}\n")

    # Display clean table report
    print("=========================================================================================")
    print("                    STAGE D-1 MULTI-REFERENCE ALIGNMENT REPORT                          ")
    print("=========================================================================================")
    cols_to_print = ['Storm', 'Reference_Event', 'Status'] + [f'{m}_lead_h' for m in METRICS_TO_ANALYZE]
    print(res_df[cols_to_print].to_string(index=False))

    # Evaluate Variance / Stability Across References
    print("\n-----------------------------------------------------------------------------------------")
    print("                   METRIC LEAD/LAG STABILITY ACROSS REFERENCES                           ")
    print("-----------------------------------------------------------------------------------------")
    valid_df = res_df[res_df['Status'] == 'Valid']
    
    for ref_name in ['Min MSLP', 'Max Deepening', 'RI Onset']:
        sub = valid_df[valid_df['Reference_Event'] == ref_name]
        print(f"\n[ Reference Event: {ref_name} ] (N = {len(sub)} storms)")
        for metric in METRICS_TO_ANALYZE:
            vals = sub[f'{metric}_lead_h'].dropna()
            if len(vals) > 0:
                mean_l = np.mean(vals)
                std_l = np.std(vals)
                median_l = np.median(vals)
                print(f"  • {metric:<16}: Mean = {mean_l:6.1f} h | Median = {median_l:6.1f} h | StdDev = {std_l:5.1f} h")

if __name__ == "__main__":
    run_multi_reference_analysis()