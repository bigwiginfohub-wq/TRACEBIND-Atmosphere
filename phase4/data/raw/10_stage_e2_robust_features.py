"""
10_stage_e2_robust_features.py
--------------------------------
Stage E-2 Robust Structural Feature Extraction Suite:
1. Replaces fragile single-frame .max() with robust feature descriptors:
   - Integrated Positive Metric Growth (IPMG): Area under positive dM/dt curve.
   - 95th Percentile Growth Rate: Noise-resistant rate descriptor.
   - Elevated Signal Duration: Hours spent above 75th percentile baseline.
2. Neutralized Nomenclature: 'Temporal Alignment Class' (A: Synced, B: Precursory, C: Post-Peak).
3. Explicit Lead/Lag convention: Lead_h = (t_reference - t_feature).
4. Generates comprehensive Integrated Feature Audit Table.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np

OUTPUT_COHORT_DIR = Path(r"C:\TRACEBIND-Atmosphere\phase4\data\raw\output_cohort")
SUMMARY_DIR = OUTPUT_COHORT_DIR / "summary"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

STORMS = ["Amphan", "Fani", "Mocha", "Yaas", "Sidr", "Nargis"]
METRICS = ['gradient_energy', 'tb_v2_intensity', 'tb_v2_cosine', 'tb_v1', 'morans_i', 'spatial_entropy']

def find_mslp_column(df: pd.DataFrame) -> str:
    candidates = ['min_mslp', 'msl', 'mslp', 'min_mslp_hpa', 'mean_sea_level_pressure']
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"No MSLP column found in DataFrame. Columns present: {list(df.columns)}")

def run_robust_feature_extraction():
    records = []

    for storm in STORMS:
        csv_path = OUTPUT_COHORT_DIR / storm / "metrics.csv"
        if not csv_path.exists():
            print(f"[!] Warning: {csv_path} not found. Skipping.")
            continue
        
        df = pd.read_csv(csv_path)
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time').reset_index(drop=True)
        mslp_col = find_mslp_column(df)
        
        # Pressure Derivative & Maximum Deepening Reference
        df['dp_dt'] = np.gradient(df[mslp_col], 1.0)
        idx_max_deep = df['dp_dt'].idxmin()
        t_max_deep = df.loc[idx_max_deep, 'time']
        max_deep_rate = abs(df.loc[idx_max_deep, 'dp_dt'])

        for m in METRICS:
            df[f'd_{m}_dt'] = np.gradient(df[m], 1.0)
            
            # Feature 1: Integrated Positive Metric Growth (IPMG)
            pos_growth = np.maximum(0, df[f'd_{m}_dt'].values)
            ipmg = getattr(np, 'trapezoid', getattr(np, 'trapz', None))(pos_growth, dx=1.0) # Numerical integration over hourly steps
            
            # Feature 2: 95th Percentile Growth Rate (Noise Resistant)
            p95_growth_rate = np.percentile(df[f'd_{m}_dt'], 95)
            
            # Feature 3: Elevated Signal Duration (> 75th percentile of metric magnitude)
            p75_baseline = np.percentile(df[m], 75)
            elevated_hours = (df[m] >= p75_baseline).sum()
            
            # Timing Feature: Peak Growth Timestamp (Inflection Point)
            idx_max_growth = df[f'd_{m}_dt'].idxmax()
            t_max_growth = df.loc[idx_max_growth, 'time']
            
            # Explicit Lead Convention
            lead_h = (t_max_deep - t_max_growth).total_seconds() / 3600.0

            # Neutralized Alignment Classification
            if abs(lead_h) <= 6.0:
                align_class = "Temporal Class A (Synced)"
            elif lead_h > 6.0:
                align_class = "Temporal Class B (Precursory)"
            else:
                align_class = "Temporal Class C (Post-Peak Lag)"

            records.append({
                'Storm': storm,
                'Metric': m,
                'Max_Deepening_Rate_hPa_h': round(max_deep_rate, 2),
                'Integrated_Growth_IPMG': round(ipmg, 4),
                'Growth_Rate_p95': round(p95_growth_rate, 5),
                'Elevated_Duration_h': elevated_hours,
                'Lead_vs_MaxDeepening_h': round(lead_h, 1),
                'Alignment_Class': align_class
            })

    results_df = pd.DataFrame(records)
    out_csv = SUMMARY_DIR / "stage_e2_robust_features.csv"
    results_df.to_csv(out_csv, index=False)

    # -------------------------------------------------------------------------
    # PRINT SUMMARY AUDIT REPORT
    # -------------------------------------------------------------------------
    print("\n=========================================================================================")
    print("               STAGE E-2: ROBUST FEATURE EXTRACTION AUDIT REPORT                          ")
    print("=========================================================================================")
    
    for m in METRICS:
        print(f"\n[ METRIC: {m.upper()} ]")
        sub = results_df[results_df['Metric'] == m]
        print(sub[['Storm', 'Integrated_Growth_IPMG', 'Growth_Rate_p95', 'Elevated_Duration_h', 'Lead_vs_MaxDeepening_h', 'Alignment_Class']].to_string(index=False))

    print("\n=========================================================================================")
    print(" Saved: stage_e2_robust_features.csv")
    print("=========================================================================================\n")

if __name__ == "__main__":
    run_robust_feature_extraction()