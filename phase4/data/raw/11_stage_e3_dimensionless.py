"""
11_stage_e3_dimensionless.py
----------------------------
Final Eulerian Feature Refinement:
1. Computes Dimensionless IPMG = IPMG / M_max.
2. Computes Quiescent Baseline from median of first 12 hours to prevent
   window-onset bias.
"""

from pathlib import Path
import pandas as pd
import numpy as np

OUTPUT_COHORT_DIR = Path(r"C:\TRACEBIND-Atmosphere\phase4\data\raw\output_cohort")
SUMMARY_DIR = OUTPUT_COHORT_DIR / "summary"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

STORMS = ["Amphan", "Fani", "Mocha", "Yaas", "Sidr", "Nargis"]
METRICS = ['gradient_energy', 'tb_v2_intensity', 'tb_v2_cosine', 'tb_v1', 'morans_i', 'spatial_entropy']

def find_mslp_column(df: pd.DataFrame) -> str:
    for col in ['min_mslp', 'msl', 'mslp', 'min_mslp_hpa', 'mean_sea_level_pressure']:
        if col in df.columns:
            return col
    raise KeyError(f"No MSLP column found. Present: {list(df.columns)}")

def run_stage_e3():
    records = []

    for storm in STORMS:
        csv_path = OUTPUT_COHORT_DIR / storm / "metrics.csv"
        if not csv_path.exists():
            continue
        
        df = pd.read_csv(csv_path)
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time').reset_index(drop=True)
        mslp_col = find_mslp_column(df)
        
        # Reference Event: Max Deepening
        df['dp_dt'] = np.gradient(df[mslp_col], 1.0)
        idx_max_deep = df['dp_dt'].idxmin()
        t_max_deep = df.loc[idx_max_deep, 'time']

        for m in METRICS:
            df[f'd_{m}_dt'] = np.gradient(df[m], 1.0)
            
            pos_growth = np.maximum(0, df[f'd_{m}_dt'].values)
            trapz_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
            ipmg_raw = trapz_fn(pos_growth, dx=1.0)
            
            m_max = df[m].max()
            
            # 1. Dimensionless IPMG
            ipmg_norm = ipmg_raw / (m_max if m_max != 0 else 1.0)

            # 2. Quiescent Baseline (Median of first 12 hours)
            n_quiescent = min(12, len(df))
            baseline_quiet = df[m].iloc[:n_quiescent].median()
            threshold_quiet = 1.5 * abs(baseline_quiet) if baseline_quiet != 0 else 0.1 * m_max
            elevated_h_quiet = (df[m] >= threshold_quiet).sum()

            # Timing Alignment
            idx_max_growth = df[f'd_{m}_dt'].idxmax()
            t_max_growth = df.loc[idx_max_growth, 'time']
            lead_h = (t_max_deep - t_max_growth).total_seconds() / 3600.0

            records.append({
                'Storm': storm,
                'Metric': m,
                'M_max': round(m_max, 4),
                'IPMG_Raw': round(ipmg_raw, 2),
                'IPMG_Norm': round(ipmg_norm, 3),
                'Quiet_Baseline': round(baseline_quiet, 4),
                'Elevated_h_Quiet': elevated_h_quiet,
                'Lead_vs_MaxDeepening_h': round(lead_h, 1)
            })

    results_df = pd.DataFrame(records)
    results_df.to_csv(SUMMARY_DIR / "stage_e3_dimensionless_features.csv", index=False)

    print("\n=========================================================================================")
    print("             STAGE E-3: DIMENSIONLESS & QUIESCENT BASELINE SUMMARY                      ")
    print("=========================================================================================")
    for m in ['gradient_energy', 'tb_v2_intensity']:
        print(f"\n[ METRIC FOCUS: {m.upper()} ]")
        sub = results_df[results_df['Metric'] == m]
        print(sub[['Storm', 'M_max', 'IPMG_Norm', 'Quiet_Baseline', 'Elevated_h_Quiet', 'Lead_vs_MaxDeepening_h']].to_string(index=False))

if __name__ == "__main__":
    run_stage_e3()