"""
15_storm_morphology_analysis.py
-------------------------------
TRACEBIND Phase 5 Post-Diagnostic Analysis
- Merges Phase 5A and Phase 5B outputs with external storm_metadata.csv
- Evaluates exploratory rank correlations between Phase 5B metrics and storm properties
- Exports clean summary tables using precise statistical terminology
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

RESULTS_DIR = Path(r"C:\TRACEBIND-Atmosphere\phase5\results")

def run_morphology_analysis():
    print("=========================================================================================")
    print("        TRACEBIND POST-DIAGNOSTIC: METADATA MERGE & EXPLORATORY ASSOCIATIONS            ")
    print("=========================================================================================\n")

    p5a_path   = RESULTS_DIR / "phase5a_spatial_qc_summary.csv"
    p5b_path   = RESULTS_DIR / "phase5b_temporal_falsification_summary.csv"
    meta_path  = RESULTS_DIR / "storm_metadata.csv"

    if not all(p.exists() for p in [p5a_path, p5b_path, meta_path]):
        print("[-] Error: One or more required CSV input files are missing in results directory.")
        return

    df_a    = pd.read_csv(p5a_path)
    df_b    = pd.read_csv(p5b_path)
    df_meta = pd.read_csv(meta_path)

    # 1. Merge datasets cleanly on 'Storm'
    merged = pd.merge(df_a[["Storm", "Cohens_d", "p_value_left"]], 
                      df_b[["Storm", "Peak_GE_Hour", "Rate_Cohens_d", "Trajectory_p_value", "Precursor_Falsification_Passed"]], 
                      on="Storm")
    
    full_df = pd.merge(merged, df_meta, on="Storm")

    # Rename to clean statistical terminology
    full_df.rename(columns={
        "p_value_left": "P5A_p_val",
        "Cohens_d": "P5A_Cohen_d",
        "Trajectory_p_value": "P5B_p_val",
        "Rate_Cohens_d": "P5B_Rate_Cohen_d",
        "Precursor_Falsification_Passed": "P5B_Significant"
    }, inplace=True)

    # Output merged morphology matrix
    summary_cols = [
        "Storm", "Category", "Vmax_kt", "RI_24h_kt", "Translation_Speed_kmh",
        "P5A_p_val", "P5B_p_val", "P5B_Rate_Cohen_d", "P5B_Significant"
    ]
    
    print("--- [1] Merged Sensitivity & Physical Metadata Table ---")
    print(full_df[summary_cols].to_string(index=False))
    
    out_matrix_path = RESULTS_DIR / "phase5_integrated_morphology_matrix.csv"
    full_df.to_csv(out_matrix_path, index=False)
    print(f"\n[+] Saved integrated matrix to: {out_matrix_path}\n")

    # 2. Exploratory Rank Associations (N=6 Cohort)
    print("--- [2] Exploratory Spearman Rank Associations (N=6 Cohort) ---")
    
    associations = [
        ("P5B_Rate_Cohen_d", "RI_24h_kt", "Trajectory Effect Size vs. 24h Intensification Rate"),
        ("P5B_Rate_Cohen_d", "Vmax_kt", "Trajectory Effect Size vs. Maximum Wind Speed"),
        ("P5B_Rate_Cohen_d", "Translation_Speed_kmh", "Trajectory Effect Size vs. Translation Speed"),
        ("P5B_p_val", "RI_24h_kt", "Trajectory p-value vs. 24h Intensification Rate")
    ]

    assoc_results = []
    for var1, var2, desc in associations:
        rho, p_val = spearmanr(full_df[var1], full_df[var2])
        assoc_results.append({
            "Comparison": desc,
            "Spearman_rho": round(rho, 3),
            "p_val": round(p_val, 3)
        })

    assoc_df = pd.DataFrame(assoc_results)
    print(assoc_df.to_string(index=False))
    
    assoc_out_path = RESULTS_DIR / "phase5_exploratory_associations.csv"
    assoc_df.to_csv(assoc_out_path, index=False)
    print(f"\n[+] Saved exploratory associations to: {assoc_out_path}\n")

if __name__ == "__main__":
    run_morphology_analysis()