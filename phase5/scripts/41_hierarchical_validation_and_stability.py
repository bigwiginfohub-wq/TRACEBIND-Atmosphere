"""
41_hierarchical_validation_and_stability.py
-----------------------------------------------------
TRACEBIND Phase 5D: Descriptor Taxonomy & Hierarchical Validation
Features:
- Bootstrapped 95% Confidence Intervals (1,000 iterations)
- Leave-One-Storm-Out (LOSO) Rank Stability (Spearman's Rho)
- Inter-Descriptor Correlation Matrix (Redundancy vs Complementarity)
- Principal Component Analysis (Unsupervised Proof of Tiered Architecture)
- Non-Parametric Hierarchical Subspace Evaluation (No Covariance/Mahalanobis)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

SCRIPTS_DIR = Path(__file__).parent
PHASE5_DIR = SCRIPTS_DIR.parent
RESULTS_DIR = PHASE5_DIR / "results"

IN_PROFILES = RESULTS_DIR / "zscore_profiles_rigorous.csv"
OUT_BOOTSTRAP = RESULTS_DIR / "bootstrapped_descriptors_ci.csv"
OUT_LOSO = RESULTS_DIR / "loso_rank_stability.csv"
OUT_CORRELATION = RESULTS_DIR / "descriptor_correlation_matrix.csv"
OUT_PCA = RESULTS_DIR / "pca_loading_analysis.csv"
OUT_HIERARCHICAL = RESULTS_DIR / "hierarchical_subspace_summary.csv"

STAGE_1_FEATURES = ["GE_z", "LE_z", "C_orient_z"]
STAGE_2_FEATURES = ["A_radial_z", "S_orient_z"]
ALL_FEATURES = STAGE_1_FEATURES + STAGE_2_FEATURES
FEATURE_LABELS = [f.replace("_z", "") for f in ALL_FEATURES]

N_BOOTSTRAP = 1000


def calculate_cliffs_delta(g1, g2):
    n1, n2 = len(g1), len(g2)
    if n1 == 0 or n2 == 0:
        return 0.0
    more = sum(x > y for x in g1 for y in g2)
    less = sum(x < y for x in g1 for y in g2)
    return (more - less) / (n1 * n2)


def run_hierarchical_validation():
    print("[*] Starting TRACEBIND Phase 5D: Descriptor Taxonomy & Hierarchical Validation...\n")
    
    if not IN_PROFILES.exists():
        print(f"[!] Error: Could not locate {IN_PROFILES}. Please run Script 40 first.")
        return
        
    df = pd.read_csv(IN_PROFILES)
    df['cohort_clean'] = df['cohort_type'].str.replace(' ', '').str.lower()
    
    tc_df = df[df['cohort_clean'].isin(['coretc', 'expandedtc', 'core', 'expanded'])].copy()
    ctrl_df = df[df['cohort_clean'].isin(['negativecontrol', 'control'])].copy()
    
    print(f"Loaded {len(tc_df)} Tropical Cyclones and {len(ctrl_df)} Negative Controls.\n")

    # =========================================================================
    # 1. BOOTSTRAPPED CONFIDENCE INTERVALS (1000 Iterations)
    # =========================================================================
    print("--- 1. Computing 95% Bootstrapped Confidence Intervals (N=1000) ---")
    boot_results = []
    
    np.random.seed(42)
    for feat in ALL_FEATURES:
        feat_clean = feat.replace('_z', '')
        tc_vals = tc_df[feat].values
        ctrl_vals = ctrl_df[feat].values
        
        deltas = []
        tc_medians = []
        
        for _ in range(N_BOOTSTRAP):
            tc_sample = np.random.choice(tc_vals, size=len(tc_vals), replace=True)
            ctrl_sample = np.random.choice(ctrl_vals, size=len(ctrl_vals), replace=True)
            
            d = calculate_cliffs_delta(tc_sample, ctrl_sample)
            deltas.append(d)
            tc_medians.append(np.median(tc_sample))
            
        boot_results.append({
            "Descriptor": feat_clean,
            "TC_Median_Z": round(float(np.median(tc_vals)), 2),
            "TC_Median_CI_95": f"[{round(float(np.percentile(tc_medians, 2.5)), 2)}, {round(float(np.percentile(tc_medians, 97.5)), 2)}]",
            "Cliffs_Delta_Obs": round(calculate_cliffs_delta(tc_vals, ctrl_vals), 3),
            "Cliffs_Delta_CI_95": f"[{round(float(np.percentile(deltas, 2.5)), 3)}, {round(float(np.percentile(deltas, 97.5)), 3)}]"
        })
        
    df_boot = pd.DataFrame(boot_results)
    df_boot.to_csv(OUT_BOOTSTRAP, index=False)
    print(df_boot.to_string(index=False))
    print(f"\n[✓] Saved: {OUT_BOOTSTRAP}\n")

    # =========================================================================
    # 2. LEAVE-ONE-STORM-OUT (LOSO) RANK STABILITY (SPEARMAN'S RHO)
    # =========================================================================
    print("--- 2. Leave-One-Storm-Out (LOSO) Rank Stability Analysis ---")
    
    # Baseline full-cohort mean absolute Z ranking
    full_mean_abs_z = np.array([np.mean(np.abs(tc_df[f].values)) for f in ALL_FEATURES])
    baseline_ranks = np.argsort(np.argsort(-full_mean_abs_z)) + 1
    
    spearman_rhos = []
    loso_records = []
    
    for _, omitted_storm in tc_df.iterrows():
        loso_subset = tc_df[tc_df['storm_id'] != omitted_storm['storm_id']]
        loso_mean_abs_z = np.array([np.mean(np.abs(loso_subset[f].values)) for f in ALL_FEATURES])
        loso_ranks = np.argsort(np.argsort(-loso_mean_abs_z)) + 1
        
        rho, _ = spearmanr(baseline_ranks, loso_ranks)
        spearman_rhos.append(rho)
        
        rec = {"omitted_storm": omitted_storm['storm_id'], "spearman_rho": round(float(rho), 4)}
        for idx, f in enumerate(FEATURE_LABELS):
            rec[f"{f}_rank"] = int(loso_ranks[idx])
        loso_records.append(rec)
        
    df_loso = pd.DataFrame(loso_records)
    df_loso.to_csv(OUT_LOSO, index=False)
    
    avg_rho = float(np.mean(spearman_rhos))
    min_rho = float(np.min(spearman_rhos))
    print(f"  [✓] Average LOSO Spearman's Rho: {avg_rho:.4f} (Min: {min_rho:.4f})")
    print(f"  [✓] Saved: {OUT_LOSO}\n")

    # =========================================================================
    # 3. INTER-DESCRIPTOR CORRELATION MATRIX
    # =========================================================================
    print("--- 3. Inter-Descriptor Correlation Matrix (ERA5 TC Cohort) ---")
    tc_matrix = tc_df[ALL_FEATURES].values
    corr_matrix = np.corrcoef(tc_matrix, rowvar=False)
    
    df_corr = pd.DataFrame(corr_matrix, index=FEATURE_LABELS, columns=FEATURE_LABELS).round(3)
    df_corr.to_csv(OUT_CORRELATION)
    print(df_corr.to_string())
    print(f"\n[✓] Saved: {OUT_CORRELATION}\n")

    # =========================================================================
    # 4. PRINCIPAL COMPONENT ANALYSIS (UNSUPERVISED SUBSPACE PROOF)
    # =========================================================================
    print("--- 4. Principal Component Analysis (Unsupervised Dimensionality) ---")
    pca = PCA(n_components=5)
    pca.fit(tc_matrix)
    
    pca_data = []
    for comp_idx in range(pca.n_components_):
        rec = {
            "Component": f"PC_{comp_idx+1}",
            "Variance_Explained_Ratio": round(float(pca.explained_variance_ratio_[comp_idx]), 4),
            "Cumulative_Variance": round(float(np.sum(pca.explained_variance_ratio_[:comp_idx+1])), 4)
        }
        for f_idx, f_name in enumerate(FEATURE_LABELS):
            rec[f"{f_name}_loading"] = round(float(pca.components_[comp_idx, f_idx]), 4)
        pca_data.append(rec)
        
    df_pca = pd.DataFrame(pca_data)
    df_pca.to_csv(OUT_PCA, index=False)
    print(df_pca[["Component", "Variance_Explained_Ratio", "Cumulative_Variance"] + [f"{f}_loading" for f in FEATURE_LABELS]].to_string(index=False))
    print(f"\n[✓] Saved: {OUT_PCA}\n")

    # =========================================================================
    # 5. NON-PARAMETRIC SUBSPACE EVALUATION
    # =========================================================================
    print("--- 5. Non-Parametric Subspace Aggregates ---")
    
    # Tier 1 Aggregate: Mean |Z| across GE, LE, C_orient
    s1_tc_agg = np.mean(np.abs(tc_df[STAGE_1_FEATURES].values), axis=1)
    s1_ctrl_agg = np.mean(np.abs(ctrl_df[STAGE_1_FEATURES].values), axis=1)
    
    # Tier 2 Aggregate: Mean |Z| across A_radial, S_orient
    s2_tc_agg = np.mean(np.abs(tc_df[STAGE_2_FEATURES].values), axis=1)
    s2_ctrl_agg = np.mean(np.abs(ctrl_df[STAGE_2_FEATURES].values), axis=1)
    
    # Full 5D Aggregate: Mean |Z| across all 5
    full_tc_agg = np.mean(np.abs(tc_df[ALL_FEATURES].values), axis=1)
    full_ctrl_agg = np.mean(np.abs(ctrl_df[ALL_FEATURES].values), axis=1)

    subspace_table = [
        {
            "Subspace": "Tier 1: Phase Organization (GE, LE, C_orient)",
            "TC_Median_Abs_Z": round(float(np.median(s1_tc_agg)), 2),
            "Ctrl_Median_Abs_Z": round(float(np.median(s1_ctrl_agg)), 2),
            "Cliffs_Delta": round(calculate_cliffs_delta(s1_tc_agg, s1_ctrl_agg), 3)
        },
        {
            "Subspace": "Tier 2: Cyclone Geometry (A_radial, S_orient)",
            "TC_Median_Abs_Z": round(float(np.median(s2_tc_agg)), 2),
            "Ctrl_Median_Abs_Z": round(float(np.median(s2_ctrl_agg)), 2),
            "Cliffs_Delta": round(calculate_cliffs_delta(s2_tc_agg, s2_ctrl_agg), 3)
        },
        {
            "Subspace": "Full 5D Architecture (Unsegmented)",
            "TC_Median_Abs_Z": round(float(np.median(full_tc_agg)), 2),
            "Ctrl_Median_Abs_Z": round(float(np.median(full_ctrl_agg)), 2),
            "Cliffs_Delta": round(calculate_cliffs_delta(full_tc_agg, full_ctrl_agg), 3)
        }
    ]
    
    df_subspace = pd.DataFrame(subspace_table)
    df_subspace.to_csv(OUT_HIERARCHICAL, index=False)
    print(df_subspace.to_string(index=False))
    print(f"\n[✓] Saved: {OUT_HIERARCHICAL}\n")
    print("[*] Hierarchical Validation Complete!")

if __name__ == "__main__":
    run_hierarchical_validation()