import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

def compute_cliffs_delta(x, y):
    """Compute Cliff's delta effect size for non-parametric comparison."""
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return 0.0
    diffs = np.subtract.outer(x, y)
    more = np.sum(diffs > 0)
    less = np.sum(diffs < 0)
    return float((more - less) / (nx * ny))

def compute_rank_biserial(u_stat, nx, ny):
    """Compute Rank-Biserial Correlation from Mann-Whitney U statistic."""
    if nx * ny == 0:
        return 0.0
    return float(1.0 - (2.0 * u_stat) / (nx * ny))

def compute_hedges_g(x, y):
    """Compute Cohen's d with Hedge's g small-sample correction."""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return 0.0
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled_std = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if pooled_std == 0:
        return 0.0
    d = (np.mean(x) - np.mean(y)) / pooled_std
    j_factor = 1.0 - (3.0 / (4.0 * (nx + ny) - 9.0))
    return float(d * j_factor)

def bootstrap_ci(x, y, func, rng, n_boot=10000, ci=95):
    """Compute percentile bootstrap confidence intervals using modern NumPy Generator."""
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return 0.0, 0.0
    
    boot_stats = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        x_resample = rng.choice(x, size=nx, replace=True)
        y_resample = rng.choice(y, size=ny, replace=True)
        boot_stats[i] = func(x_resample, y_resample)
    
    lower = np.percentile(boot_stats, (100 - ci) / 2.0)
    upper = np.percentile(boot_stats, 100 - (100 - ci) / 2.0)
    return float(lower), float(upper)

def apply_fdr_benjamini_hochberg(p_values):
    """Benjamini-Hochberg FDR correction on pre-registered primary tests."""
    p_values = np.asarray(p_values, dtype=np.float64)
    n = len(p_values)
    if n == 0:
        return np.array([])
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]
    adjusted_p = np.zeros(n, dtype=np.float64)
    
    cum_min = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        adj = (sorted_p[i] * n) / rank
        cum_min = min(cum_min, adj)
        adjusted_p[sorted_indices[i]] = min(cum_min, 1.0)
        
    return adjusted_p

def isolate_cohorts(df):
    """Robustly isolate TC vs Control cohorts across metadata fields."""
    tc_mask = pd.Series(False, index=df.index)
    for col in ["cohort_id", "cohort_name", "system_class"]:
        if col in df.columns:
            tc_mask = tc_mask | df[col].astype(str).str.upper().str.contains("TC")
    
    tc_df = df[tc_mask].copy()
    ctrl_df = df[~tc_mask].copy()
    return tc_df, ctrl_df

def run_negative_control_permutations(df, metric_cols, tc_count, rng, n_permutations=1000):
    """Generate negative controls by randomly permuting cohort labels."""
    print("[INFO] Running Empirical Negative Control Engine (Label Permutations)...")
    false_positives = {col: 0 for col in metric_cols}
    n_total = len(df)

    if tc_count == 0 or tc_count >= n_total:
        return {col: 0.0 for col in metric_cols}

    for _ in range(n_permutations):
        permuted_labels = rng.permutation(n_total)
        pseudo_tc = df.iloc[permuted_labels[:tc_count]]
        pseudo_ctrl = df.iloc[permuted_labels[tc_count:]]

        for col in metric_cols:
            x = pseudo_tc[col].dropna().values
            y = pseudo_ctrl[col].dropna().values
            if len(x) >= 3 and len(y) >= 3:
                stat, p_val = stats.mannwhitneyu(x, y, alternative='two-sided')
                if p_val < 0.05:
                    false_positives[col] += 1

    emp_fpr = {col: false_positives[col] / n_permutations for col in metric_cols}
    return emp_fpr

def main():
    print("[INFO] Starting Phase 6E Rigorous Statistical Evaluation Engine...")
    rng = np.random.default_rng(42)
    base_dir = Path(__file__).resolve().parent
    metrics_path = base_dir / "data" / "summary" / "structural_tracebind_metrics.parquet"
    output_dir = base_dir / "Phase6E"
    output_dir.mkdir(exist_ok=True)

    if not metrics_path.exists():
        metrics_path = base_dir / "data" / "summary" / "structural_tracebind_metrics.csv"
        df = pd.read_csv(metrics_path)
    else:
        df = pd.read_parquet(metrics_path)

    target_metrics = [
        "circulation_250km_mean",
        "compactness_ratio_mean",
        "asymmetry_index_mean",
        "filamentation_fraction_mean",
        "coherence_index_mean",
        "boundary_entropy_bits_mean",
        "boundary_sharpness_mean"
    ]
    
    metric_cols = [c for c in target_metrics if c in df.columns and df[c].notna().sum() > 0]
    dropped = set(target_metrics) - set(metric_cols)
    if dropped:
        print(f"[WARN] Omitted missing/empty metrics: {dropped}")

    # Explicit Cohort Selection using Flexible Isolation
    tc_df, ctrl_df = isolate_cohorts(df)

    print(f"[INFO] Cohort metadata separation: TC (N={len(tc_df)}), Control (N={len(ctrl_df)})")

    # 1. Exploratory Statistics
    exploratory_rows = []
    for col in metric_cols:
        for c_name, c_sub in [("TC", tc_df), ("Control", ctrl_df), ("Combined", df)]:
            vals = c_sub[col].dropna().values
            if len(vals) == 0:
                continue
            shapiro_stat, shapiro_p = stats.shapiro(vals) if len(vals) >= 3 else (np.nan, np.nan)
            q25, q75 = np.percentile(vals, 25), np.percentile(vals, 75)
            
            exploratory_rows.append({
                "metric": col,
                "cohort": c_name,
                "sample_size_N": len(vals),
                "mean": float(np.mean(vals)),
                "std_dev": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "median": float(np.median(vals)),
                "iqr": float(q75 - q25),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "skewness": float(stats.skew(vals)) if len(vals) > 2 else 0.0,
                "kurtosis": float(stats.kurtosis(vals)) if len(vals) > 3 else 0.0,
                "shapiro_stat": float(shapiro_stat),
                "shapiro_p_value": float(shapiro_p)
            })

    exp_df = pd.DataFrame(exploratory_rows)
    exp_df.to_parquet(output_dir / "exploratory_statistics.parquet", index=False)
    exp_df.to_csv(output_dir / "exploratory_statistics.csv", index=False)

    # 2. Negative Controls
    emp_fpr = run_negative_control_permutations(df, metric_cols, len(tc_df), rng=rng, n_permutations=1000)

    # 3. Hypothesis Testing & Effect Sizes
    test_rows, effect_rows, summary_json_data = [], [], []
    raw_p_mw = []

    for col in metric_cols:
        x = tc_df[col].dropna().values
        y = ctrl_df[col].dropna().values

        if len(x) < 3 or len(y) < 3:
            continue

        # Homoscedasticity: Brown-Forsythe (Levene with median)
        bf_stat, bf_p = stats.levene(x, y, center='median')

        # Primary Pre-registered Test: Mann-Whitney U
        mw_res = stats.mannwhitneyu(x, y, alternative='two-sided')
        raw_p_mw.append(mw_res.pvalue)

        # Secondary Exploratory Tests (Uncorrected)
        ks_res = stats.ks_2samp(x, y)
        mood_res = stats.median_test(x, y)

        # Effect Sizes & 10,000 Iteration Bootstrap CIs
        c_delta = compute_cliffs_delta(x, y)
        r_biserial = compute_rank_biserial(mw_res.statistic, len(x), len(y))
        hedges_g = compute_hedges_g(x, y)

        c_delta_low, c_delta_high = bootstrap_ci(x, y, compute_cliffs_delta, rng=rng, n_boot=10000)
        med_diff = float(np.median(x) - np.median(y))
        med_diff_low, med_diff_high = bootstrap_ci(x, y, lambda a, b: np.median(a) - np.median(b), rng=rng, n_boot=10000)

        test_rows.append({
            "metric": col,
            "tc_N": len(x),
            "ctrl_N": len(y),
            "brown_forsythe_stat": float(bf_stat),
            "brown_forsythe_p": float(bf_p),
            "mw_u_stat": float(mw_res.statistic),
            "mw_p_raw": float(mw_res.pvalue),
            "ks_stat": float(ks_res.statistic),
            "ks_p_value_uncorrected": float(ks_res.pvalue),
            "mood_stat": float(mood_res[0]),
            "mood_p_value_uncorrected": float(mood_res[1]),
            "median_difference": med_diff,
            "median_diff_ci95_low": med_diff_low,
            "median_diff_ci95_high": med_diff_high,
            "empirical_false_positive_rate": emp_fpr[col]
        })

        effect_rows.append({
            "metric": col,
            "cliffs_delta": c_delta,
            "cliffs_delta_ci95_low": c_delta_low,
            "cliffs_delta_ci95_high": c_delta_high,
            "rank_biserial_corr": r_biserial,
            "hedges_g": hedges_g,
            "effect_magnitude": "Strong" if abs(c_delta) >= 0.474 else ("Medium" if abs(c_delta) >= 0.33 else ("Small" if abs(c_delta) >= 0.147 else "Negligible"))
        })

    # FDR Correction on Primary Mann-Whitney U Tests
    fdr_p_values = apply_fdr_benjamini_hochberg(raw_p_mw)
    for i, r in enumerate(test_rows):
        r["mw_p_fdr_bh"] = float(fdr_p_values[i])
        p_fdr = r["mw_p_fdr_bh"]
        c_delta = effect_rows[i]["cliffs_delta"]
        c_low, c_high = effect_rows[i]["cliffs_delta_ci95_low"], effect_rows[i]["cliffs_delta_ci95_high"]
        ci_excludes_zero = not (c_low <= 0 <= c_high)

        # Objective Phrasing Decision
        if p_fdr < 0.05 and abs(c_delta) >= 0.33 and ci_excludes_zero:
            eval_phrase = "Evidence consistent with preregistered criterion"
        elif p_fdr < 0.05:
            eval_phrase = "Statistically significant, limited practical effect"
        else:
            eval_phrase = "No statistically supported difference under current dataset"

        r["preregistered_evaluation"] = eval_phrase

        summary_json_data.append({
            "metric": r["metric"],
            "p_fdr_bh": p_fdr,
            "cliffs_delta": c_delta,
            "effect_class": effect_rows[i]["effect_magnitude"],
            "ci95_excludes_zero": ci_excludes_zero,
            "evaluation": eval_phrase
        })

    test_df = pd.DataFrame(test_rows)
    effect_df = pd.DataFrame(effect_rows)

    test_df.to_csv(output_dir / "hypothesis_test_results.csv", index=False)
    effect_df.to_csv(output_dir / "effect_sizes.csv", index=False)

    with open(output_dir / "preregistered_evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_json_data, f, indent=2)

    report_md = f"""# Phase 6E: Rigorous Statistical & Hypothesis Evaluation

**Execution Scope:** Evaluation of TRACEBIND structural and phase boundary metrics across TC ($N={len(tc_df)}$) and Control ($N={len(ctrl_df)}$) cohorts.  
**Primary Test Preregistration:** Mann-Whitney U Test (Two-sided) with Benjamini-Hochberg FDR Multiple Comparison Adjustment ($q < 0.05$).  
**Effect Size Bootstrapping:** 10,000 iterations using NumPy Generator (`default_rng(42)`).

---

## 1. Preregistered Evaluation Summary

| Metric | Primary MW-U $p$ (FDR) | Cliff's $\\delta$ [95% Bootstrap CI] | Empirical False Positive Rate ($\\alpha_{{\\text{{emp}}}}$) | Preregistered Evaluation Outcome |
| :--- | :--- | :--- | :--- | :--- |
"""

    for t_row, e_row in zip(test_rows, effect_rows):
        m_name = t_row["metric"]
        p_fdr = t_row["mw_p_fdr_bh"]
        c_delta = e_row["cliffs_delta"]
        c_low, c_high = e_row["cliffs_delta_ci95_low"], e_row["cliffs_delta_ci95_high"]
        fpr = t_row["empirical_false_positive_rate"]
        outcome = t_row["preregistered_evaluation"]

        report_md += f"| `{m_name}` | {p_fdr:.4e} | {c_delta:+0.3f} [{c_low:+0.3f}, {c_high:+0.3f}] | {fpr:.1%} | **{outcome}** |\n"

    report_md += """
---

## 2. Statistical Methodology & Diagnostics

* **Exploratory Distributional Diagnostics:** Shapiro-Wilk test for normality and Brown-Forsythe (median-centered Levene) test for variance equality were recorded in `exploratory_statistics.parquet`.
* **Negative Controls:** Empirical false-positive rates were established by running 1,000 label-swapping permutations across the dataset.
* **Secondary Uncorrected Tests:** Kolmogorov-Smirnov and Mood's Median tests are logged in `hypothesis_test_results.csv` as exploratory metrics.

---

## 3. Data Artifacts Directory

* `Phase6E/exploratory_statistics.parquet` (and `.csv`)
* `Phase6E/hypothesis_test_results.csv`
* `Phase6E/effect_sizes.csv`
* `Phase6E/preregistered_evaluation_summary.json`
"""

    report_path = output_dir / "Phase6E_Statistical_Evaluation.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"[INFO] Phase 6E Evaluation complete. Summary written to:\n - {report_path}\n - {output_dir / 'preregistered_evaluation_summary.json'}")

if __name__ == "__main__":
    main()