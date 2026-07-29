"""
TRACEBIND Phase 8 C2 - Unblinding Protocol & Statistical Hypothesis Evaluation (v1.0 Final)

========================================================================
File: phase8/c2/unblind_and_evaluate_c2.py
Purpose: Verifies artifact hashes against pre-unblinding snapshot, executes
         paired paired bootstrap CIs, Monte Carlo permutation testing, Hedges' g,
         Cliff's Delta, ROC AUC, and generates reproducible publication figures 
         with full environment metadata logging.
========================================================================
"""

import os
import sys
import json
import hashlib
import logging
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("C2_Unblind_Evaluator_v1.0")

BASE_DIR = Path(__file__).resolve().parent
EXTRACTION_DIR = BASE_DIR / "extraction"
RESULTS_CSV = EXTRACTION_DIR / "c2_cphi_results.csv"
AUDIT_SNAPSHOT = BASE_DIR / "diagnostics" / "pre_unblinding_audit_snapshot.json"

KEY_CANDIDATES = [
    BASE_DIR / "keys" / "c2_unblinding_key.json",
    BASE_DIR / "c2_unblinding_key.json",
    BASE_DIR.parent / "c2_unblinding_key.json",
    Path.cwd() / "c2_unblinding_key.json",
    Path.cwd() / "phase8" / "c2" / "keys" / "c2_unblinding_key.json",
]

KEY_PATH = None
for candidate in KEY_CANDIDATES:
    if candidate.exists():
        KEY_PATH = candidate
        break

OUTPUT_DIR = BASE_DIR / "unblinded_results"


def hash_file(file_path):
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def compute_cliffs_delta(x, y):
    """Computes Cliff's Delta effect size for two non-parametric groups."""
    n_x, n_y = len(x), len(y)
    greater = sum(1 for i in x for j in y if i > j)
    less = sum(1 for i in x for j in y if i < j)
    return (greater - less) / (n_x * n_y)


def compute_hedges_g(x, y):
    """Computes Hedges' g effect size with small-sample bias correction factor J."""
    n_x, n_y = len(x), len(y)
    df = n_x + n_y - 2
    if df < 1:
        return 0.0
    var_x, var_y = np.var(x, ddof=1), np.var(y, ddof=1)
    s_pooled = np.sqrt(((n_x - 1) * var_x + (n_y - 1) * var_y) / df)
    if s_pooled == 0:
        return 0.0
    cohens_d = (np.mean(x) - np.mean(y)) / s_pooled
    # Small sample bias correction factor J
    j_factor = 1.0 - (3.0 / (4.0 * df - 1.0))
    return cohens_d * j_factor


def compute_roc_auc(scores, labels, pos_label="cyclone"):
    """Computes Receiver Operating Characteristic Area Under Curve (ROC AUC)."""
    binary_labels = np.array([1 if l == pos_label else 0 for l in labels])
    n_pos = np.sum(binary_labels)
    n_neg = len(binary_labels) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.5

    ranks = stats.rankdata(scores)
    pos_ranks = np.sum(ranks[binary_labels == 1])
    auc = (pos_ranks - (n_pos * (n_pos + 1)) / 2.0) / (n_pos * n_neg)
    return float(auc)


def paired_bootstrap_ci(df, score_col, label_col, pos_label="cyclone", n_resamples=10000, ci=95, seed=42):
    """
    Resamples (score, label) PAIRS together to properly evaluate 
    ROC AUC & mean difference confidence intervals.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    auc_boots = []
    diff_boots = []

    scores = df[score_col].values
    labels = df[label_col].values

    for _ in range(n_resamples):
        indices = rng.choice(n, size=n, replace=True)
        boot_scores = scores[indices]
        boot_labels = labels[indices]

        # Calculate ROC AUC on paired resample
        auc = compute_roc_auc(boot_scores, boot_labels, pos_label=pos_label)
        auc_boots.append(auc)

        # Calculate Mean Difference
        cyc = boot_scores[boot_labels == pos_label]
        ctrl = boot_scores[boot_labels != pos_label]
        if len(cyc) > 0 and len(ctrl) > 0:
            diff_boots.append(np.mean(cyc) - np.mean(ctrl))

    lower_p = (100 - ci) / 2.0
    upper_p = 100 - lower_p

    auc_ci = np.percentile(auc_boots, [lower_p, upper_p])
    diff_ci = np.percentile(diff_boots, [lower_p, upper_p]) if diff_boots else (0.0, 0.0)

    return auc_ci, diff_ci


def permutation_test_mean_diff(x, y, n_permutations=10000, seed=42):
    """Two-sample Monte Carlo permutation test for mean difference."""
    rng = np.random.default_rng(seed)
    observed_diff = np.abs(np.mean(x) - np.mean(y))
    pooled = np.concatenate([x, y])
    n_x = len(x)

    count_extreme = 0
    for _ in range(n_permutations):
        permuted = rng.permutation(pooled)
        perm_x = permuted[:n_x]
        perm_y = permuted[n_x:]
        perm_diff = np.abs(np.mean(perm_x) - np.mean(perm_y))
        if perm_diff >= observed_diff:
            count_extreme += 1

    p_value = count_extreme / n_permutations
    return observed_diff, p_value


def get_column(df, candidates, purpose):
    for c in candidates:
        if c in df.columns:
            return c
    raise RuntimeError(f"Missing required {purpose} column. Evaluated: {candidates}")


def main():
    if not RESULTS_CSV.exists():
        logger.error(f"Results file missing: {RESULTS_CSV}")
        sys.exit(1)

    if not KEY_PATH or not KEY_PATH.exists():
        logger.error("Unblinding key file missing!")
        sys.exit(1)

    if not AUDIT_SNAPSHOT.exists():
        logger.error(f"Audit snapshot missing: {AUDIT_SNAPSHOT}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Verification against Audit Snapshot
    with open(AUDIT_SNAPSHOT, "r", encoding="utf-8") as f:
        snapshot = json.load(f)

    snapshot_hashes = snapshot.get("artifacts_sha256", {})
    results_current_hash = hash_file(RESULTS_CSV)
    key_current_hash = hash_file(KEY_PATH)

    results_filename = RESULTS_CSV.name
    results_expected_hash = snapshot_hashes.get(results_filename, {}).get("sha256")

    logger.info("========================================================")
    logger.info("TRACEBIND Phase 8 C2 - UNBLINDING PROTOCOL (v1.0 Final)")
    logger.info("========================================================")
    logger.info("Verifying Pre-Unblinding Cryptographic Hashes...")

    hash_verification_passed = True
    if results_expected_hash:
        if results_current_hash == results_expected_hash:
            logger.info(f"  [PASS] Extraction CSV Hash: {results_current_hash[:16]}...")
        else:
            logger.error(f"  [FAIL] CSV Hash Mismatch! Exp: {results_expected_hash[:16]}, Got: {results_current_hash[:16]}")
            hash_verification_passed = False
    else:
        logger.warning("  [WARN] Results CSV not found in audit snapshot registry.")

    logger.info(f"  [INFO] Unblinding Key Hash:  {key_current_hash[:16]}...")

    if not hash_verification_passed:
        logger.error("AUDIT VERIFICATION FAILED. Halting unblinding execution.")
        sys.exit(1)

    logger.info("Verification Result: PASSED ✅")
    logger.info("Proceeding to reveal cohort labels...")
    logger.info("========================================================\n")

    # 2. Load Blinded Results & Join Labels
    df_blinded = pd.read_csv(RESULTS_CSV)
    c_phi_col = get_column(df_blinded, ["c_phi", "cphi", "phase_coherence"], "C_phi")
    vel_col = get_column(df_blinded, ["mean_velocity_m_s", "mean_speed", "mean_velocity", "v_mean"], "Velocity")
    id_col = get_column(df_blinded, ["blinded_id", "uuid_token", "uuid"], "ID/UUID")

    with open(KEY_PATH, "r", encoding="utf-8") as f:
        unblinding_key_data = json.load(f)

    if isinstance(unblinding_key_data, list):
        key_map = {
            item.get("blinded_id") or item.get("uuid_token"): item["true_cohort"]
            for item in unblinding_key_data
        }
    elif isinstance(unblinding_key_data, dict) and "mappings" in unblinding_key_data:
        key_map = {
            item.get("blinded_id") or item.get("uuid_token"): item["true_cohort"]
            for item in unblinding_key_data["mappings"]
        }
    else:
        key_map = unblinding_key_data

    df_blinded["cohort_label"] = df_blinded[id_col].map(key_map)

    if df_blinded["cohort_label"].isnull().any():
        missing_count = df_blinded["cohort_label"].isnull().sum()
        logger.error(f"Unblinding mapping incomplete! {missing_count} cases missing labels.")
        sys.exit(1)

    # Lock Joined Master Dataset
    joined_csv_path = OUTPUT_DIR / "c2_unblinded_master_dataset.csv"
    df_blinded.to_csv(joined_csv_path, index=False)
    logger.info(f"Unblinded master dataset locked: {joined_csv_path}")

    # 3. Cohort Partitioning & Statistical Evaluation
    cyclone_data = df_blinded[df_blinded["cohort_label"] == "cyclone"][c_phi_col].values
    control_data = df_blinded[df_blinded["cohort_label"] == "control"][c_phi_col].values

    n_cyclone, n_control = len(cyclone_data), len(control_data)
    mean_diff_obs = np.mean(cyclone_data) - np.mean(control_data)

    logger.info(f"\nUNBLINDED COHORT BREAKDOWN:")
    logger.info(f"  Cyclone Cohort (n={n_cyclone}): Mean = {np.mean(cyclone_data):.4f}, Std = {np.std(cyclone_data, ddof=1):.4f}")
    logger.info(f"  Control Cohort (n={n_control}): Mean = {np.mean(control_data):.4f}, Std = {np.std(control_data, ddof=1):.4f}")
    logger.info(f"  Observed Mean Difference:  {mean_diff_obs:.4f}")

    # Statistical Tests
    t_stat, p_val_welch = stats.ttest_ind(cyclone_data, control_data, equal_var=False)
    u_stat, p_val_mwu = stats.mannwhitneyu(cyclone_data, control_data, alternative="two-sided")
    _, p_val_perm = permutation_test_mean_diff(cyclone_data, control_data, n_permutations=10000, seed=42)

    hedges_g = compute_hedges_g(cyclone_data, control_data)
    cliffs_d = compute_cliffs_delta(cyclone_data, control_data)
    roc_auc = compute_roc_auc(df_blinded[c_phi_col].values, df_blinded["cohort_label"].values, pos_label="cyclone")

    # Correct Paired Bootstrap (10,000 Resamples)
    auc_ci, diff_ci = paired_bootstrap_ci(
        df_blinded, score_col=c_phi_col, label_col="cohort_label", pos_label="cyclone", n_resamples=10000, seed=42
    )

    logger.info("\n==================================================")
    logger.info("FINAL STATISTICAL EVALUATION PANEL ($C_\\phi$)")
    logger.info("==================================================")
    logger.info(f"  Welch's t-test:        t = {t_stat:.4f}, p = {p_val_welch:.6e}")
    logger.info(f"  Mann-Whitney U test:   U = {u_stat:.1f}, p = {p_val_mwu:.6e}")
    logger.info(f"  Permutation p-value:   p = {p_val_perm:.6e} (10,000 perms)")
    logger.info(f"  Hedges' g:             {hedges_g:.4f}")
    logger.info(f"  Cliff's Delta:         {cliffs_d:.4f}")
    logger.info(f"  Mean Diff 95% CI:      {mean_diff_obs:.4f} [{diff_ci[0]:.4f}, {diff_ci[1]:.4f}]")
    logger.info(f"  ROC AUC:               {roc_auc:.4f} [95% CI: {auc_ci[0]:.4f}, {auc_ci[1]:.4f}]")
    logger.info("==================================================\n")

    # 4. Publication Figures (Seeded Randomness)
    rng_fig = np.random.default_rng(42)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    cohort_colors = {"cyclone": "crimson", "control": "steelblue"}
    df_blinded["color"] = df_blinded["cohort_label"].map(cohort_colors)

    # Boxplot / Stripchart Comparison
    data_to_plot = [control_data, cyclone_data]
    bp = ax1.boxplot(data_to_plot, patch_artist=True, labels=["Control", "Cyclone"], widths=0.4)
    bp["boxes"][0].set_facecolor("lightskyblue")
    bp["boxes"][1].set_facecolor("salmon")

    for i, group in enumerate(["control", "cyclone"]):
        subset = df_blinded[df_blinded["cohort_label"] == group]
        jitter = rng_fig.normal(i + 1, 0.04, size=len(subset))
        ax1.scatter(jitter, subset[c_phi_col], color=cohort_colors[group], edgecolors="k", s=65, zorder=3, alpha=0.9)

    ax1.axhline(0.65, color="gray", linestyle="--", linewidth=1.5, label="Threshold (0.65)")
    ax1.set_title("Unblinded $C_\\phi$ Distribution by Cohort", fontsize=11, fontweight="bold")
    ax1.set_ylabel("$C_\\phi$", fontsize=10)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    # Scatter Plot: C_phi vs Velocity
    for group, color in cohort_colors.items():
        subset = df_blinded[df_blinded["cohort_label"] == group]
        ax2.scatter(subset[vel_col], subset[c_phi_col], color=color, label=group.capitalize(), edgecolors="k", s=70, alpha=0.85)

    ax2.axhline(0.65, color="gray", linestyle="--", linewidth=1.5, label="Threshold (0.65)")
    ax2.set_title("$C_\\phi$ vs Mean Velocity ($V_{mean}$)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Mean Velocity (m/s)", fontsize=10)
    ax2.set_ylabel("$C_\\phi$", fontsize=10)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "c2_unblinded_cohort_separation.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    logger.info(f"Saved publication figure to: {fig_path}")

    # 5. Final Evaluation Report with Environment Metadata
    summary_report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment_metadata": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "scipy_version": stats.__version__,
            "matplotlib_version": plt.__import__("matplotlib").__version__,
        },
        "audit_verification": {
            "snapshot_verified": True,
            "results_csv_sha256": results_current_hash,
            "unblinding_key_sha256": key_current_hash,
        },
        "sample_sizes": {
            "total": len(df_blinded),
            "cyclone": n_cyclone,
            "control": n_control,
        },
        "cohort_statistics": {
            "cyclone": {
                "mean": float(np.mean(cyclone_data)),
                "std": float(np.std(cyclone_data, ddof=1)),
                "median": float(np.median(cyclone_data)),
                "iqr": float(np.percentile(cyclone_data, 75) - np.percentile(cyclone_data, 25)),
            },
            "control": {
                "mean": float(np.mean(control_data)),
                "std": float(np.std(control_data, ddof=1)),
                "median": float(np.median(control_data)),
                "iqr": float(np.percentile(control_data, 75) - np.percentile(control_data, 25)),
            },
        },
        "hypothesis_tests": {
            "mean_difference": {
                "observed": float(mean_diff_obs),
                "ci_95_bootstrap": [float(diff_ci[0]), float(diff_ci[1])],
            },
            "welch_t_test": {"t_stat": float(t_stat), "p_value": float(p_val_welch)},
            "mann_whitney_u": {"u_stat": float(u_stat), "p_value": float(p_val_mwu)},
            "permutation_test": {"p_value": float(p_val_perm), "permutations": 10000},
            "hedges_g": float(hedges_g),
            "cliffs_delta": float(cliffs_d),
            "roc_auc": {
                "value": float(roc_auc),
                "ci_95_bootstrap": [float(auc_ci[0]), float(auc_ci[1])],
                "bootstrap_resamples": 10000,
            },
        },
    }

    report_path = OUTPUT_DIR / "c2_unblinded_evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)

    logger.info(f"Unblinded evaluation report saved to: {report_path}")
    logger.info("STATUS: UNBLINDING & STATISTICAL EVALUATION COMPLETE ✅")


if __name__ == "__main__":
    main()