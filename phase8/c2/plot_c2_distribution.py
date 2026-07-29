"""
TRACEBIND Phase 8 C2 - Pre-Unblinding Distribution Audit & Artifact Snapshot (v1.0)

========================================================================
File: phase8/c2/plot_c2_distribution.py
Purpose: Audits extraction output against manifest, captures system metadata,
         computes non-parametric stats + ECDF, prints ASCII histogram, exports
         plots, and generates pre-unblinding SHA-256 audit snapshot.
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
from scipy.stats import skew, kurtosis

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("C2_Audit_v1.0")

BASE_DIR = Path(__file__).resolve().parent
RESULTS_CSV = BASE_DIR / "extraction" / "c2_cphi_results.csv"
SUMMARY_JSON = BASE_DIR / "extraction" / "c2_validation_summary.json"

MANIFEST_CANDIDATES = [
    BASE_DIR / "manifest" / "c2_cohort_manifest_blinded.json",
    BASE_DIR / "c2_cohort_manifest_blinded.json",
    BASE_DIR.parent / "c2_cohort_manifest_blinded.json",
    Path.cwd() / "c2_cohort_manifest_blinded.json",
    Path.cwd() / "phase8" / "c2" / "manifest" / "c2_cohort_manifest_blinded.json",
]

MANIFEST_PATH = None
for candidate in MANIFEST_CANDIDATES:
    if candidate.exists():
        MANIFEST_PATH = candidate
        break

OUTPUT_DIR = BASE_DIR / "diagnostics"


def get_column(df, candidates, column_purpose):
    """Dynamically resolves column names from DataFrame."""
    for c in candidates:
        if c in df.columns:
            return c
    raise RuntimeError(
        f"Missing required {column_purpose} column. Evaluated candidates: {candidates}"
    )


def hash_file(file_path):
    """Computes SHA-256 hash of target file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def generate_ascii_histogram(data, bins=10):
    """Outputs clean terminal ASCII histogram for immediate visual check."""
    counts, bin_edges = np.histogram(data, bins=bins, range=(0.0, 1.0))
    logger.info("\n==================================================")
    logger.info("PRE-UNBLINDING C_phi ASCII HISTOGRAM (BIN SIZE: 0.1)")
    logger.info("==================================================")
    for i in range(len(counts)):
        edge_start = bin_edges[i]
        edge_end = bin_edges[i + 1]
        stars = "*" * counts[i]
        logger.info(f"[{edge_start:.1f} - {edge_end:.1f}] | {counts[i]:2d} | {stars}")
    logger.info("==================================================\n")


def main():
    logger.info("========================================================================")
    logger.info("TRACEBIND PHASE 8 C2 - PRE-UNBLINDING AUDIT SNAPSHOT (v1.0)")
    logger.info("========================================================================")

    if not RESULTS_CSV.exists():
        logger.error(f"Results CSV missing: {RESULTS_CSV}")
        sys.exit(1)

    if not MANIFEST_PATH or not MANIFEST_PATH.exists():
        logger.error("Blinded manifest missing!")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RESULTS_CSV)

    # 1. Dynamic Column Resolution
    c_phi_col = get_column(df, ["c_phi", "cphi", "phase_coherence"], "C_phi")
    vel_col = get_column(
        df, ["mean_velocity_m_s", "mean_speed", "mean_velocity", "v_mean"], "Velocity"
    )
    id_col = get_column(df, ["blinded_id", "uuid_token", "uuid"], "ID/UUID")

    # 2. Manifest Consistency Check
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    manifest_uuids = {
        case.get("blinded_id") or case.get("uuid_token")
        for case in manifest_data.get("cases", [])
    }
    extracted_uuids = set(df[id_col].values)

    missing_in_manifest = extracted_uuids - manifest_uuids
    if missing_in_manifest:
        logger.error(f"MANIFEST CONSISTENCY FAILURE! UUIDs not in manifest: {missing_in_manifest}")
        sys.exit(1)
    logger.info(f"Manifest Consistency Check: PASSED ({len(extracted_uuids)}/20 matched)")

    # 3. Label-Independent Distribution Analysis
    c_phi_vals = df[c_phi_col].values
    q1, median, q3 = np.percentile(c_phi_vals, [25, 50, 75])
    iqr = q3 - q1
    sk = float(skew(c_phi_vals))
    kt = float(kurtosis(c_phi_vals))

    logger.info("-" * 50)
    logger.info("LABEL-INDEPENDENT DISTRIBUTION STATISTICS ($C_\\phi$):")
    logger.info(f"  Median:   {median:.6f}")
    logger.info(f"  Q1 / Q3:  {q1:.6f} / {q3:.6f}")
    logger.info(f"  IQR:      {iqr:.6f}")
    logger.info(f"  Skewness: {sk:.6f}")
    logger.info(f"  Kurtosis: {kt:.6f}")
    logger.info("-" * 50)

    # 4. Terminal ASCII Histogram
    generate_ascii_histogram(c_phi_vals)

    # 5. Figure Generation (Histogram, Scatter, Empirical CDF)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Histogram
    ax1.hist(c_phi_vals, bins=10, range=(0.0, 1.0), color="skyblue", edgecolor="black", alpha=0.8)
    ax1.axvline(0.65, color="red", linestyle="--", linewidth=1.5, label="Threshold (0.65)")
    ax1.set_title("Pre-Unblinding $C_\\phi$ Histogram", fontsize=11, fontweight="bold")
    ax1.set_xlabel("$C_\\phi$", fontsize=10)
    ax1.set_ylabel("Frequency", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    # Panel 2: Scatter (C_phi vs Mean Velocity)
    ax2.scatter(df[vel_col], c_phi_vals, color="navy", alpha=0.8, edgecolors="k", s=60)
    ax2.axhline(0.65, color="red", linestyle="--", linewidth=1.5, label="Threshold (0.65)")
    ax2.set_title("$C_\\phi$ vs Mean Velocity ($V_{mean}$)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Mean Velocity (m/s)", fontsize=10)
    ax2.set_ylabel("$C_\\phi$", fontsize=10)
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    # Panel 3: Empirical CDF (ECDF)
    sorted_data = np.sort(c_phi_vals)
    y_vals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    ax3.step(sorted_data, y_vals, where="post", color="darkgreen", linewidth=2)
    ax3.axvline(0.65, color="red", linestyle="--", linewidth=1.5, label="Threshold (0.65)")
    ax3.set_title("Empirical CDF (ECDF)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("$C_\\phi$", fontsize=10)
    ax3.set_ylabel("Cumulative Probability", fontsize=10)
    ax3.set_xlim(0, 1.0)
    ax3.set_ylim(0, 1.05)
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend()

    plt.tight_layout()
    plot_path = OUTPUT_DIR / "c2_pre_unblinding_diagnostics.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logger.info(f"Saved diagnostic plots (Hist, Scatter, ECDF) to: {plot_path}")

    # 6. Environmental Metadata & Audit Snapshot
    artifacts_to_hash = [RESULTS_CSV, SUMMARY_JSON, MANIFEST_PATH]
    file_hashes = {}

    for artifact in artifacts_to_hash:
        if artifact.exists():
            file_hashes[artifact.name] = {
                "relative_path": str(artifact.relative_to(BASE_DIR)),
                "sha256": hash_file(artifact),
            }

    audit_snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
        },
        "consistency_checks": {
            "manifest_matched_cases": len(extracted_uuids),
            "manifest_consistency_passed": True,
        },
        "label_independent_stats": {
            "count": len(c_phi_vals),
            "median": float(median),
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(iqr),
            "skewness": float(sk),
            "kurtosis": float(kt),
        },
        "artifacts_sha256": file_hashes,
    }

    audit_snapshot_path = OUTPUT_DIR / "pre_unblinding_audit_snapshot.json"
    with open(audit_snapshot_path, "w", encoding="utf-8") as f:
        json.dump(audit_snapshot, f, indent=2)

    logger.info(f"Audit snapshot saved to: {audit_snapshot_path}")
    logger.info("STATUS: PRE-UNBLINDING ARTIFACTS HASHED & SNAPSHOT CREATED ✅")


if __name__ == "__main__":
    main()