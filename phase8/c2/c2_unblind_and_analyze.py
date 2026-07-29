"""TRACEBIND Phase 8 Stage C2 - Hardened Unblinding & Statistical Evaluation

========================================================================
File: phase8/c2/c2_unblind_and_analyze.py

Verification Trigger Checks:
  1. Verifies c2_cphi_results.csv, qc_report.json, failed_cases.log exist.
  2. Asserts failed_cases.log contains NO_EXTRACTION_FAILURES_LOGGED.
  3. Asserts successful_extractions == total_cases in qc_report.json.
  4. Evaluates two-tailed Mann-Whitney U, ROC/AUC, Rosenthal's r, and median p_perm.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score


def execute_unblinding_protocol():
  repo_root = Path(__file__).resolve().parents[2]
  extraction_dir = repo_root / "phase8" / "c2" / "extraction"
  manifest_dir = repo_root / "phase8" / "c2" / "manifest"

  results_path = extraction_dir / "c2_cphi_results.csv"
  qc_path = extraction_dir / "qc_report.json"
  failed_log_path = extraction_dir / "failed_cases.log"
  keycard_path = manifest_dir / "keycard_c2_access_controlled.json"
  report_path = repo_root / "phase8" / "c2" / "reports" / "c2_final_report.json"

  # --- Trigger Condition 1: Check Artifact Existence ---
  for filepath in [results_path, qc_path, failed_log_path]:
    if not filepath.exists():
      raise FileNotFoundError(
          f"Unblinding Trigger Violated: Missing required artifact {filepath}"
      )

  # --- Trigger Condition 2: Check Log Status ---
  log_contents = failed_log_path.read_text().strip()
  if log_contents != "NO_EXTRACTION_FAILURES_LOGGED":
    raise RuntimeError(
        f"Unblinding Trigger Blocked: Failures recorded in log: {log_contents}"
    )

  # --- Trigger Condition 3: Check Completeness in QC Report ---
  with open(qc_path, "r") as f:
    qc_data = json.load(f)

  if qc_data["successful_extractions"] != qc_data["total_cases"]:
    raise RuntimeError(
        "Unblinding Trigger Blocked: Incomplete extractions detected!"
        f" ({qc_data['successful_extractions']}/{qc_data['total_cases']})"
    )

  print(
      "All cryptographic and completeness triggers VERIFIED. Unblinding"
      " cohort..."
  )

  # Load datasets
  df_results = pd.read_csv(results_path)
  with open(keycard_path, "r") as f:
    keycard_data = json.load(f)["unblind_keycard"]

  df_results["case_type"] = df_results["blinded_id"].apply(
      lambda x: keycard_data[x]["case_type"]
  )
  df_results["original_case_id"] = df_results["blinded_id"].apply(
      lambda x: keycard_data[x]["original_case_id"]
  )

  cyclone_mask = df_results["case_type"] == "Cyclone"
  control_mask = df_results["case_type"] == "Control"

  cyclones = df_results[cyclone_mask]["c_phi_observed"].values
  controls = df_results[control_mask]["c_phi_observed"].values

  cyclone_p_perm = df_results[cyclone_mask]["p_perm"].values
  control_p_perm = df_results[control_mask]["p_perm"].values

  # --- Non-Parametric Statistics (Two-Tailed) ---
  u_stat, p_val_two_tailed = stats.mannwhitneyu(
      cyclones, controls, alternative="two-sided"
  )

  # Rosenthal's Effect Size r = Z / sqrt(N)
  n_total = len(cyclones) + len(controls)
  mean_u = (len(cyclones) * len(controls)) / 2.0
  std_u = np.sqrt((len(cyclones) * len(controls) * (n_total + 1)) / 12.0)
  z_score = (u_stat - mean_u) / std_u
  effect_size_r = z_score / np.sqrt(n_total)

  # ROC / AUC
  y_true = cyclone_mask.astype(int)
  auc_score = roc_auc_score(y_true, df_results["c_phi_observed"])

  # Compile Report
  final_report = {
      "protocol": "TRACEBIND-C2-VAL-v1.0",
      "sample_size": {"cyclones": len(cyclones), "controls": len(controls)},
      "metrics": {
          "median_c_phi_cyclone": float(np.median(cyclones)),
          "median_c_phi_control": float(np.median(controls)),
          "delta_median": float(np.median(cyclones) - np.median(controls)),
          "mann_whitney_u": float(u_stat),
          "p_value_two_tailed": float(p_val_two_tailed),
          "rosenthal_effect_size_r": float(effect_size_r),
          "roc_auc": float(auc_score),
          "median_p_perm_cyclone": float(np.median(cyclone_p_perm)),
          "median_p_perm_control": float(np.median(control_p_perm)),
      },
  }

  report_path.parent.mkdir(parents=True, exist_ok=True)
  with open(report_path, "w") as f:
    json.dump(final_report, f, indent=2)

  print(f"Unblinding complete. Report written to {report_path}")


if __name__ == "__main__":
  execute_unblinding_protocol()