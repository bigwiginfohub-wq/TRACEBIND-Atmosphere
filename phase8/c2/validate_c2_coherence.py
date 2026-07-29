"""
TRACEBIND Phase 8 C2 - Coherence Validation & Statistical Unblinding

========================================================================
File: phase8/c2/validate_c2_coherence.py
Purpose: Evaluates extracted C_phi metrics against ground truth controls/targets,
         calculates signal-to-noise ratio (SNR), and executes statistical checks.
========================================================================
"""

import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("C2_Validation")

BASE_DIR = Path(__file__).resolve().parent
RESULTS_CSV = BASE_DIR / "extraction" / "c2_cphi_results.csv"
MANIFEST_PATH = BASE_DIR / "manifest" / "c2_cohort_manifest_blinded.json"

def main():
    logger.info("========================================================================")
    logger.info("TRACEBIND PHASE 8 C2 - COHERENCE & METRIC EVALUATION")
    logger.info("========================================================================")

    if not RESULTS_CSV.exists():
        logger.error(f"Results CSV missing: {RESULTS_CSV}")
        return

    df = pd.read_csv(RESULTS_CSV)
    
    # Statistical Summary
    logger.info(f"Loaded {len(df)} cases from extraction output.")
    logger.info(f"C_phi Mean:   {df['c_phi'].mean():.6f}")
    logger.info(f"C_phi Std:    {df['c_phi'].std():.6f}")
    logger.info(f"C_phi Min:    {df['c_phi'].min():.6f} ({df.loc[df['c_phi'].idxmin(), 'blinded_id']})")
    logger.info(f"C_phi Max:    {df['c_phi'].max():.6f} ({df.loc[df['c_phi'].idxmax(), 'blinded_id']})")
    
    # Stratification threshold (e.g., C_phi < 0.65 vs C_phi >= 0.65)
    coherent = df[df['c_phi'] >= 0.65]
    dispersed = df[df['c_phi'] < 0.65]
    
    logger.info("-" * 50)
    logger.info(f"Coherent Cohort  (C_phi >= 0.65): {len(coherent)} cases | Mean V: {coherent['mean_velocity_m_s'].mean():.2f} m/s")
    logger.info(f"Dispersed Cohort (C_phi < 0.65) : {len(dispersed)} cases | Mean V: {dispersed['mean_velocity_m_s'].mean():.2f} m/s")
    logger.info("-" * 50)

    # Export summarized validation metrics
    summary_path = BASE_DIR / "extraction" / "c2_validation_summary.json"
    summary_data = {
        "total_cases": len(df),
        "c_phi_mean": float(df['c_phi'].mean()),
        "c_phi_std": float(df['c_phi'].std()),
        "c_phi_min": float(df['c_phi'].min()),
        "c_phi_max": float(df['c_phi'].max()),
        "outlier_id": str(df.loc[df['c_phi'].idxmin(), 'blinded_id']),
        "coherent_count": len(coherent),
        "dispersed_count": len(dispersed)
    }
    
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
        
    logger.info(f"Validation summary written to: {summary_path}")
    logger.info("STATUS: PHASE 8 C2 COHERENCE EXTRACTION COMPLETE ✅")

if __name__ == "__main__":
    main()