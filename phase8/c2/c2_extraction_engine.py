"""
TRACEBIND Phase 8 C2 - Kinematic Phase Coherence Extraction Engine

========================================================================
File: phase8/c2/c2_extraction_engine.py
Purpose: Extracts C2 kinematic phase coherence (C_phi) from ERA5 NetCDF
         wind fields for blinded test cohort.
========================================================================
"""

import os
import sys
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import netCDF4 as nc

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("C2_Extraction")

BASE_DIR = Path(__file__).resolve().parent

# Manifest Resolution
MANIFEST_CANDIDATES = [
    BASE_DIR / "manifest" / "c2_cohort_manifest_blinded.json",
    BASE_DIR / "c2_cohort_manifest_blinded.json",
    BASE_DIR.parent / "c2_cohort_manifest_blinded.json",
    Path.cwd() / "c2_cohort_manifest_blinded.json",
    Path.cwd() / "phase8" / "c2" / "manifest" / "c2_cohort_manifest_blinded.json"
]

MANIFEST_PATH = None
for candidate in MANIFEST_CANDIDATES:
    if candidate.exists():
        MANIFEST_PATH = candidate
        break

RAW_DIR = BASE_DIR / "raw"
OUTPUT_DIR = BASE_DIR / "extraction"

def get_variable(dataset, candidate_names):
    """Retrieve first existing variable from netCDF dataset matching candidate list."""
    for var in candidate_names:
        if var in dataset.variables:
            return var
    return None

def compute_phase_coherence(u_data, v_data):
    """
    Computes kinematic phase coherence metric (C_phi) from zonal and meridional 
    wind velocity arrays across spatial domain.
    """
    # Squeeze out single-time or single-level dimensions if present
    u = np.squeeze(u_data)
    v = np.squeeze(v_data)

    # Compute phase angle theta = arctan2(v, u)
    phase_angles = np.arctan2(v, u)

    # Compute order parameter / phase coherence: |1/N * sum(exp(i * theta))|
    complex_phases = np.exp(1j * phase_angles)
    c_phi = np.abs(np.mean(complex_phases))

    # Compute mean wind velocity magnitude
    v_mag = np.mean(np.sqrt(u**2 + v**2))

    return float(c_phi), float(v_mag)

def run_extraction():
    logger.info("========================================================================")
    logger.info("TRACEBIND PHASE 8 C2 - PHASE COHERENCE EXTRACTION ENGINE")
    logger.info("========================================================================")

    if not MANIFEST_PATH or not MANIFEST_PATH.exists():
        logger.error("Blinded manifest not found!")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUTPUT_DIR / "c2_cphi_results.csv"

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    cases = manifest_data.get("cases", [])
    if not cases:
        logger.error("No 'cases' list found in manifest!")
        sys.exit(1)

    results = []
    success_count = 0

    u_candidates = ["u10", "u_component_of_wind", "u", "10m_u_component_of_wind"]
    v_candidates = ["v10", "v_component_of_wind", "v", "10m_v_component_of_wind"]

    for idx, case in enumerate(cases, 1):
        token = case.get("blinded_id") or case.get("uuid_token")
        if not token:
            logger.warning(f"[{idx}/{len(cases)}] Skipping entry missing ID.")
            continue

        nc_path = RAW_DIR / f"{token}.nc"
        if not nc_path.exists():
            logger.error(f"[{idx}/{len(cases)}] NetCDF missing: {nc_path}")
            continue

        try:
            with nc.Dataset(nc_path, "r") as ds:
                u_var = get_variable(ds, u_candidates)
                v_var = get_variable(ds, v_candidates)

                if not u_var or not v_var:
                    raise KeyError(f"Neither {u_candidates} nor {v_candidates} found in {list(ds.variables.keys())}")

                u_data = ds.variables[u_var][:]
                v_data = ds.variables[v_var][:]

                c_phi, mean_velocity = compute_phase_coherence(u_data, v_data)

                results.append({
                    "blinded_id": token,
                    "center_lat": case.get("center_coordinates", [None, None])[0],
                    "center_lon": case.get("center_coordinates", [None, None])[1],
                    "c_phi": c_phi,
                    "mean_velocity_m_s": mean_velocity,
                    "u_var_used": u_var,
                    "v_var_used": v_var
                })

                success_count += 1
                logger.info(f"[{idx}/{len(cases)}] {token} -> C_phi: {c_phi:.6f} | V_mean: {mean_velocity:.2f} m/s")

        except Exception as e:
            logger.error(f"Failed extraction for {token}: {str(e)}")

    if results:
        df = pd.DataFrame(results)
        df.to_csv(out_csv, index=False)

    logger.info(f"Extraction complete. Results written to {out_csv}. Successful: {success_count}/{len(cases)}")

if __name__ == "__main__":
    run_extraction()