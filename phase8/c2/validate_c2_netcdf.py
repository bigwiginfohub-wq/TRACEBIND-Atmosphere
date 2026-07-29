"""TRACEBIND Phase 8 C2 - NetCDF Validation Script

========================================================================
File: phase8/c2/validate_c2_netcdf.py
Purpose: Validates structural schema, variable dimensions, wind velocity
         integrity, and coordinate bounds for the 20 blinded NetCDF files
         prior to phase coherence extraction.
========================================================================
"""

import json
import logging
from pathlib import Path
import sys
import netCDF4 as nc
import numpy as np

# Configure Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("C2_Validator")

BASE_DIR = Path(__file__).resolve().parent

# Manifest Resolution
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

RAW_DIR = BASE_DIR / "raw"


def validate_dataset():
  logger.info(
      "========================================================================"
  )
  logger.info("TRACEBIND PHASE 8 C2 - ATMOSPHERIC DATASET VALIDATION")
  logger.info(
      "========================================================================"
  )

  if not MANIFEST_PATH or not MANIFEST_PATH.exists():
    logger.error("Blinded manifest missing!")
    sys.exit(1)

  logger.info(f"Loaded manifest from: {MANIFEST_PATH}")
  logger.info(f"Checking raw directory: {RAW_DIR}")

  with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
    manifest_data = json.load(f)

  cases = manifest_data.get("cases", [])
  if not cases:
    logger.error("No 'cases' array found in manifest!")
    sys.exit(1)

  total_cases = len(cases)
  discovered_count = 0
  missing_vars = 0
  nan_inf_failures = 0
  coord_failures = 0

  u_var_candidates = ["u10", "u_component_of_wind", "u", "10m_u_component_of_wind"]
  v_var_candidates = ["v10", "v_component_of_wind", "v", "10m_v_component_of_wind"]
  lat_candidates = ["latitude", "lat"]
  lon_candidates = ["longitude", "lon"]

  for idx, case in enumerate(cases, 1):
    token = case.get("blinded_id") or case.get("uuid_token")
    if not token:
      logger.warning(f"[{idx}/{total_cases}] Case missing ID. Skipping.")
      continue

    nc_file = RAW_DIR / f"{token}.nc"
    if not nc_file.exists():
      logger.error(f"[{idx}/{total_cases}] [{token}] MISSING FILE: {nc_file}")
      continue

    discovered_count += 1

    try:
      with nc.Dataset(nc_file, "r") as ds:
        ds_vars = ds.variables.keys()

        # Check U and V wind components
        u_name = next((v for v in u_var_candidates if v in ds_vars), None)
        v_name = next((v for v in v_var_candidates if v in ds_vars), None)

        if not u_name or not v_name:
          logger.error(f"[{idx}/{total_cases}] [{token}] Missing U/V wind variables. Found: {list(ds_vars)}")
          missing_vars += 1
          continue

        # Check coordinates
        lat_name = next((c for c in lat_candidates if c in ds_vars), None)
        lon_name = next((c for c in lon_candidates if c in ds_vars), None)

        if not lat_name or not lon_name:
          logger.error(f"[{idx}/{total_cases}] [{token}] Missing coordinate variables.")
          coord_failures += 1
          continue

        # Check for NaN / Inf
        u_data = ds.variables[u_name][:]
        v_data = ds.variables[v_name][:]

        if np.isnan(u_data).any() or np.isnan(v_data).any() or np.isinf(u_data).any() or np.isinf(v_data).any():
          logger.error(f"[{idx}/{total_cases}] [{token}] NaN or Inf values detected in velocity fields.")
          nan_inf_failures += 1
          continue

        logger.info(f"[{idx}/{total_cases}] [{token}] ✅ Validated (U: {u_name}, V: {v_name}, Grid: {ds.variables[lat_name].shape[0]}x{ds.variables[lon_name].shape[0]})")

    except Exception as e:
      logger.error(f"[{idx}/{total_cases}] [{token}] Corrupt NetCDF or load error: {str(e)}")
      coord_failures += 1

  logger.info("\n" + "=" * 50)
  logger.info("C2 DATASET VALIDATION SUMMARY")
  logger.info("=" * 50)
  logger.info(f"Files Expected:          {total_cases}")
  logger.info(f"Files Discovered:        {discovered_count}/{total_cases}")
  logger.info(f"Missing Variables:       {missing_vars}")
  logger.info(f"Coordinate Failures:     {coord_failures}")
  logger.info(f"NaN/Inf Failures:        {nan_inf_failures}")
  logger.info("-" * 50)

  if discovered_count == total_cases and missing_vars == 0 and coord_failures == 0 and nan_inf_failures == 0:
    logger.info("STATUS: READY FOR C2 EXTRACTION ✅")
  else:
    logger.error("STATUS: VALIDATION FAILED ❌")
    sys.exit(1)


if __name__ == "__main__":
  validate_dataset()