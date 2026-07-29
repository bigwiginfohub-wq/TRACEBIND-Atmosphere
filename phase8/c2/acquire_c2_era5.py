"""TRACEBIND Phase 8 C2 - Automated ERA5 Data Acquisition & Blinding Script

========================================================================
File: phase8/c2/acquire_c2_era5.py
Purpose: Automates ERA5 wind field retrieval via ECMWF CDS API using exact
         c2_cohort_manifest_blinded.json schema.
========================================================================
"""

import json
import logging
from pathlib import Path
import sys
import cdsapi

# Configure Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("C2_Acquisition")

BASE_DIR = Path(__file__).resolve().parent

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


def acquire_era5_cohort():
  if not MANIFEST_PATH or not MANIFEST_PATH.exists():
    logger.error("Blinded manifest missing! Searched in:")
    for path in MANIFEST_CANDIDATES:
      logger.error(f"  - {path}")
    sys.exit(1)

  RAW_DIR.mkdir(parents=True, exist_ok=True)

  logger.info(f"Loaded manifest from: {MANIFEST_PATH}")
  logger.info(f"Target raw directory: {RAW_DIR}")

  with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
    manifest_data = json.load(f)

  cases = manifest_data.get("cases", [])
  if not cases:
    logger.error("No 'cases' list found in the manifest JSON!")
    sys.exit(1)

  logger.info(
      "========================================================================"
  )
  logger.info("TRACEBIND C2 - AUTOMATED ERA5 ACQUISITION & BLINDING")
  logger.info(f"Targeting {len(cases)} blinded cases...")
  logger.info(
      "========================================================================"
  )

  c = cdsapi.Client()

  for idx, case in enumerate(cases, 1):
    token = case.get("blinded_id")
    if not token:
      logger.warning(f"[{idx}/{len(cases)}] Missing 'blinded_id'. Skipping.")
      continue

    target_path = RAW_DIR / f"{token}.nc"

    if target_path.exists():
      logger.info(f"[{idx}/{len(cases)}] {token}.nc already exists. Skipping.")
      continue

    # Bounding Box Resolution
    bbox = case.get("bounding_box")
    if not bbox and "center_coordinates" in case:
      lat, lon = case["center_coordinates"]
      # Default 10x10 deg domain [lat_min, lat_max, lon_min, lon_max]
      bbox = [lat - 5.0, lat + 5.0, lon - 5.0, lon + 5.0]

    # Analysis Time Resolution
    dt_utc = case.get("analysis_time_utc", "2023-05-15T12:00:00Z")

    date_str = dt_utc.split("T")[0]
    time_str = dt_utc.split("T")[1][:5]

    logger.info(
        f"[{idx}/{len(cases)}] Fetching ERA5 field for UUID: {token} | Center:"
        f" {case.get('center_coordinates')}"
    )

    request_params = {
        "product_type": "reanalysis",
        "format": "netcdf",
        "variable": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
        "year": date_str.split("-")[0],
        "month": date_str.split("-")[1],
        "day": date_str.split("-")[2],
        "time": time_str,
        # CDS Area format: [North, West, South, East]
        "area": [bbox[1], bbox[2], bbox[0], bbox[3]],
    }

    try:
      c.retrieve("reanalysis-era5-single-levels", request_params, str(target_path))
      logger.info(f"  └─ Successfully downloaded and saved to: {target_path.name}")
    except Exception as e:
      logger.error(f"  └─ Failed retrieval for {token}: {str(e)}")

  logger.info("\nAcquisition loop finished. Execute validate_c2_netcdf.py next.")


if __name__ == "__main__":
  acquire_era5_cohort()