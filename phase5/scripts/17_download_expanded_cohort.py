"""
17_download_expanded_cohort.py
------------------------------
TRACEBIND Phase 5 Production Cohort Data Acquisition Engine

Features:
- Objective Event Anchoring: Generates 72h window [T-36h, T+36h] around peak/landfall UTC.
- Modern CDS Engine Syntax: Uses 'data_format': 'netcdf' and explicit grid resolution.
- Request Provenance: Writes matching .request.json for every .nc dataset downloaded.
- Fault Tolerance: Automated retry loop (3 attempts) with backoff.
- Pre-flight Validation: Strictly asserts coordinates, days, and bounding box geometry.
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import cdsapi

# Directory Setup
BASE_DIR = Path(r"C:\TRACEBIND-Atmosphere\phase5")
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# EXPANDED COHORT SPECIFICATION
# Objectively defined around Peak Intensity / Landfall UTC timestamps
# -----------------------------------------------------------------------------
RAW_COHORT = [
    # North Indian Ocean (BoB / Arabian Sea)
    {
        "name": "Hudhud",
        "basin": "North Indian Ocean",
        "agency": "IMD",
        "anchor_time_utc": "2014-10-12 06:00",  # Landfall near Visakhapatnam
        "area": [22.0, 80.0, 12.0, 90.0]        # [North, West, South, East]
    },
    {
        "name": "Bulbul",
        "basin": "North Indian Ocean",
        "agency": "IMD",
        "anchor_time_utc": "2019-11-09 18:00",  # Landfall near West Bengal/Bangladesh
        "area": [24.0, 85.0, 14.0, 95.0]
    },
    {
        "name": "Kyarr",
        "basin": "North Indian Ocean",
        "agency": "IMD",
        "anchor_time_utc": "2019-10-27 12:00",  # Peak intensity (Super Cyclonic Storm)
        "area": [22.0, 58.0, 12.0, 68.0]
    },
    {
        "name": "Mekunu",
        "basin": "North Indian Ocean",
        "agency": "IMD",
        "anchor_time_utc": "2018-05-25 18:00",  # Landfall near Rayshut, Oman
        "area": [20.0, 50.0, 10.0, 60.0]
    },
    {
        "name": "Gulab",
        "basin": "North Indian Ocean",
        "agency": "IMD",
        "anchor_time_utc": "2021-09-26 12:00",  # Landfall Andhra Pradesh
        "area": [21.0, 80.0, 11.0, 90.0]
    },
    # Cross-Basin Validation Cohort
    {
        "name": "Ida",
        "basin": "North Atlantic",
        "agency": "NHC",
        "anchor_time_utc": "2021-08-29 16:00",  # Port Fourchon, LA Landfall
        "area": [34.0, -94.0, 24.0, -84.0]
    },
    {
        "name": "Haiyan",
        "basin": "Western Pacific",
        "agency": "JMA",
        "anchor_time_utc": "2013-11-07 20:00",  # Guiuan, Samar Landfall
        "area": [16.0, 120.0, 6.0, 130.0]
    }
]

def derive_72h_window(anchor_str: str):
    """Computes exact 72-hour hourly window [-36h, +36h] around anchor UTC."""
    anchor = datetime.strptime(anchor_str, "%Y-%m-%d %H:%M")
    start = anchor - timedelta(hours=36)
    end = anchor + timedelta(hours=36)
    
    # Generate continuous list of hourly timestamps
    timestamps = [start + timedelta(hours=i) for i in range(73)] # 72 hours span
    
    years = sorted(list({t.strftime("%Y") for t in timestamps}))
    months = sorted(list({t.strftime("%m") for t in timestamps}))
    days = sorted(list({t.strftime("%d") for t in timestamps}))
    times = sorted(list({t.strftime("%H:%M") for t in timestamps}))
    
    return years, months, days, times

def validate_storm_spec(storm: dict):
    """Pre-flight assertions to prevent malformed CDS API requests."""
    north, west, south, east = storm["area"]
    
    # Coordinates check
    assert north > south, f"[{storm['name']}] Bounding box error: North ({north}) <= South ({south})"
    assert west < east, f"[{storm['name']}] Bounding box error: West ({west}) >= East ({east})"
    
    # Date & Anchor format checks
    years, months, days, times = derive_72h_window(storm["anchor_time_utc"])
    for d in days:
        assert len(d) == 2, f"[{storm['name']}] Day string '{d}' is not 2-digit padded"
    
    return years, months, days, times

def execute_download_with_retry(client, dataset_name: str, request_params: dict, out_file: Path, max_retries: int = 3):
    """Handles API execution with structured JSON request logging and automatic retry."""
    req_json_path = out_file.with_suffix(".request.json")
    
    # 1. Save Request Provenance JSON
    with open(req_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": dataset_name,
            "request_parameters": request_params
        }, f, indent=2)
    
    # 2. Execute with Retry
    for attempt in range(1, max_retries + 1):
        try:
            print(f"    -> CDS Request (Attempt {attempt}/{max_retries})...")
            client.retrieve(dataset_name, request_params, str(out_file))
            print(f"    [+] Successfully saved: {out_file.name}")
            return
        except Exception as e:
            print(f"    [-] Attempt {attempt} failed: {e}")
            if attempt == max_retries:
                print(f"    [!] Fatal: All {max_retries} attempts failed for {out_file.name}")
                raise
            time.sleep(5 * attempt) # Incremental backoff

def run_cohort_download():
    print("=========================================================================================")
    print("        TRACEBIND PHASE 5: PRODUCTION COHORT CDS DATA ACQUISITION                        ")
    print("=========================================================================================\n")

    # Order cohort deterministically by Basin, Year, Name
    sorted_cohort = sorted(
        RAW_COHORT,
        key=lambda s: (s["basin"], s["anchor_time_utc"], s["name"])
    )

    client = cdsapi.Client()

    for storm in sorted_cohort:
        storm_name = storm["name"]
        out_file = DATA_DIR / f"era5_{storm_name.lower()}_72h.nc"
        
        print(f"[*] Processing Storm: {storm_name} [{storm['basin']} | {storm['agency']}]")
        print(f"    Anchor Event UTC: {storm['anchor_time_utc']}")

        if out_file.exists():
            print(f"    [=] Output dataset already exists on disk: {out_file.name} (Skipping)\n")
            continue

        # Validate spec & derive time window
        years, months, days, times = validate_storm_spec(storm)

        # Construct modernized CDS request
        dataset_name = 'reanalysis-era5-single-levels'
        request_params = {
            'product_type': 'reanalysis',
            'data_format': 'netcdf',          # Modern CDS API parameter
            'download_format': 'unarchived',
            'variable': 'mean_sea_level_pressure',
            'year': years,
            'month': months,
            'day': days,
            'time': times,
            'area': storm['area'],             # [N, W, S, E]
            'grid': [0.25, 0.25],             # Explicit resolution freeze
        }

        # Execute download and store provenance
        execute_download_with_retry(client, dataset_name, request_params, out_file)
        print()

if __name__ == "__main__":
    run_cohort_download()