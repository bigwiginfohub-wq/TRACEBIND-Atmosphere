"""
32_download_era5.py
-----------------------------------------------------
TRACEBIND Phase 5: Automated ERA5 Data Acquisition
Downloads 72-hour single-level Mean Sea Level Pressure (MSLP)
and Geopotential data for newly added expanded storms and negative controls.
"""

from pathlib import Path
import pandas as pd
import cdsapi
from datetime import datetime, timedelta

# Path Setup
SCRIPTS_DIR = Path(__file__).parent
PHASE5_DIR = SCRIPTS_DIR.parent
DATA_DIR = PHASE5_DIR / "data"
CATALOG_PATH = DATA_DIR / "storm_catalog.csv"

def compute_72h_window(analysis_time_str: str):
    """Generates a 72-hour window (-36h to +36h) around analysis time."""
    t_center = datetime.fromisoformat(analysis_time_str.replace("Z", "+00:00"))
    t_start = t_center - timedelta(hours=36)
    t_end = t_center + timedelta(hours=36)
    
    # Collect unique dates in YYYY-MM-DD
    current = t_start
    dates = set()
    while current <= t_end:
        dates.add(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
        
    sorted_dates = sorted(list(dates))
    years = sorted(list(set(d.split("-")[0] for d in sorted_dates)))
    months = sorted(list(set(d.split("-")[1] for d in sorted_dates)))
    days = sorted(list(set(d.split("-")[2] for d in sorted_dates)))
    
    return years, months, days

def compute_bounding_box(lat: float, lon: float, box_size: float = 15.0):
    """Calculates North, West, South, East bounding box centered on lat/lon."""
    half = box_size / 2.0
    north = round(lat + half, 2)
    south = round(lat - half, 2)
    west = round(lon - half, 2)
    east = round(lon + half, 2)
    return [north, west, south, east]

def download_missing_era5():
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(f"Catalog missing at {CATALOG_PATH}")
        
    catalog = pd.read_csv(CATALOG_PATH)
    client = None  # Lazy initialize CDS client only if downloads are needed
    
    print(f"[*] Checking ERA5 data presence for {len(catalog)} catalog entries...\n")
    
    download_count = 0
    skipped_count = 0
    
    for _, row in catalog.iterrows():
        storm_id = str(row['storm_id']).strip()
        target_file = DATA_DIR / f"era5_{storm_id}_72h.nc"
        
        # Check primary or secondary naming presence
        alt_file = DATA_DIR / f"era5_{storm_id}.nc"
        if target_file.exists() or alt_file.exists():
            print(f"  [✓] {storm_id}: File already exists. Skipping.")
            skipped_count += 1
            continue

        if client is None:
            print("[*] Initializing CDS API Client...")
            client = cdsapi.Client()

        years, months, days = compute_72h_window(row['analysis_datetime_utc'])
        area = compute_bounding_box(float(row['center_lat']), float(row['center_lon']), float(row['box_size_deg']))

        request = {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': [
                'mean_sea_level_pressure',
                'geopotential',
            ],
            'year': years,
            'month': months,
            'day': days,
            'time': [
                '00:00', '01:00', '02:00', '03:00', '04:00', '05:00',
                '06:00', '07:00', '08:00', '09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00', '15:00', '16:00', '17:00',
                '18:00', '19:00', '20:00', '21:00', '22:00', '23:00',
            ],
            'area': area,  # [N, W, S, E]
        }

        print(f"\n==================================================")
        print(f" Requesting ERA5: {storm_id} ({row['storm_name']})")
        print(f" Target File: {target_file.name}")
        print(f" Bounding Box: {area}")
        print(f"==================================================")

        try:
            client.retrieve('reanalysis-era5-single-levels', request, str(target_file))
            print(f"  [✓] Successfully downloaded {target_file.name}")
            download_count += 1
        except Exception as e:
            print(f"  [!] Download failed for {storm_id}: {e}")

    print("\n==================================================")
    print("           ERA5 DOWNLOAD SUMMARY                  ")
    print("==================================================")
    print(f" Total Catalog Entries: N = {len(catalog)}")
    print(f" Skipped (Existing):   N = {skipped_count}")
    print(f" Newly Downloaded:     N = {download_count}")
    print("==================================================")

if __name__ == "__main__":
    download_missing_era5()