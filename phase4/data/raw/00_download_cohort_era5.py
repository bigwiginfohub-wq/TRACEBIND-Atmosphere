"""
00_download_cohort_era5.py
--------------------------
Downloads 72-hour ERA5 hourly single-level/pressure netCDF files 
for the remaining TRACEBIND cyclone cohort: Fani, Mocha, Yaas, Sidr, Nargis.
"""

import os
from pathlib import Path
import cdsapi

RAW_DIR = Path(r"C:\TRACEBIND-Atmosphere\phase4\data\raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# 72-hour bounding metadata & spatial sub-domains (N, W, S, E)
STORM_CONFIGS = [
    {
        "name": "Fani",
        "file": "era5_fani_72h.nc",
        "year": "2019",
        "month": "05",
        "days": ["01", "02", "03", "04"],
        "area": [25.0, 80.0, 5.0, 100.0],  # North, West, South, East
    },
    {
        "name": "Mocha",
        "file": "era5_mocha_72h.nc",
        "year": "2023",
        "month": "05",
        "days": ["12", "13", "14", "15"],
        "area": [25.0, 80.0, 5.0, 100.0],
    },
    {
        "name": "Yaas",
        "file": "era5_yaas_72h.nc",
        "year": "2021",
        "month": "05",
        "days": ["24", "25", "26", "27"],
        "area": [25.0, 80.0, 5.0, 100.0],
    },
    {
        "name": "Sidr",
        "file": "era5_sidr_72h.nc",
        "year": "2007",
        "month": "11",
        "days": ["13", "14", "15", "16"],
        "area": [25.0, 80.0, 5.0, 100.0],
    },
    {
        "name": "Nargis",
        "file": "era5_nargis_72h.nc",
        "year": "2008",
        "month": "04",
        "days": ["29", "30"],
        "extra_month": {"month": "05", "days": ["01", "02"]},
        "area": [23.0, 80.0, 5.0, 100.0],
    },
]

def download_storm_era5(storm: dict, client: cdsapi.Client):
    target_path = RAW_DIR / storm["file"]
    if target_path.exists():
        print(f"[✓] {storm['file']} already exists. Skipping.")
        return

    print(f"\n================================================================================")
    print(f"REQUESTING ERA5 DATA FOR: {storm['name'].upper()}")
    print(f"Target File: {target_path}")
    print(f"================================================================================")

    # Base retrieval query parameters for single-level variables (MSLP & Geopotential)
    request = {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': [
            'mean_sea_level_pressure',
            'geopotential',
        ],
        'year': storm['year'],
        'month': storm['month'],
        'day': storm['days'],
        'time': [
            '00:00', '01:00', '02:00', '03:00', '04:00', '05:00',
            '06:00', '07:00', '08:00', '09:00', '10:00', '11:00',
            '12:00', '13:00', '14:00', '15:00', '16:00', '17:00',
            '18:00', '19:00', '20:00', '21:00', '22:00', '23:00',
        ],
        'area': storm['area'],  # [N, W, S, E]
    }

    try:
        client.retrieve('reanalysis-era5-single-levels', request, str(target_path))
        print(f"[✓] Download successful: {target_path.name}")
    except Exception as e:
        print(f"[!] Download failed for {storm['name']}: {e}")

def main():
    c = cdsapi.Client()
    for storm in STORM_CONFIGS:
        download_storm_era5(storm, c)

if __name__ == "__main__":
    main()