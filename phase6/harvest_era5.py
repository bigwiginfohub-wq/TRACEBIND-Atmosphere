"""
Phase 6A: ERA5 Cutout Downloader with Exhaustive Provenance Stamping
"""
import os
import json
import subprocess
from datetime import datetime
import cdsapi
import xarray as xr
from validate_dataset import validate_netcdf

# FROZEN SCIENTIFIC DEFINITIONS
TRACEBIND_VERSION = "1.0.0"
DESCRIPTOR_VERSION = "v1_frozen_5D"
DESCRIPTOR_SET = ("GE", "LE", "C_orient", "A_radial", "S_orient")

def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        return "UNKNOWN_COMMIT"

def harvest_and_stamp(entry: dict, data_dir: str = "data"):
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, entry["era5_filename"])
    
    dt = datetime.fromisoformat(entry["analysis_time"].replace("Z", "+00:00"))
    lat_min, lat_max, lon_min, lon_max = entry["bounding_box"]

    request_params = {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': ['mean_sea_level_pressure', '10m_u_component_of_wind', '10m_v_component_of_wind'],
        'year': f"{dt.year:04d}",
        'month': f"{dt.month:02d}",
        'day': f"{dt.day:02d}",
        'time': f"{dt.hour:02d}:00",
        'area': [lat_max, lon_min, lat_min, lon_max],
    }

    c = cdsapi.Client()
    temp_file = out_path + ".tmp"
    c.retrieve('reanalysis-era5-single-levels', request_params, temp_file)

    # Stamp Rich Provenance Attributes
    ds = xr.open_dataset(temp_file)
    ds.attrs["Tracebind Version"] = TRACEBIND_VERSION
    ds.attrs["Git Commit"] = get_git_commit()
    ds.attrs["Extraction Script"] = "harvest_era5.py"
    ds.attrs["Creation Time UTC"] = datetime.utcnow().isoformat() + "Z"
    ds.attrs["ERA5 Product"] = "reanalysis-era5-single-levels"
    ds.attrs["Bounding Box"] = str(entry["bounding_box"])
    ds.attrs["Grid Resolution"] = "0.25 deg"
    ds.attrs["Descriptor Version"] = DESCRIPTOR_VERSION
    ds.attrs["Frozen Descriptors"] = ",".join(DESCRIPTOR_SET)
    ds.attrs["System ID"] = entry["system_id"]

    ds.to_netcdf(out_path)
    ds.close()
    
    if os.path.exists(temp_file):
        os.remove(temp_file)

    # Validate Immediately
    qc = validate_netcdf(out_path)
    entry["status"]["downloaded"] = True
    entry["status"]["qc_passed"] = qc["passed"]
    print(f"[{entry['system_id']}] Download complete. QC Passed: {qc['passed']}")