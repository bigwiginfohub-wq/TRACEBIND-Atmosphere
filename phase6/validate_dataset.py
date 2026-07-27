"""
Phase 6A: NetCDF Quality Control and Physical Sanity Verification
"""
import xarray as xr
import numpy as np
from typing import Dict, Any, Tuple

PHYSICAL_LIMITS = {
    "MSLP_HPA_MIN": 850.0,
    "MSLP_HPA_MAX": 1080.0
}

def validate_netcdf(nc_path: str, expected_res: float = 0.25) -> Dict[str, Any]:
    qc_summary = {
        "passed": False,
        "grid_ok": False,
        "monotonic_ok": False,
        "physical_range_ok": False,
        "errors": []
    }

    try:
        ds = xr.open_dataset(nc_path)
    except Exception as e:
        qc_summary["errors"].append(f"Cannot open NetCDF: {str(e)}")
        return qc_summary

    # Coordinate Name Discovery
    lat_key = next((k for k in ds.coords if k.lower() in ["latitude", "lat"]), None)
    lon_key = next((k for k in ds.coords if k.lower() in ["longitude", "lon"]), None)
    time_key = next((k for k in ds.coords if k.lower() in ["time", "valid_time"]), None)

    if not (lat_key and lon_key and time_key):
        qc_summary["errors"].append(f"Missing coordinates. Found: {list(ds.coords.keys())}")
        ds.close()
        return qc_summary

    lats = ds[lat_key].values
    lons = ds[lon_key].values

    # Monotonicity & Uniform Spacing
    lat_diffs = np.diff(lats)
    lon_diffs = np.diff(lons)
    is_lat_mono = np.all(lat_diffs > 0) or np.all(lat_diffs < 0)
    is_lon_mono = np.all(lon_diffs > 0) or np.all(lon_diffs < 0)

    if is_lat_mono and is_lon_mono:
        qc_summary["monotonic_ok"] = True
    else:
        qc_summary["errors"].append("Grid is not strictly monotonic.")

    lat_space_ok = np.allclose(np.abs(lat_diffs), expected_res, atol=1e-3)
    lon_space_ok = np.allclose(np.abs(lon_diffs), expected_res, atol=1e-3)
    
    if lat_space_ok and lon_space_ok:
        qc_summary["grid_ok"] = True
    else:
        qc_summary["errors"].append(f"Grid spacing deviates from {expected_res} deg.")

    # Data Integrity & Physical Limits
    phys_ok = True
    for var_name in ds.data_vars:
        arr = ds[var_name].values
        if np.isnan(arr).any() or np.isinf(arr).any():
            phys_ok = False
            qc_summary["errors"].append(f"NaN/Inf in variable {var_name}")

        if "msl" in var_name.lower() or "pressure" in var_name.lower():
            arr_hpa = arr / 100.0 if np.mean(arr) > 10000 else arr
            min_p, max_p = np.min(arr_hpa), np.max(arr_hpa)
            if min_p < PHYSICAL_LIMITS["MSLP_HPA_MIN"] or max_p > PHYSICAL_LIMITS["MSLP_HPA_MAX"]:
                phys_ok = False
                qc_summary["errors"].append(f"MSLP out of physical bounds: [{min_p:.1f}, {max_p:.1f}] hPa")

    qc_summary["physical_range_ok"] = phys_ok
    qc_summary["passed"] = (
        qc_summary["grid_ok"] and 
        qc_summary["monotonic_ok"] and 
        qc_summary["physical_range_ok"]
    )

    ds.close()
    return qc_summary