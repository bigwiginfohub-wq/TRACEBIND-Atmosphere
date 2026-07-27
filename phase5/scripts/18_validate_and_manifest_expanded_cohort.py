"""
18_validate_and_manifest_expanded_cohort.py
-------------------------------------------
TRACEBIND Phase 5 Post-Acquisition Audit & Provenance Generator

Updates:
1. Replaced deprecated datetime.utcnow() with datetime.now(timezone.utc)
2. Added explicit Grid Shape (Lat x Lon points) to summary table
3. Added Bounding Box ranges (Lat Min/Max, Lon Min/Max) to summary table
"""

import sys
import platform
import json
import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import scipy
import xarray as xr
import pandas as pd

BASE_DIR = Path(r"C:\TRACEBIND-Atmosphere\phase5")
DATA_DIR = BASE_DIR / "data"
MANIFEST_DIR = BASE_DIR / "manifests"
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

def compute_file_hash(filepath: Path, algorithm: str = "sha256") -> str:
    if not filepath.exists():
        return "FILE_NOT_FOUND"
    hasher = hashlib.sha256() if algorithm == "sha256" else hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def get_git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
    except Exception:
        return "UNKNOWN_OR_NOT_A_GIT_REPO"

def audit_and_manifest():
    print("=========================================================================================")
    print("        TRACEBIND PHASE 5: EXPANDED COHORT DATASET AUDIT & MANIFEST GENERATOR           ")
    print("=========================================================================================\n")

    nc_files = sorted(list(DATA_DIR.glob("era5_*_72h.nc")))
    scripts_dir = BASE_DIR / "scripts"
    script_hashes = {f.name: compute_file_hash(f, "sha256") for f in sorted(scripts_dir.glob("*.py"))} if scripts_dir.exists() else {}

    audit_summary = []

    for nc_path in nc_files:
        storm_id = nc_path.stem.replace("era5_", "").replace("_72h", "")
        md5_hash = compute_file_hash(nc_path, "md5")
        sha256_hash = compute_file_hash(nc_path, "sha256")

        try:
            ds = xr.open_dataset(nc_path)
        except Exception as e:
            print(f"[-] FATAL: Failed to open {nc_path.name}: {e}")
            continue

        time_coord = "valid_time" if "valid_time" in ds.coords else "time"
        lat_coord = "latitude" if "latitude" in ds.coords else "lat"
        lon_coord = "longitude" if "longitude" in ds.coords else "lon"

        lats, lons, times = ds[lat_coord].values, ds[lon_coord].values, ds[time_coord].values
        n_timesteps = len(times)
        
        # Grid shape and bounds
        grid_shape = f"{len(lats)}x{len(lons)}"
        lat_bounds = f"[{lats.min():.1f}, {lats.max():.1f}]"
        lon_bounds = f"[{lons.min():.1f}, {lons.max():.1f}]"

        lat_diffs, lon_diffs = np.diff(lats), np.diff(lons)
        lat_res = round(abs(float(lat_diffs[0])), 4) if len(lat_diffs) > 0 else 0.0
        lon_res = round(abs(float(lon_diffs[0])), 4) if len(lon_diffs) > 0 else 0.0
        
        lat_ordering = "Descending" if (len(lat_diffs) > 0 and lat_diffs[0] < 0) else "Ascending"
        lon_ordering = "Ascending" if (len(lon_diffs) > 0 and lon_diffs[0] > 0) else "Descending"
        lat_monotonic = bool(np.all(lat_diffs < 0) if lat_ordering == "Descending" else np.all(lat_diffs > 0))
        lon_monotonic = bool(np.all(lon_diffs > 0) if lon_ordering == "Ascending" else np.all(lon_diffs < 0))

        time_diffs_h = np.diff(times).astype("timedelta64[h]").astype(int)
        time_spacing_valid = bool(np.all(time_diffs_h == 1))

        msl_var = "msl" if "msl" in ds.data_vars else list(ds.data_vars.keys())[0]
        msl_da = ds[msl_var]
        msl_data = msl_da.values
        raw_units = msl_da.attrs.get("units", "unknown")
        units_valid = raw_units.lower() in ["pa", "pascal", "pascals"]

        nan_count = int(np.isnan(msl_data).sum())
        mean_msl, std_msl = float(np.nanmean(msl_data)), float(np.nanstd(msl_data))
        min_msl, max_msl = float(np.nanmin(msl_data)), float(np.nanmax(msl_data))
        temporal_delta_pa = float(np.max(np.abs(msl_data[-1] - msl_data[0])))

        passed_audit = (
            (nan_count == 0) and (max_msl > min_msl) and (temporal_delta_pa > 0.0) and 
            (lat_res == 0.25) and (lon_res == 0.25) and 
            lat_monotonic and lon_monotonic and time_spacing_valid and units_valid
        )

        # Updated datetime call
        timestamp_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        manifest = {
            "metadata": {
                "algorithm_id": "TRACEBIND-P5.0-FROZEN",
                "timestamp_utc": timestamp_iso,
                "git_commit": get_git_commit_hash(),
                "audit_status": "PASSED" if passed_audit else "FAILED"
            },
            "script_provenance_sha256": script_hashes,
            "dataset": {
                "storm": storm_id,
                "input_md5": md5_hash,
                "time_steps": n_timesteps,
                "grid_shape": grid_shape,
                "coordinate_ordering": {"latitude": lat_ordering, "longitude": lon_ordering},
                "bounding_box": {"lat": lat_bounds, "lon": lon_bounds},
                "msl_summary": {
                    "units": raw_units,
                    "temporal_delta_pa": temporal_delta_pa
                }
            }
        }

        with open(MANIFEST_DIR / f"{storm_id}.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        ds.close()
        audit_summary.append({
            "Storm": storm_id,
            "Steps": n_timesteps,
            "Shape": grid_shape,
            "Lat_Bounds": lat_bounds,
            "Lon_Bounds": lon_bounds,
            "Temp_Δ_Pa": f"{temporal_delta_pa:.0f}",
            "Audit": "PASSED" if passed_audit else "FAILED"
        })

    print("=========================================================================================")
    print("                        EXPANDED COHORT AUDIT SUMMARY                                   ")
    print("=========================================================================================")
    print(pd.DataFrame(audit_summary).to_string(index=False))

if __name__ == "__main__":
    audit_and_manifest()