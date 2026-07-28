"""
TRACEBIND Phase 6B: Standalone Pilot Verification & Provenance Inspector
Validates and stamps NetCDF files already downloaded on disk.
"""

import os
import json
import hashlib
from datetime import datetime, timezone
import numpy as np
import netCDF4 as nc

def compute_sha256(filepath: str) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(4096), b""):
            sha256.update(block)
    return sha256.hexdigest()

def stamp_netcdf_provenance(nc_path: str, system_data: dict, catalog_meta: dict):
    if not os.path.exists(nc_path):
        return

    with nc.Dataset(nc_path, "r+", format="NETCDF4") as ds:
        ds.tracebind_system_id = system_data["system_id"]
        ds.tracebind_ibtracs_sid = str(system_data.get("ibtracs_sid", "N/A"))
        ds.tracebind_cohort_id = system_data["cohort_id"]
        ds.tracebind_lifecycle_stage = system_data["lifecycle_stage"]
        ds.tracebind_analysis_time = system_data["analysis_time"]
        ds.tracebind_catalog_version = catalog_meta.get("catalog_version", "1.0")
        ds.tracebind_selection_rules_hash = catalog_meta.get("selection_rules_hash", "UNKNOWN")
        ds.tracebind_stamped_utc = datetime.now(timezone.utc).isoformat()

def run_quality_control(nc_path: str) -> bool:
    try:
        with nc.Dataset(nc_path, "r") as ds:
            required_vars = ["u10", "v10", "msl"]

            for var in required_vars:
                if var not in ds.variables:
                    print(f"  ❌ [QC FAIL] Missing variable '{var}'")
                    return False

                arr = ds.variables[var][:]

                if arr.size == 0:
                    print(f"  ❌ [QC FAIL] Variable '{var}' is empty.")
                    return False

                if np.ma.isMaskedArray(arr):
                    if np.all(np.ma.getmaskarray(arr)):
                        print(f"  ❌ [QC FAIL] Variable '{var}' is completely masked.")
                        return False

                if np.isnan(np.asarray(arr)).all():
                    print(f"  ❌ [QC FAIL] Variable '{var}' contains only NaNs.")
                    return False

            lat_dim = ds.dimensions.get("latitude") or ds.dimensions.get("lat")
            lon_dim = ds.dimensions.get("longitude") or ds.dimensions.get("lon")

            if lat_dim is None or lon_dim is None:
                print("  ❌ [QC FAIL] Spatial dimensions missing.")
                return False

            lat_len, lon_len = len(lat_dim), len(lon_dim)
            if lat_len < 50 or lon_len < 50:
                print(f"  ❌ [QC FAIL] Truncated grid shape: ({lat_len}, {lon_len})")
                return False

            print(f"  ✓ [QC PASS] Fields {required_vars} verified (Grid: {lat_len}x{lon_len}).")
            return True

    except Exception as e:
        print(f"  ❌ [QC FAIL] NetCDF read exception: {e}")
        return False

def save_catalog_atomically(catalog_path: str, catalog_data: dict):
    tmp_path = catalog_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(catalog_data, f, indent=2)
    os.replace(tmp_path, catalog_path)

def verify_pilot_files():
    catalog_path = "catalog.json"
    if not os.path.exists(catalog_path):
        print("❌ Catalog missing.")
        return

    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    meta = catalog.get("catalog_metadata", {})
    systems = catalog.get("systems", [])
    output_dir = "data/raw/era5_nc"

    print("="*60)
    print("TRACEBIND Phase 6B: Local Pilot Artifact Inspector")
    print("="*60)

    verified_count = 0

    for sys_rec in systems:
        out_path = os.path.join(output_dir, sys_rec["era5_filename"])
        if not os.path.exists(out_path):
            continue

        sys_id = sys_rec["system_id"]
        print(f"\n[INSPECTING] System: {sys_id}")
        print(f"  File: '{out_path}'")

        # 1. Stamp provenance metadata headers
        stamp_netcdf_provenance(out_path, sys_rec, meta)

        # 2. Inspect global attributes
        with nc.Dataset(out_path, "r") as ds:
            prov_id = getattr(ds, "tracebind_system_id", "MISSING")
            prov_version = getattr(ds, "tracebind_catalog_version", "MISSING")
            prov_utc = getattr(ds, "tracebind_stamped_utc", "MISSING")
            
            print(f"  • Header [tracebind_system_id]     : {prov_id}")
            print(f"  • Header [tracebind_catalog_version]: {prov_version}")
            print(f"  • Header [tracebind_stamped_utc]    : {prov_utc}")

        # 3. Run Quality Control Gate
        if run_quality_control(out_path):
            file_hash = compute_sha256(out_path)
            sys_rec.setdefault("status", {})
            sys_rec["sha256"] = file_hash
            sys_rec["status"]["downloaded"] = True
            sys_rec["status"]["qc_passed"] = True
            verified_count += 1
            print(f"  ✓ Locked SHA256: {file_hash}")

    save_catalog_atomically(catalog_path, catalog)

    print("\n" + "="*60)
    print(f"[SUMMARY] Successfully inspected and locked {verified_count} local pilot artifacts.")
    print("="*60)

if __name__ == "__main__":
    verify_pilot_files()