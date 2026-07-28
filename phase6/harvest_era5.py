"""
TRACEBIND Phase 6B: Production ERA5 Harvester v1.2.0
Features: Smart Error Classification, Exponential Backoff + Jitter, State Lifecycle Tracking.
"""

import os
import sys
import time
import json
import random
import hashlib
import argparse
from datetime import datetime, timezone
import numpy as np
import netCDF4 as nc

HARVESTER_VERSION = "1.2.0-STABLE"

def compute_sha256(filepath: str) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(4096), b""):
            sha256.update(block)
    return sha256.hexdigest()

def normalize_bbox_025(bbox: list) -> list:
    lat_min, lat_max, lon_min, lon_max = bbox
    return [
        round(lat_min * 4.0) / 4.0,
        round(lat_max * 4.0) / 4.0,
        round(lon_min * 4.0) / 4.0,
        round(lon_max * 4.0) / 4.0
    ]

def stamp_netcdf_provenance(nc_path: str, system_data: dict, catalog_meta: dict, raw_sha256: str):
    if not os.path.exists(nc_path):
        return

    with nc.Dataset(nc_path, "r+", format="NETCDF4") as ds:
        ds.tracebind_system_id = system_data["system_id"]
        ds.tracebind_ibtracs_sid = str(system_data.get("ibtracs_sid", "N/A"))
        ds.tracebind_cohort_id = system_data["cohort_id"]
        ds.tracebind_lifecycle_stage = system_data["lifecycle_stage"]
        ds.tracebind_analysis_time = system_data["analysis_time"]
        ds.tracebind_catalog_version = catalog_meta.get("catalog_version", "1.0")
        ds.tracebind_harvester_version = HARVESTER_VERSION
        ds.tracebind_selection_rules_hash = catalog_meta.get("selection_rules_hash", "UNKNOWN")
        ds.tracebind_raw_sha256 = raw_sha256
        ds.tracebind_stamped_utc = datetime.now(timezone.utc).isoformat()

def run_quality_control(nc_path: str, expected_shape: tuple = (121, 121)) -> bool:
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

                if np.ma.isMaskedArray(arr) and np.all(arr.mask):
                    print(f"  ❌ [QC FAIL] Variable '{var}' is completely masked.")
                    return False

                filled = np.ma.filled(arr, np.nan)
                if not np.isfinite(filled).any():
                    print(f"  ❌ [QC FAIL] Variable '{var}' contains no finite values.")
                    return False

            lat_key = "latitude" if "latitude" in ds.dimensions else "lat"
            lon_key = "longitude" if "longitude" in ds.dimensions else "lon"

            if lat_key not in ds.variables or lon_key not in ds.variables:
                print("  ❌ [QC FAIL] Missing spatial coordinate arrays.")
                return False

            lats = ds.variables[lat_key][:]
            lons = ds.variables[lon_key][:]
            lat_len, lon_len = len(lats), len(lons)

            lat_diffs = np.diff(lats)
            lon_diffs = np.diff(lons)

            if not (np.all(lat_diffs > 0) or np.all(lat_diffs < 0)) or not np.all(lon_diffs > 0):
                print("  ❌ [QC FAIL] Coordinates are not strictly monotonic.")
                return False

            avg_lat_step = abs(np.mean(lat_diffs))
            avg_lon_step = abs(np.mean(lon_diffs))

            if not (0.24 <= avg_lat_step <= 0.26 and 0.24 <= avg_lon_step <= 0.26):
                print(f"  ❌ [QC FAIL] Invalid grid spacing: ({avg_lat_step:.3f}°, {avg_lon_step:.3f}°)")
                return False

            target_lat, target_lon = expected_shape
            if abs(lat_len - target_lat) > 1 or abs(lon_len - target_lon) > 1:
                print(f"  ❌ [QC FAIL] Shape ({lat_len}x{lon_len}) deviates from target {expected_shape}.")
                return False

            print(f"  ✓ [QC PASS] Shape: ({lat_len}x{lon_len}) | Step: {avg_lat_step:.2f}° | Fields: {required_vars}")
            return True

    except Exception as e:
        print(f"  ❌ [QC FAIL] Read error: {e}")
        return False

def save_catalog_atomically(catalog_path: str, catalog_data: dict):
    tmp_path = catalog_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(catalog_data, f, indent=2)
    os.replace(tmp_path, catalog_path)

def classify_cds_error(error_msg: str) -> str:
    """Classifies CDS API error strings into deterministic harvest states."""
    msg_lower = error_msg.lower()
    if "latest date available" in msg_lower or "data not yet available" in msg_lower:
        return "future_data"
    if "unauthorized" in msg_lower or "authentication failed" in msg_lower or "401" in msg_lower:
        return "auth_error"
    if "502" in msg_lower or "503" in msg_lower or "gateway" in msg_lower or "timeout" in msg_lower:
        return "temporary_failure"
    return "permanent_failure"

def execute_harvest(args):
    catalog_path = "catalog.json"
    if not os.path.exists(catalog_path):
        print(f"❌ Catalog '{catalog_path}' missing.")
        sys.exit(1)

    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    meta = catalog.get("catalog_metadata", {})
    systems = catalog.get("systems", [])

    selected_systems = systems
    if args.system_id:
        selected_systems = [s for s in selected_systems if s["system_id"] == args.system_id]
    if args.cohort:
        selected_systems = [s for s in selected_systems if s["cohort_id"] == args.cohort]
    if args.limit and args.limit > 0:
        selected_systems = selected_systems[:args.limit]

    print("="*60)
    print(f"TRACEBIND Phase 6B: ERA5 Harvester Engine (v{HARVESTER_VERSION})")
    print(f"Targeting {len(selected_systems)} / {len(systems)} system requests.")
    print("="*60)

    if args.dry_run:
        print("\n [DRY-RUN SUMMARY]")
        for sys_rec in selected_systems:
            snapped_bbox = normalize_bbox_025(sys_rec['bounding_box'])
            print(f"  • Request ID: {sys_rec['system_id']:<25} Cohort: {sys_rec['cohort_id']}  BBox: {snapped_bbox}")
        return

    import cdsapi
    client = cdsapi.Client()

    output_dir = "data/raw/era5_nc"
    os.makedirs(output_dir, exist_ok=True)

    completed, skipped, deferred = 0, 0, 0

    for sys_rec in selected_systems:
        sys_id = sys_rec["system_id"]
        out_path = os.path.join(output_dir, sys_rec["era5_filename"])
        status = sys_rec.setdefault("status", {})
        
        # =========================================================
        # NEW: Automatic Deferred State Promotion Check
        # =========================================================
        if status.get("harvest_state") == "deferred":
            available_after_str = status.get("available_after")
            if available_after_str:
                avail_dt = datetime.strptime(available_after_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                if current_utc >= avail_dt:
                    print(f"\n[PROMOTION] System '{sys_id}' reached availability boundary. Promoting 'deferred' → 'pending'.")
                    status["harvest_state"] = "pending"
                    status["deferred_reason"] = None
                    save_catalog_atomically(catalog_path, catalog)
        # =========================================================

        # Resumability & state filter
        current_state = status.get("harvest_state", "pending")
        
        if status.get("downloaded") and status.get("qc_passed") and os.path.exists(out_path):
            print(f"\n[SKIP] '{sys_id}' completed & verified.")
            skipped += 1
            continue

        # Note: Changed from 'future_data' to 'deferred' to match new builder logic
        if current_state == "deferred" and not args.force_retry_future:
            print(f"\n[DEFERRED] '{sys_id}' awaiting ERA5 release. (Available after: {status.get('available_after')})")
            deferred += 1
            continue

        print(f"\n[HARVEST] Requesting {sys_id} ({sys_rec['cohort_name']})...")

        snapped_bbox = normalize_bbox_025(sys_rec["bounding_box"])
        cds_area = [snapped_bbox[1], snapped_bbox[2], snapped_bbox[0], snapped_bbox[3]]
        dt = datetime.strptime(sys_rec["analysis_time"], "%Y-%m-%dT%H:%M:%SZ")

        cds_request = {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': ['10m_u_component_of_wind', '10m_v_component_of_wind', 'mean_sea_level_pressure'],
            'year': dt.strftime('%Y'),
            'month': dt.strftime('%m'),
            'day': dt.strftime('%d'),
            'time': dt.strftime('%H:00'),
            'area': cds_area,
        }

        download_success = False
        max_attempts = 4

        for attempt in range(1, max_attempts + 1):
            try:
                print(f"  → Contacting CDS API (Attempt {attempt}/{max_attempts})...")
                client.retrieve('reanalysis-era5-single-levels', cds_request, out_path)
                download_success = True
                print(f"  ✓ Saved raw CDS payload.")
                break
            except Exception as err:
                err_str = str(err)
                error_class = classify_cds_error(err_str)

                if error_class == "future_data":
                    print(f"  ⏸ [DEFERRED] ERA5 archive delay: Requested date {dt.strftime('%Y-%m-%d')} not yet available.")
                    status["harvest_state"] = "future_data"
                    status["last_error"] = err_str
                    save_catalog_atomically(catalog_path, catalog)
                    deferred += 1
                    break

                elif error_class == "auth_error":
                    print(f"  ❌ [FATAL AUTH ERROR] Invalid CDS API key or credentials. Aborting pipeline.")
                    status["harvest_state"] = "auth_error"
                    save_catalog_atomically(catalog_path, catalog)
                    sys.exit(1)

                elif error_class == "temporary_failure":
                    # Exponential backoff: 30s, 60s, 120s, 240s + random jitter (0-15s)
                    backoff = (30 * (2 ** (attempt - 1))) + random.uniform(0, 15)
                    print(f"  ⚠️ [GATEWAY / TIMEOUT] Attempt {attempt}/{max_attempts} failed. Retrying in {backoff:.1f}s...")
                    if attempt < max_attempts:
                        time.sleep(backoff)

                else:  # permanent_failure
                    print(f"  ❌ [PERMANENT ERROR] Unrecoverable CDS error: {err_str}")
                    status["harvest_state"] = "permanent_failure"
                    status["last_error"] = err_str
                    save_catalog_atomically(catalog_path, catalog)
                    break

        if not download_success:
            continue

        # Post-download processing pipeline
        raw_sha256 = compute_sha256(out_path)
        stamp_netcdf_provenance(out_path, sys_rec, meta, raw_sha256)
        artifact_sha256 = compute_sha256(out_path)

        if run_quality_control(out_path):
            sys_rec["raw_sha256"] = raw_sha256
            sys_rec["artifact_sha256"] = artifact_sha256
            status["downloaded"] = True
            status["qc_passed"] = True
            status["harvest_state"] = "completed"
            completed += 1
            print(f"  ✓ Locked Raw SHA256     : {raw_sha256[:16]}...")
            print(f"  ✓ Locked Artifact SHA256: {artifact_sha256[:16]}...")
            save_catalog_atomically(catalog_path, catalog)

    print("\n" + "="*60)
    print(f"[SESSION COMPLETE] Completed: {completed} | Skipped: {skipped} | Deferred (Future Data): {deferred}")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TRACEBIND ERA5 Harvester Engine")
    parser.add_argument("--dry-run", action="store_true", help="Validate payload without contacting CDS API")
    parser.add_argument("--limit", type=int, default=0, help="Limit total requests")
    parser.add_argument("--system-id", type=str, default=None, help="Target specific system ID")
    parser.add_argument("--cohort", type=str, default=None, help="Target specific cohort ID")
    parser.add_argument("--force-retry-future", action="store_true", help="Force re-querying records marked as future_data")
    args = parser.parse_args()

    execute_harvest(args)