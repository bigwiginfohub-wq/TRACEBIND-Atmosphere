"""
Phase 6A.5: TC & Monsoon Cohort Builder (Deterministic State Machine & Scheduled Deferral)
"""

import os
import json
import hashlib
import urllib.request
import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

def compute_sha256(filepath: str) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(4096), b""):
            sha256.update(block)
    return sha256.hexdigest()

def normalize_longitude(lon: float) -> float:
    """Normalizes longitude to standard range [-180.0, 180.0]."""
    return round(((lon + 180.0) % 360.0) - 180.0, 4)

def ensure_raw_dataset(sources_cfg: dict) -> str:
    local_path = sources_cfg["dataset_sources"]["ibtracs"]["local_path"]
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    if not os.path.exists(local_path):
        url = sources_cfg["dataset_sources"]["ibtracs"]["remote_url"]
        print(f"[FETCH] Frozen raw dataset missing. Downloading to '{local_path}'...")
        urllib.request.urlretrieve(url, local_path)
    
    file_hash = compute_sha256(local_path)
    print(f"[INPUT LOCK] IBTrACS Local Raw SHA256: {file_hash}")
    return local_path

def build_bbox(lat: float, lon: float, pad: float) -> list:
    lat_min = round(max(-90.0, lat - pad), 2)
    lat_max = round(min(90.0, lat + pad), 2)
    lon_min = normalize_longitude(lon - pad)
    lon_max = normalize_longitude(lon + pad)
    return [lat_min, lat_max, lon_min, lon_max]

def get_cds_latest_available_time(fallback_days: int = 5) -> datetime:
    """Calculates CDS availability horizon locked strictly to UTC."""
    return datetime.now(timezone.utc) - timedelta(days=fallback_days)

def process_tc_samples():
    with open("sources.yaml", "r", encoding="utf-8") as f:
        sources = yaml.safe_load(f)
    with open("selection_rules.yaml", "r", encoding="utf-8") as f:
        rules = yaml.safe_load(f)

    raw_path = ensure_raw_dataset(sources)
    pad_deg = rules.get("bounding_box_padding_deg", 15.0)
    target_n = rules.get("target_samples_per_cohort", 25)
    latency_days = rules.get("era5_latency_days", 5)

    raw_df = pd.read_csv(raw_path, low_memory=False, skiprows=[1])
    raw_df['USA_WIND'] = pd.to_numeric(raw_df['USA_WIND'], errors='coerce')
    raw_df['USA_PRES'] = pd.to_numeric(raw_df['USA_PRES'], errors='coerce')
    raw_df['LAT'] = pd.to_numeric(raw_df['LAT'], errors='coerce')
    raw_df['LON'] = pd.to_numeric(raw_df['LON'], errors='coerce')
    raw_df['ISO_TIME'] = pd.to_datetime(raw_df['ISO_TIME'], utc=True, errors='coerce')
    
    df = raw_df.dropna(subset=['LAT', 'LON', 'ISO_TIME', 'USA_WIND'])

    # Determine current archive boundary
    latest_cds_utc = get_cds_latest_available_time(fallback_days=latency_days)
    print(f"[CDS SYNCHRONIZATION] Archive limit locked to: {latest_cds_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}")

    systems = []
    seen_sids = set()

    for key, c_cfg in rules["cohort_definitions"].items():
        cohort_id = c_cfg.get("cohort_id")
        if cohort_id == "G":
            continue # Processed by random controls builder

        constraints = c_cfg.get("selection_constraints", {})
        
        subset = df.copy()
        if "usa_wind_min_kt" in constraints:
            subset = subset[subset['USA_WIND'] >= constraints["usa_wind_min_kt"]]
        if "usa_wind_max_kt" in constraints:
            subset = subset[(subset['USA_WIND'] > 0) & (subset['USA_WIND'] <= constraints["usa_wind_max_kt"])]
        if "basin" in constraints:
            subset = subset[subset['BASIN'] == constraints["basin"]]

        # Vector Deduplication: Isolate peak intensity state per storm
        subset = (
            subset
            .sort_values(by=['USA_WIND', 'ISO_TIME'], ascending=[False, False])
            .drop_duplicates(subset=['SID'], keep='first')
        )

        count = 0
        for _, row in subset.iterrows():
            sid = str(row['SID']).strip()
            if sid in seen_sids or count >= target_n:
                continue

            lat = float(row['LAT'])
            lon = normalize_longitude(float(row['LON']))
            iso_timestamp = row['ISO_TIME']
            sys_id = f"TC_{iso_timestamp.year}_{sid}"

            # Evaluate Harvestability State against CDS Archive Horizon
            is_harvestable = iso_timestamp <= latest_cds_utc
            
            # Predict harvest readiness time: Event Time + Publication Latency
            available_after = (iso_timestamp + timedelta(days=latency_days)).strftime("%Y-%m-%dT%H:%M:%SZ")

            if is_harvestable:
                status_block = {
                    "downloaded": False,
                    "qc_passed": False,
                    "processed": False,
                    "harvest_state": "pending",
                    "deferred_reason": None,
                    "available_after": available_after
                }
            else:
                status_block = {
                    "downloaded": False,
                    "qc_passed": False,
                    "processed": False,
                    "harvest_state": "deferred",
                    "deferred_reason": "ERA5_NOT_YET_AVAILABLE",
                    "available_after": available_after
                }

            systems.append({
                "system_id": sys_id,
                "ibtracs_sid": sid,
                "cohort_id": cohort_id,
                "cohort_name": c_cfg["name"],
                "system_name": str(row['NAME']).title(),
                "event_year": int(iso_timestamp.year),
                "system_class": constraints.get("system_class", "TC"),
                "subclass": constraints.get("subclass", "Observed Vortex"),
                "basin": str(row['BASIN']),
                "source_dataset": "ERA5",
                "source_reference": f"IBTrACS_{sid}",
                "analysis_time": iso_timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "center_lat": round(lat, 4),
                "center_lon": lon,
                "min_pressure_hpa": float(row['USA_PRES']) if not np.isnan(row['USA_PRES']) else None,
                "max_wind_kt": float(row['USA_WIND']),
                "lifecycle_stage": c_cfg.get("default_lifecycle_stage", "Sample Peak"),
                "bounding_box": build_bbox(lat, lon, pad_deg),
                "era5_filename": f"{sys_id}.nc",
                "raw_sha256": None,
                "artifact_sha256": None,
                "status": status_block,
                "notes": c_cfg["hypothesis"]
            })
            seen_sids.add(sid)
            count += 1

    with open("temp_tc_catalog.json", "w", encoding="utf-8") as f:
        json.dump(systems, f, indent=2)
    
    deferred_count = sum(1 for s in systems if s["status"]["harvest_state"] == "deferred")
    print(f"[TC BUILDER] Saved {len(systems)} records ({len(systems) - deferred_count} pending, {deferred_count} deferred).")

if __name__ == "__main__":
    process_tc_samples()