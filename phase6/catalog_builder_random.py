"""
Phase 6A.5: Random Background Controls Builder (With Explicit Provenance)
"""

import json
import yaml

def normalize_longitude(lon: float) -> float:
    return round(((lon + 180.0) % 360.0) - 180.0, 4)

def process_random_samples():
    with open("selection_rules.yaml", "r", encoding="utf-8") as f:
        rules = yaml.safe_load(f)

    c_cfg = rules["cohort_definitions"]["COHORT_G"]
    cohort_id = c_cfg["cohort_id"]
    pad_deg = rules.get("bounding_box_padding_deg", 15.0)

    controls = [
        {"id": "CTRL_2023_PAC_01", "name": "Quiet Pacific Subtropics", "lat": 15.0, "lon": -140.0, "basin": "EP"},
        {"id": "CTRL_2023_ATL_01", "name": "Quiet Atlantic Subtropics", "lat": 25.0, "lon": -45.0, "basin": "NA"},
        {"id": "CTRL_2023_IND_01", "name": "Quiet Indian Subtropics", "lat": -20.0, "lon": 80.0, "basin": "SI"},
    ]

    systems = []
    for c in controls:
        lat = c["lat"]
        lon = normalize_longitude(c["lon"])
        bbox = [
            round(max(-90.0, lat - pad_deg), 2),
            round(min(90.0, lat + pad_deg), 2),
            normalize_longitude(lon - pad_deg),
            normalize_longitude(lon + pad_deg)
        ]

        systems.append({
            "system_id": c["id"],
            "ibtracs_sid": None,
            "cohort_id": cohort_id,
            "cohort_name": c_cfg["name"],
            "system_name": c["name"],
            "event_year": 2023,
            "system_class": c_cfg["selection_constraints"]["system_class"],
            "subclass": c_cfg["selection_constraints"]["subclass"],
            "basin": c["basin"],
            "source_dataset": "ERA5",
            "source_reference": "Synthetic_Control_Grid",
            "analysis_time": "2023-07-15T12:00:00Z",
            "center_lat": round(lat, 4),
            "center_lon": lon,
            "min_pressure_hpa": 1013.2,
            "max_wind_kt": 5.0,
            "lifecycle_stage": c_cfg.get("default_lifecycle_stage", "Non-Coherent Baseline"),
            "bounding_box": bbox,
            "era5_filename": f"{c['id']}.nc",
            "sha256": None,
            "status": {"downloaded": False, "qc_passed": False, "processed": False},
            "control_provenance": c_cfg.get("control_generation", {}),
            "notes": c_cfg["hypothesis"]
        })

    with open("temp_random_catalog.json", "w", encoding="utf-8") as f:
        json.dump(systems, f, indent=2)
    print(f"[RANDOM BUILDER] Saved {len(systems)} control records with generation metadata.")

if __name__ == "__main__":
    process_random_samples()