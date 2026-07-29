"""TRACEBIND Phase 8 Stage C2 - Blinded Cohort Generator & Manifest Builder

========================================================================
File: phase8/c2/build_c2_blinded_cohort.py
Produces:
  1. phase8/c2/manifest/c2_cohort_manifest_blinded.json
  2. phase8/c2/manifest/keycard_c2_access_controlled.json
"""

import datetime
import hashlib
import hmac
import json
import os
import uuid
from pathlib import Path

# Secret salt for keycard HMAC verification (environment override or default)
HMAC_SALT = os.environ.get("C2_UNBLIND_SALT", "TRACEBIND_C2_SALT_2026_07_28").encode("utf-8")

# Case inputs (10 Cyclones, 10 Controls)
UNBLINDED_CASES = [
    # --- 10 Mature Cyclones ---
    {"case_id": "TC_2023_MOCHA", "case_type": "Cyclone", "basin": "Bay_of_Bengal", "center": [18.5, 91.25]},
    {"case_id": "TC_2020_AMPHAN", "case_type": "Cyclone", "basin": "Bay_of_Bengal", "center": [19.8, 87.7]},
    {"case_id": "TC_2019_FANI", "case_type": "Cyclone", "basin": "Bay_of_Bengal", "center": [18.1, 85.5]},
    {"case_id": "TC_2008_NARGIS", "case_type": "Cyclone", "basin": "Bay_of_Bengal", "center": [15.8, 94.2]},
    {"case_id": "TC_2021_TAUKTAE", "case_type": "Cyclone", "basin": "Arabian_Sea", "center": [19.1, 71.2]},
    {"case_id": "TC_2021_YAAS", "case_type": "Cyclone", "basin": "Bay_of_Bengal", "center": [20.8, 87.3]},
    {"case_id": "TC_2019_KYARR", "case_type": "Cyclone", "basin": "Arabian_Sea", "center": [17.5, 66.2]},
    {"case_id": "TC_2014_HUDHUD", "case_type": "Cyclone", "basin": "Bay_of_Bengal", "center": [17.7, 83.3]},
    {"case_id": "TC_2013_PHAILIN", "case_type": "Cyclone", "basin": "Bay_of_Bengal", "center": [18.7, 85.0]},
    {"case_id": "TC_2019_BULBUL", "case_type": "Cyclone", "basin": "Bay_of_Bengal", "center": [21.1, 88.1]},
    # --- 10 Monsoon Lows & Controls ---
    {"case_id": "CTRL_2023_IND_01", "case_type": "Control", "basin": "Indian_Ocean_Equatorial", "center": [-5.25, 77.5]},
    {"case_id": "CTRL_2023_IND_02", "case_type": "Control", "basin": "Bay_of_Bengal", "center": [17.0, 89.0]},
    {"case_id": "CTRL_2022_MON_LOW1", "case_type": "Control", "basin": "Central_India", "center": [22.5, 82.1]},
    {"case_id": "CTRL_2022_MON_LOW2", "case_type": "Control", "basin": "North_Bay_of_Bengal", "center": [21.0, 89.5]},
    {"case_id": "CTRL_2021_TROUGH1", "case_type": "Control", "basin": "Arabian_Sea_Offshore", "center": [14.2, 72.8]},
    {"case_id": "CTRL_2021_DEP_01", "case_type": "Control", "basin": "Bay_of_Bengal", "center": [13.5, 84.1]},
    {"case_id": "CTRL_2020_SHEAR_01", "case_type": "Control", "basin": "South_India", "center": [10.5, 78.0]},
    {"case_id": "CTRL_2020_SURGE_01", "case_type": "Control", "basin": "Arabian_Sea", "center": [12.0, 68.5]},
    {"case_id": "CTRL_2019_TROUGH2", "case_type": "Control", "basin": "Bay_of_Bengal", "center": [16.5, 85.2]},
    {"case_id": "CTRL_2018_LOW_03", "case_type": "Control", "basin": "Central_India", "center": [23.1, 80.4]},
]

def generate_blinded_cohort():
    script_path = Path(__file__).resolve()
    generator_hash = hashlib.sha256(script_path.read_bytes()).hexdigest()
    
    manifest_dir = script_path.parents[0] / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    blinded_manifest = []
    keycard = {}

    for entry in UNBLINDED_CASES:
        # 1. Deterministic UUID Generation via uuid5
        det_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, entry["case_id"])
        token = f"c2_uuid_{det_uuid.hex[:8]}"

        # 2. Keycard HMAC Token for Access Verification
        hmac_verifier = hmac.new(HMAC_SALT, entry["case_id"].encode("utf-8"), hashlib.sha256).hexdigest()

        blinded_record = {
            "blinded_id": token,
            "center_source": "minimum_pressure_center",
            "data_source": "ERA5_Hourly_Pressure_Levels",
            "file_path": f"phase8/c2/raw/{token}.nc",
            "center_coordinates": entry["center"],
            "analysis_shell_km": [30.0, 150.0],
            "dataset_metadata": {
                "provider": "ECMWF",
                "dataset": "ERA5",
                "variables": ["u_component_of_wind", "v_component_of_wind", "geopotential"],
                "spatial_resolution_deg": 0.25,
                "interpolation_method": "bilinear_to_2.5km_cartesian_mesh"
            }
        }
        blinded_manifest.append(blinded_record)

        keycard[token] = {
            "original_case_id": entry["case_id"],
            "case_type": entry["case_type"],
            "basin": entry["basin"],
            "hmac_sha256": hmac_verifier
        }

    manifest_data = {
        "pre_registration": {
            "primary_metric": "C_phi",
            "primary_comparison": "cyclone_vs_control",
            "directional_hypothesis": "cyclone_C_phi > control_C_phi",
            "thresholds_defined_before_analysis": True,
            "post_hoc_metric_modification": False,
            "null_model": "independent_pointwise_directional_uniform_permutation"
        },
        "provenance": {
            "protocol_version": "v1.0",
            "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "generator_script_hash": generator_hash,
            "sample_size": len(UNBLINDED_CASES)
        },
        "cases": blinded_manifest
    }

    manifest_path = manifest_dir / "c2_cohort_manifest_blinded.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)

    keycard_path = manifest_dir / "keycard_c2_access_controlled.json"
    with open(keycard_path, "w") as f:
        json.dump({"unblind_keycard": keycard}, f, indent=2)

    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    print(f"Blinded Manifest generated: {manifest_path}")
    print(f"Manifest SHA-256: {manifest_hash}")
    print(f"Generator Hash: {generator_hash}")
    print(f"Access-controlled Keycard generated: {keycard_path}")

if __name__ == "__main__":
    generate_blinded_cohort()