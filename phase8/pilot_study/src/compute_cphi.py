"""
TRACEBIND Phase 8 - Stage C1 Metric Extractor
=============================================
Processes ONLY cohorts that achieved QC PASS in Phase C0 v1.0.
Records full SHA-256 provenance tracking for Phase 7 operators.
"""

import sys
import json
import hashlib
from pathlib import Path
import numpy as np
import xarray as xr
from pyproj import Geod

PILOT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PILOT_DIR.parents[1]
sys.path.append(str(REPO_ROOT))

# Direct Import of Frozen Source Operators
from phase7.sandbox.metrics.coherence import compute_relative_vorticity, compute_phase_coherence

geod = Geod(ellps="WGS84")

def get_file_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()

def compute_snapshot_cphi(nc_path: Path, entry_config: dict, qc_info: dict):
    ds = xr.open_dataset(nc_path)
    
    lat_key = "latitude" if "latitude" in ds else "lat"
    lon_key = "longitude" if "longitude" in ds else "lon"
    u_key = "u10" if "u10" in ds else "u"
    v_key = "v10" if "v10" in ds else "v"
    msl_key = "msl" if "msl" in ds else "mslp"
    
    lats = np.squeeze(ds[lat_key].values)
    lons = np.squeeze(ds[lon_key].values)
    
    if lats[0] > lats[-1]:
        lats = lats[::-1]
        u_raw = np.flipud(np.squeeze(ds[u_key].values))
        v_raw = np.flipud(np.squeeze(ds[v_key].values))
        msl_raw = np.flipud(np.squeeze(ds[msl_key].values))
    else:
        u_raw = np.squeeze(ds[u_key].values)
        v_raw = np.squeeze(ds[v_key].values)
        msl_raw = np.squeeze(ds[msl_key].values)
        
    lat_center, lon_center = qc_info["mslp_center"]
    
    # Dynamic Annular Masking around Verified MSLP Center
    r_in = entry_config["r_inner_km"]
    r_out = entry_config["r_outer_km"]
    
    lon_mesh, lat_mesh = np.meshgrid(lons, lats)
    _, _, dist_m = geod.inv(
        np.full_like(lon_mesh, lon_center), np.full_like(lat_mesh, lat_center),
        lon_mesh, lat_mesh
    )
    dist_grid_km = dist_m / 1000.0
    eyewall_mask = (dist_grid_km >= r_in) & (dist_grid_km <= r_out)
    
    # Execute Frozen Operator
    c_phi = compute_phase_coherence(u_raw, v_raw, mask=eyewall_mask)
    wind_spd = np.sqrt(u_raw**2 + v_raw**2)
    
    return {
        "v_max_kts": float(np.max(wind_spd) * 1.94384),
        "p_min_hpa": float(np.min(msl_raw) / 100.0) if np.min(msl_raw) > 50000 else float(np.min(msl_raw)),
        "c_phi_frozen": float(c_phi),
        "center_mslp": [lat_center, lon_center],
        "shell_mask_bounds_km": [r_in, r_out]
    }

if __name__ == "__main__":
    manifest_path = PILOT_DIR / "manifests" / "pilot_cohort.json"
    qc_report_path = PILOT_DIR / "reports" / "phase_c0_qc_report.json"
    coherence_module_path = REPO_ROOT / "phase7" / "sandbox" / "metrics" / "coherence.py"
    
    if not qc_report_path.exists():
        print("[!] Phase C0 QC Report missing. Run src/qc_report.py first.")
        sys.exit(1)
        
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    with open(qc_report_path, "r") as f:
        qc_data = json.load(f)
        
    # Provenance Hash Extraction
    coherence_sha256 = get_file_sha256(coherence_module_path)
    
    print("==================================================")
    print(" STAGE C1: GATED METRIC EXTRACTION")
    print("==================================================")
    print(f" Frozen Engine Hash (SHA-256) : {coherence_sha256}")
    print("==================================================")
    
    stage_c1_output = {
        "provenance": {
            "phase7_release": "sandbox-v1.0",
            "metric_function": "compute_phase_coherence",
            "metric_sha256": coherence_sha256,
            "qc_auditor_release": "phase_c0_v1.0"
        },
        "results": {}
    }
    
    for entry in manifest["cohort"]:
        entry_id = entry["id"]
        qc_info = qc_data.get(entry_id, {})
        
        if qc_info.get("qc_status") != "PASS":
            print(f"\n[EXCLUDED] [{entry_id}] -> {entry['name']} (Failed QC Gate)")
            continue
            
        nc_file = REPO_ROOT / entry["file_path"]
        metrics = compute_snapshot_cphi(nc_file, entry, qc_info)
        stage_c1_output["results"][entry_id] = metrics
        
        print(f"\n[EXTRACTED] [{entry_id}] -> {entry['name']}:")
        print(f"  - Center Location : ({metrics['center_mslp'][0]:.2f}°, {metrics['center_mslp'][1]:.2f}°)")
        print(f"  - V_max           : {metrics['v_max_kts']:.1f} kts")
        print(f"  - P_min           : {metrics['p_min_hpa']:.1f} hPa")
        print(f"  - C_phi (Frozen)  : {metrics['c_phi_frozen']:.6f}")
        
    out_report = PILOT_DIR / "reports" / "stage_c1_summary.json"
    out_report.parent.mkdir(parents=True, exist_ok=True)
    with open(out_report, "w") as f:
        json.dump(stage_c1_output, f, indent=2)
    print(f"\n[✓] Stage C1 Summary exported -> {out_report}")