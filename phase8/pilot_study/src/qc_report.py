"""
TRACEBIND Phase 8 - Phase C0 Quality Control & Local Center Stability Auditor
=============================================================================
Audits:
1. Grid spacing & missing values.
2. Local Search Masking: Enforces local vorticity maximum search within max_center_sep_km.
3. Center Stability: Measures separation between MSLP min and local max vorticity.
4. Generates eyewall_mask.png QC visual artifact.
"""

import sys
import json
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pyproj import Geod

PILOT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PILOT_DIR.parents[1]
sys.path.append(str(REPO_ROOT))

# Core Frozen Operator
from phase7.sandbox.metrics.coherence import compute_relative_vorticity

geod = Geod(ellps="WGS84")

def audit_and_verify_center(nc_path: Path, entry_config: dict):
    ds = xr.open_dataset(nc_path)
    
    lat_key = "latitude" if "latitude" in ds else "lat"
    lon_key = "longitude" if "longitude" in ds else "lon"
    u_key = "u10" if "u10" in ds else "u"
    v_key = "v10" if "v10" in ds else "v"
    msl_key = "msl" if "msl" in ds else "mslp"
    
    lats = np.squeeze(ds[lat_key].values)
    lons = np.squeeze(ds[lon_key].values)
    
    lat_descending = bool(lats[0] > lats[-1])
    if lat_descending:
        lats = lats[::-1]
        u_raw = np.flipud(np.squeeze(ds[u_key].values))
        v_raw = np.flipud(np.squeeze(ds[v_key].values))
        msl_raw = np.flipud(np.squeeze(ds[msl_key].values))
    else:
        u_raw = np.squeeze(ds[u_key].values)
        v_raw = np.squeeze(ds[v_key].values)
        msl_raw = np.squeeze(ds[msl_key].values)
        
    # Geodesic spacing
    mid_lon = float(lons[len(lons) // 2])
    mid_lat = float(lats[len(lats) // 2])
    
    _, _, dy_array = geod.inv(
        np.full(len(lats) - 1, mid_lon), lats[:-1],
        np.full(len(lats) - 1, mid_lon), lats[1:]
    )
    dy_m = float(np.mean(np.abs(dy_array)))
    
    _, _, dx_array = geod.inv(
        lons[:-1], np.full(len(lons) - 1, mid_lat),
        lons[1:], np.full(len(lons) - 1, mid_lat)
    )
    dx_m = float(np.mean(np.abs(dx_array)))
    
    # 1. Vorticity Calculation
    vort_2d = compute_relative_vorticity(u_raw, v_raw, dx_m, dy_m)
    
    # 2. Locate MSLP Minimum Center
    mslp_min_idx = np.unravel_index(np.argmin(msl_raw), msl_raw.shape)
    lat_mslp, lon_mslp = float(lats[mslp_min_idx[0]]), float(lons[mslp_min_idx[1]])
    
    # 3. Local Search Radius Masking
    search_radius_km = float(entry_config.get("max_center_sep_km", 75.0))
    lon_mesh, lat_mesh = np.meshgrid(lons, lats)
    
    _, _, dist_from_mslp_m = geod.inv(
        np.full_like(lon_mesh, lon_mslp), np.full_like(lat_mesh, lat_mslp),
        lon_mesh, lat_mesh
    )
    dist_from_mslp_km = dist_from_mslp_m / 1000.0
    
    local_search_mask = dist_from_mslp_km <= search_radius_km
    candidate_cells = int(np.sum(local_search_mask))
    
    if candidate_cells == 0:
        # Fallback safeguard if search bounds are narrower than grid resolution
        vort_center_lat, vort_center_lon = lat_mslp, lon_mslp
        local_max_vort = float(vort_2d[mslp_min_idx])
        center_sep_km = 0.0
        mask_error = True
    else:
        mask_error = False
        sub_vort = np.where(local_search_mask, vort_2d, -np.inf)
        vort_max_idx = np.unravel_index(np.argmax(sub_vort), sub_vort.shape)
        vort_center_lat, vort_center_lon = float(lats[vort_max_idx[0]]), float(lons[vort_max_idx[1]])
        local_max_vort = float(vort_2d[vort_max_idx])
        
        _, _, center_sep_m = geod.inv(lon_mslp, lat_mslp, vort_center_lon, vort_center_lat)
        center_sep_km = float(center_sep_m / 1000.0)

    center_stable = (center_sep_km <= search_radius_km) and (not mask_error)
    
    # 4. Compute Shell Mask for Eyewall
    r_in = entry_config["r_inner_km"]
    r_out = entry_config["r_outer_km"]
    eyewall_mask = (dist_from_mslp_km >= r_in) & (dist_from_mslp_km <= r_out)
    
    # 5. Export QC Artifact Figure
    fig, ax = plt.subplots(figsize=(8, 7))
    wind_spd = np.sqrt(u_raw**2 + v_raw**2)
    im = ax.pcolormesh(lons, lats, wind_spd, cmap="viridis", shading="nearest")
    fig.colorbar(im, ax=ax, label="10m Wind Speed (m/s)")
    
    # Overlay Shell Mask and Centers
    ax.contour(lons, lats, eyewall_mask.astype(int), levels=[0.5], colors="red", linewidths=2)
    ax.plot(lon_mslp, lat_mslp, "ro", markersize=8, label="MSLP Min")
    ax.plot(vort_center_lon, vort_center_lat, "bx", markersize=10, markeredgewidth=2, label="Local Vort Max")
    
    ax.set_title(f"{entry_config['name']} QC - Center Sep: {center_sep_km:.1f} km")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="upper right")
    
    artifact_dir = PILOT_DIR / "artifacts" / "qc_masks"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    plot_path = artifact_dir / f"qc_mask_{entry_config['id']}.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    missing_count = int(sum(ds[var].isnull().sum().item() for var in ds.data_vars))
    qc_pass = (missing_count == 0) and center_stable and (candidate_cells > 0)
    
    return {
        "file": nc_path.name,
        "dx_m": round(dx_m, 2),
        "dy_m": round(dy_m, 2),
        "mslp_center": [lat_mslp, lon_mslp],
        "search_radius_km": search_radius_km,
        "candidate_cells": candidate_cells,
        "local_max_vorticity": local_max_vort,
        "vort_center": [vort_center_lat, vort_center_lon],
        "center_separation_km": round(center_sep_km, 2),
        "center_stability": "STABLE" if center_stable else "UNSTABLE",
        "missing_values": missing_count,
        "qc_artifact": str(plot_path),
        "qc_status": "PASS" if qc_pass else "FAIL"
    }

if __name__ == "__main__":
    manifest_path = PILOT_DIR / "manifests" / "pilot_cohort.json"
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
        
    print("==================================================")
    print(" PHASE C0: LOCAL CENTER STABILITY & QC AUDIT")
    print("==================================================")
    
    qc_results = {}
    for entry in manifest["cohort"]:
        nc_file = REPO_ROOT / entry["file_path"]
        report = audit_and_verify_center(nc_file, entry)
        qc_results[entry["id"]] = report
        
        print(f"\n[{entry['id']}] {entry['name']}:")
        print("  Center Detection Diagnostics")
        print("  ----------------------------")
        print(f"  MSLP Minimum Center  : ({report['mslp_center'][0]:.2f}°, {report['mslp_center'][1]:.2f}°)")
        print(f"  Local Search Radius  : {report['search_radius_km']} km")
        print(f"  Candidate Cells      : {report['candidate_cells']}")
        print(f"  Local Max Vorticity  : {report['local_max_vorticity']:.4e} s^-1")
        print(f"  Local Vort Center    : ({report['vort_center'][0]:.2f}°, {report['vort_center'][1]:.2f}°)")
        print(f"  Center Separation    : {report['center_separation_km']} km")
        print(f"  Center Stability     : [{report['center_stability']}]")
        print(f"  Missing Values       : {report['missing_values']}")
        print(f"  QC Mask Artifact     : {report['qc_artifact']}")
        print(f"  Final Gate Status    : [{report['qc_status']}]")
        
    out_qc = PILOT_DIR / "reports" / "phase_c0_qc_report.json"
    out_qc.parent.mkdir(parents=True, exist_ok=True)
    with open(out_qc, "w") as f:
        json.dump(qc_results, f, indent=2)
    print(f"\n[✓] QC Report saved -> {out_qc}")