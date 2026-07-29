"""
TRACEBIND Phase 8 - Stage C1 Diagnostic & Invariance Suite
============================================================
Evaluates Hypothesis H1 (Translation Sensitivity), Rotational Invariance,
and Mask Interaction without modifying the frozen Phase 7 implementation.
"""

import sys
import json
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
from pyproj import Geod

PILOT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PILOT_DIR.parents[1]
sys.path.append(str(REPO_ROOT))

from phase7.sandbox.metrics.coherence import compute_phase_coherence

geod = Geod(ellps="WGS84")

def generate_lamb_oseen_vortex(nx=121, ny=121, dx=2500.0, r0=50000.0, gamma=1e5):
    """Generates an idealized synthetic Lamb-Oseen vortex centered on grid."""
    x = (np.arange(nx) - (nx - 1) / 2.0) * dx
    y = (np.arange(ny) - (ny - 1) / 2.0) * dx
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx**2 + yy**2) + 1e-5
    
    v_theta = (gamma / (2 * np.pi * r)) * (1.0 - np.exp(-(r / r0)**2))
    u = -v_theta * (yy / r)
    v = v_theta * (xx / r)
    return u, v

def run_invariance_tests():
    """Characterizes translation sensitivity and rotational invariance of Phase 7 operator."""
    u_orig, v_orig = generate_lamb_oseen_vortex(nx=121, ny=121)
    
    # 1. Baseline Centered (eta = 0.0)
    c_phi_centered = compute_phase_coherence(u_orig, v_orig)
    
    # 2. Rotational Invariance (Rotate wind vectors by 90 degrees)
    # Rotating (u, v) -> (-v, u)
    c_phi_rot90 = compute_phase_coherence(-v_orig, u_orig)
    rot_diff = float(abs(c_phi_centered - c_phi_rot90))
    
    # 3. Translation Sensitivity across varying offsets eta
    # Domain radius = 60 grid cells
    ny, nx = u_orig.shape
    domain_radius_cells = (nx - 1) / 2.0
    
    translation_series = []
    offsets_cells = [0, 5, 10, 15, 20, 25, 30]
    
    for shift in offsets_cells:
        u_shifted = np.roll(u_orig, shift=(shift, shift), axis=(0, 1))
        v_shifted = np.roll(v_orig, shift=(shift, shift), axis=(0, 1))
        
        # Calculate offset magnitude in cells
        offset_pixels = float(np.sqrt(shift**2 + shift**2))
        eta = float(offset_pixels / domain_radius_cells)
        
        c_phi_val = float(compute_phase_coherence(u_shifted, v_shifted))
        translation_series.append({
            "shift_cells_xy": [shift, shift],
            "pixel_offset": round(offset_pixels, 2),
            "eta_normalized_offset": round(eta, 4),
            "c_phi": c_phi_val
        })
        
    return {
        "rotational_invariance": {
            "c_phi_0_deg": float(c_phi_centered),
            "c_phi_90_deg": float(c_phi_rot90),
            "absolute_diff": rot_diff,
            "invariant": bool(rot_diff < 1e-12)
        },
        "translation_sensitivity_h1": translation_series
    }

def run_cohort_diagnostics(nc_path: Path, entry_config: dict, qc_info: dict):
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
    
    # Array Midpoint Geometry
    ny, nx = u_raw.shape
    array_mid_y, array_mid_x = (ny - 1) / 2.0, (nx - 1) / 2.0
    grid_center_lat, grid_center_lon = lats[int(array_mid_y)], lons[int(array_mid_x)]
    
    # Index of MSLP Center in Array
    mslp_idx_y = np.argmin(np.abs(lats - lat_center))
    mslp_idx_x = np.argmin(np.abs(lons - lon_center))
    
    # Offsets
    pixel_offset = float(np.sqrt((mslp_idx_x - array_mid_x)**2 + (mslp_idx_y - array_mid_y)**2))
    
    _, _, phys_offset_m = geod.inv(lon_center, lat_center, grid_center_lon, grid_center_lat)
    phys_offset_km = float(phys_offset_m / 1000.0)
    
    # Calculate domain radius in km (from array center to top-right corner)
    _, _, domain_radius_m = geod.inv(grid_center_lon, grid_center_lat, lons[-1], lats[-1])
    domain_radius_km = float(domain_radius_m / 1000.0)
    
    eta_normalized = float(phys_offset_km / domain_radius_km) if domain_radius_km > 0 else 0.0
    
    # Annular Shell Masking
    r_in = entry_config["r_inner_km"]
    r_out = entry_config["r_outer_km"]
    lon_mesh, lat_mesh = np.meshgrid(lons, lats)
    
    _, _, dist_m = geod.inv(
        np.full_like(lon_mesh, lon_center), np.full_like(lat_mesh, lat_center),
        lon_mesh, lat_mesh
    )
    dist_grid_km = dist_m / 1000.0
    eyewall_mask = (dist_grid_km >= r_in) & (dist_grid_km <= r_out)
    
    # Metric Extraction
    c_phi_full = float(compute_phase_coherence(u_raw, v_raw, mask=None))
    c_phi_shell = float(compute_phase_coherence(u_raw, v_raw, mask=eyewall_mask))
    
    # Visual Diagnostic Plot
    fig, ax = plt.subplots(figsize=(8, 7))
    wind_spd = np.sqrt(u_raw**2 + v_raw**2)
    im = ax.pcolormesh(lons, lats, wind_spd, cmap="viridis", shading="nearest")
    fig.colorbar(im, ax=ax, label="Wind Speed (m/s)")
    
    ax.contour(lons, lats, eyewall_mask.astype(int), levels=[0.5], colors="cyan", linewidths=2)
    ax.plot(lon_center, lat_center, "ro", markersize=8, label=f"MSLP Center ({lat_center:.2f}°, {lon_center:.2f}°)")
    ax.plot(grid_center_lon, grid_center_lat, "y^", markersize=8, label=f"Array Midpoint ({grid_center_lat:.2f}°, {grid_center_lon:.2f}°)")
    
    ax.set_title(f"{entry_config['name']}\nη = {eta_normalized:.3f} | Offset: {phys_offset_km:.1f} km ({pixel_offset:.1f} cells)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(loc="upper right")
    
    artifact_dir = PILOT_DIR / "artifacts" / "cphi_diagnostics"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    plot_path = artifact_dir / f"geometry_diag_{entry_config['id']}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    return {
        "id": entry_config["id"],
        "name": entry_config["name"],
        "c_phi_full_domain": c_phi_full,
        "c_phi_eyewall_shell": c_phi_shell,
        "phase7_geometry_assumptions": {
            "array_center_indices": [float(array_mid_y), float(array_mid_x)],
            "array_center_lat_lon": [float(grid_center_lat), float(grid_center_lon)],
            "detected_mslp_indices": [int(mslp_idx_y), int(mslp_idx_x)],
            "detected_mslp_lat_lon": [float(lat_center), float(lon_center)],
            "pixel_offset_cells": round(pixel_offset, 2),
            "physical_offset_km": round(phys_offset_km, 2),
            "domain_radius_km": round(domain_radius_km, 2),
            "eta_normalized_offset": round(eta_normalized, 4)
        },
        "diagnostic_plot": str(plot_path)
    }

if __name__ == "__main__":
    manifest_path = PILOT_DIR / "manifests" / "pilot_cohort.json"
    qc_report_path = PILOT_DIR / "reports" / "phase_c0_qc_report.json"
    
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    with open(qc_report_path, "r") as f:
        qc_data = json.load(f)
        
    print("==================================================")
    print(" STAGE C1 DIAGNOSTICS: PHASE 7 INVARIANCE SUITE")
    print("==================================================")
    
    # 1. Run Operator Invariance Tests
    invariance_res = run_invariance_tests()
    print("\n--- Rotational Invariance Test ---")
    rot_info = invariance_res["rotational_invariance"]
    print(f"  C_phi (0°)  : {rot_info['c_phi_0_deg']:.6f}")
    print(f"  C_phi (90°) : {rot_info['c_phi_90_deg']:.6f}")
    print(f"  Invariant   : {rot_info['invariant']} (Diff: {rot_info['absolute_diff']:.2e})")
    
    print("\n--- Translation Sensitivity (Hypothesis H1) ---")
    print("  Shift (XY cells) | Pixel Offset | Normalized Offset (η) | C_phi")
    print("  --------------------------------------------------------------")
    for step in invariance_res["translation_sensitivity_h1"]:
        print(f"  {str(step['shift_cells_xy']):<16} | {step['pixel_offset']:<12} | {step['eta_normalized_offset']:<21} | {step['c_phi']:.6f}")
        
    # 2. Run Cohort Observational Diagnostics
    cohort_diags = {}
    print("\n--- Cohort Geometry Diagnostics ---")
    for entry in manifest["cohort"]:
        entry_id = entry["id"]
        qc_info = qc_data.get(entry_id, {})
        
        if qc_info.get("qc_status") != "PASS":
            continue
            
        nc_file = REPO_ROOT / entry["file_path"]
        diag = run_cohort_diagnostics(nc_file, entry, qc_info)
        cohort_diags[entry_id] = diag
        
        geom = diag["phase7_geometry_assumptions"]
        print(f"\n[{entry_id}] {entry['name']}:")
        print(f"  - Array Center       : ({geom['array_center_lat_lon'][0]:.2f}°, {geom['array_center_lat_lon'][1]:.2f}°)")
        print(f"  - Detected Center     : ({geom['detected_mslp_lat_lon'][0]:.2f}°, {geom['detected_mslp_lat_lon'][1]:.2f}°)")
        print(f"  - Pixel Offset       : {geom['pixel_offset_cells']} cells")
        print(f"  - Physical Offset    : {geom['physical_offset_km']} km")
        print(f"  - Normalized Offsetη : {geom['eta_normalized_offset']}")
        print(f"  - C_phi (Full Domain): {diag['c_phi_full_domain']:.6f}")
        print(f"  - C_phi (Shell Mask) : {diag['c_phi_eyewall_shell']:.6f}")
        
    full_report = {
        "invariance_characterization": invariance_res,
        "cohort_diagnostics": cohort_diags
    }
    
    out_file = PILOT_DIR / "reports" / "stage_c1_diagnostics.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(full_report, f, indent=2)
    print(f"\n[✓] Diagnostic Report saved -> {out_file}")