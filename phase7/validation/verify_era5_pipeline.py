#!/usr/bin/env python3
"""
TRACEBIND Phase 7B - ERA5 Data Ingestion & Field Operator Verification
========================================================================
File: phase7/validation/verify_era5_pipeline.py
"""

import sys
from pathlib import Path
import numpy as np
import xarray as xr
import metpy.calc as mpcalc
from metpy.units import units
import matplotlib.pyplot as plt

VALIDATION_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = VALIDATION_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

# Extended Candidate Lists for Field Discovery
CANDIDATE_U = ["u", "u10", "ua", "UGRD", "eastward_wind", "u_component_of_wind", "u_wind"]
CANDIDATE_V = ["v", "v10", "va", "VGRD", "northward_wind", "v_component_of_wind", "v_wind"]
CANDIDATE_LAT = ["latitude", "lat", "LAT", "y"]
CANDIDATE_LON = ["longitude", "lon", "LON", "x"]


def resolve_key(ds: xr.Dataset, candidates: list) -> str:
    """Finds the matching key in ds.coords or ds.data_vars."""
    all_keys = list(ds.coords.keys()) + list(ds.data_vars.keys())
    for name in candidates:
        if name in all_keys:
            return name
    return None


def verify_era5_ingestion(nc_path: Path):
    print("=" * 65)
    print(" ERA5 DATA INGESTION & DERIVATIVE PIPELINE VERIFICATION")
    print("=" * 65)
    print(f" Target File: {nc_path}\n")

    if not nc_path.exists():
        raise FileNotFoundError(f"ERA5 benchmark file not found: {nc_path}")

    # 1. Open Dataset & Inspect Structure
    ds = xr.open_dataset(nc_path)
    print("--- DATASET STRUCTURE ---")
    print(ds)
    print("-------------------------\n")

    # 2. Variable & Coordinate Key Resolution
    lat_key = resolve_key(ds, CANDIDATE_LAT)
    lon_key = resolve_key(ds, CANDIDATE_LON)
    u_key = resolve_key(ds, CANDIDATE_U)
    v_key = resolve_key(ds, CANDIDATE_V)

    if not lat_key or not lon_key or not u_key or not v_key:
        print("\n[!] Key Resolution Failure:")
        print(f"    - Latitude Key  : {lat_key}")
        print(f"    - Longitude Key : {lon_key}")
        print(f"    - U Wind Key    : {u_key}")
        print(f"    - V Wind Key    : {v_key}")
        print("\nAvailable Data Variables:", list(ds.data_vars.keys()))
        print("Available Coordinates:   ", list(ds.coords.keys()))
        raise ValueError("Could not resolve essential wind or coordinate variables.")

    print(f"[✓] Key Resolution Succeeded:")
    print(f"    - Lat: '{lat_key}' | Lon: '{lon_key}' | U: '{u_key}' | V: '{v_key}'")

    # 3. Handle Pressure / Vertical Level Dimensions
    u_field = ds[u_key]
    v_field = ds[v_key]

    for level_dim in ["level", "pressure_level", "plev", "isobaricInhPa"]:
        if level_dim in u_field.dims:
            print(f"\n[*] Multi-level dimension detected ('{level_dim}'). Selecting first level...")
            u_field = u_field.isel({level_dim: 0})
            v_field = v_field.isel({level_dim: 0})

    # Select single time slice if present
    for time_dim in ["valid_time", "time"]:
        if time_dim in u_field.dims:
            u_field = u_field.isel({time_dim: 0})
            v_field = v_field.isel({time_dim: 0})

    # 4. Extract 1D Coordinates & Squeeze 2D Field Arrays
    lats = np.squeeze(ds[lat_key].values)
    lons = np.squeeze(ds[lon_key].values)

    if lats.ndim > 1:
        lats = lats[:, 0] if lats.ndim == 2 else lats[0, :, 0]
    if lons.ndim > 1:
        lons = lons[0, :] if lons.ndim == 2 else lons[0, 0, :]

    u_slice = np.squeeze(u_field.values)
    v_slice = np.squeeze(v_field.values)

    # Orientation Check (Latitude increasing along axis 0)
    is_lat_descending = lats[0] > lats[-1]
    print(f"    - Latitude Range  : [{lats.min():.2f}°, {lats.max():.2f}°] (Descending: {is_lat_descending})")
    print(f"    - Longitude Range : [{lons.min():.2f}°, {lons.max():.2f}°]")

    ny, nx = len(lats), len(lons)
    assert u_slice.shape == (ny, nx), f"Squeezed u shape {u_slice.shape} does not match grid ({ny}, {nx})"
    assert v_slice.shape == (ny, nx), f"Squeezed v shape {v_slice.shape} does not match grid ({ny}, {nx})"

    if is_lat_descending:
        lats = lats[::-1]
        u_slice = np.flipud(u_slice)
        v_slice = np.flipud(v_slice)

    # 5. Grid Spacing Calculation (Spherical to Cartesian dx/dy)
    dlat = np.abs(np.mean(np.diff(lats)))
    dlon = np.abs(np.mean(np.diff(lons)))
    
    mean_lat = np.mean(lats)
    R_earth = 6371000.0  # meters
    dy_m = dlat * (np.pi / 180.0) * R_earth
    dx_m = dlon * (np.pi / 180.0) * R_earth * np.cos(np.deg2rad(mean_lat))

    print(f"\n[2] Metric Grid Spacing (at mean lat {mean_lat:.2f}°):")
    print(f"    - dx : {dx_m:.2f} m")
    print(f"    - dy : {dy_m:.2f} m")

    # 6. Compute Vorticity: Internal vs MetPy
    du_dy = np.gradient(u_slice, dy_m, axis=0)
    dv_dx = np.gradient(v_slice, dx_m, axis=1)
    vort_internal = dv_dx - du_dy

    u_metpy = u_slice * units("m/s")
    v_metpy = v_slice * units("m/s")
    vort_metpy = mpcalc.vorticity(u_metpy, v_metpy, dx=dx_m * units.meter, dy=dy_m * units.meter).magnitude

    # 7. Interior Comparison (Halo Masking)
    halo = 5
    interior_mask = np.zeros_like(vort_internal, dtype=bool)
    interior_mask[halo:-halo, halo:-halo] = True

    diff = (vort_internal - vort_metpy)[interior_mask]
    vort_ref = vort_metpy[interior_mask]

    vort_scale = float(np.max(np.abs(vort_ref)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    rel_rmse = rmse / vort_scale if vort_scale > 0 else rmse
    corr_coef = float(np.corrcoef(vort_internal[interior_mask], vort_metpy[interior_mask])[0, 1])

    print("\n[3] In-Situ ERA5 Operator Validation:")
    print(f"    - Peak Relative Vorticity : {vort_scale:.6e} s^-1")
    print(f"    - Relative RMSE           : {rel_rmse:.3e}")
    print(f"    - Pearson Correlation (r) : {corr_coef:.12f}")

    # 8. Visual Diagnostics Generation
    wind_speed = np.sqrt(u_slice**2 + v_slice**2)
    max_idx = np.unravel_index(np.argmax(vort_internal), vort_internal.shape)
    vort_center_lat, vort_center_lon = lats[max_idx[0]], lons[max_idx[1]]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    im0 = axes[0].pcolormesh(lons, lats, wind_speed, cmap="Blues", shading="nearest")
    fig.colorbar(im0, ax=axes[0], label="m/s")
    axes[0].set_title("10m Wind Speed Field")

    im1 = axes[1].pcolormesh(lons, lats, vort_internal, cmap="PuOr", shading="nearest")
    axes[1].plot(vort_center_lon, vort_center_lat, "rx", markersize=12, markeredgewidth=2, label="Vorticity Peak")
    fig.colorbar(im1, ax=axes[1], label="s^-1")
    axes[1].set_title("Internal Relative Vorticity (ζ)")
    axes[1].legend(loc="upper right")

    diff_full = vort_internal - vort_metpy
    im2 = axes[2].pcolormesh(lons, lats, diff_full, cmap="coolwarm", shading="nearest")
    fig.colorbar(im2, ax=axes[2], label="s^-1")
    axes[2].set_title("Difference Field (Internal - MetPy)")

    for ax in axes:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    plt.tight_layout()
    plot_path = ARTIFACTS_DIR / f"era5_ingestion_{nc_path.stem}.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"\n[4] Diagnostic Output Saved -> {plot_path}")

    # Gate Verification
    assert rel_rmse < 1e-4, f"ERA5 Relative RMSE {rel_rmse:.3e} exceeds tolerance 1e-4"
    assert corr_coef > 0.9999, f"ERA5 Correlation {corr_coef:.6f} below threshold 0.9999"
    print("\n[✓] PASS: ERA5 pipeline ingested and verified successfully.")


if __name__ == "__main__":
    REPO_ROOT = Path(__file__).resolve().parents[2]
    
    if len(sys.argv) > 1:
        raw_path = Path(sys.argv[1])
        target_nc = raw_path if raw_path.is_absolute() else (REPO_ROOT / raw_path)
    else:
        target_nc = REPO_ROOT / "phase5" / "data" / "era5_mocha_72h.nc"

    verify_era5_ingestion(target_nc)