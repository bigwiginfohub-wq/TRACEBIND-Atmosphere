"""
02_visualize_frames.py
----------------------
Generates frame-by-frame visual diagnostics (MSLP + Z500 overlay) 
and compiles an animated GIF across all 96 hourly ERA5 frames.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import imageio.v2 as imageio

def find_variable(ds: xr.Dataset, candidates: list[str]) -> str:
    for cand in candidates:
        if cand in ds.data_vars or cand in ds.coords:
            return cand
    raise KeyError(f"None of candidate variables {candidates} found in dataset. Available: {list(ds.data_vars)}")

def run_visual_qa(nc_path: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "qa_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    ds = xr.open_dataset(nc_path)

    # Robust coordinate and variable resolution
    time_var = find_variable(ds, ['valid_time', 'time'])
    z_var = find_variable(ds, ['z', 'gh', 'geopotential'])
    msl_var = find_variable(ds, ['msl', 'mslp', 'mean_sea_level_pressure'])
    lat_var = find_variable(ds, ['latitude', 'lat'])
    lon_var = find_variable(ds, ['longitude', 'lon'])

    # Handle pressure_level dimension explicitly
    z_ds = ds[z_var]
    if 'pressure_level' in z_ds.dims:
        z_ds = z_ds.isel(pressure_level=0)
    elif 'level' in z_ds.dims:
        z_ds = z_ds.isel(level=0)

    msl_ds = ds[msl_var]
    if 'pressure_level' in msl_ds.dims:
        msl_ds = msl_ds.isel(pressure_level=0)
    elif 'level' in msl_ds.dims:
        msl_ds = msl_ds.isel(level=0)

    times = pd.to_datetime(ds[time_var].values)
    lats = ds[lat_var].values
    lons = ds[lon_var].values

    # Units normalization
    z_data = z_ds.squeeze().values
    msl_data = msl_ds.squeeze().values

    if np.nanmean(z_data) > 10000:
        z_data = z_data / 9.80665  # m^2/s^2 -> geopotential meters
    if np.nanmean(msl_data) > 2000:
        msl_data = msl_data / 100.0  # Pa -> hPa

    inspect_indices = [0, 12, 24, 36, 48, 60, 72, 84, 95]
    inspect_indices = [idx for idx in inspect_indices if idx < len(times)]

    image_paths = []

    print(f"Generating frames for dataset ({len(times)} total timesteps)...")

    for i, t in enumerate(times):
        time_str = pd.Timestamp(t).strftime('%Y-%m-%d %H:%M UTC')
        frame_z = z_data[i]
        frame_p = msl_data[i]

        # Tracking center (min MSLP)
        min_idx = np.unravel_index(np.argmin(frame_p), frame_p.shape)
        cy, cx = min_idx
        center_lat, center_lon = lats[cy], lons[cx]
        min_p = frame_p[cy, cx]

        fig, ax = plt.subplots(figsize=(8, 7), dpi=120)

        # Base layer: Z500 Geopotential Height
        mesh = ax.pcolormesh(lons, lats, frame_z, cmap='YlGnBu_r', shading='auto')
        cbar = plt.colorbar(mesh, ax=ax, orientation='vertical', pad=0.02)
        cbar.set_label('Z500 Geopotential Height (gpm)', fontsize=10)

        # Overlay: MSLP Contours
        contours = ax.contour(lons, lats, frame_p, levels=15, colors='k', linewidths=0.8, alpha=0.7)
        ax.clabel(contours, inline=True, fontsize=8, fmt='%1.0f')

        # Highlight tracked core
        ax.plot(center_lon, center_lat, 'r*', markersize=14, markeredgecolor='black', label=f'Min MSLP Core ({min_p:.1f} hPa)')

        ax.set_title(f"Cyclone Amphan | Frame {i:02d} | {time_str}", fontsize=11, fontweight='bold')
        ax.set_xlabel('Longitude (°E)')
        ax.set_ylabel('Latitude (°N)')
        ax.legend(loc='upper right', frameon=True)

        frame_path = frames_dir / f"amphan_frame_{i:02d}.png"
        plt.savefig(frame_path, bbox_inches='tight')
        plt.close()

        image_paths.append(frame_path)

        if i in inspect_indices:
            print(f" Saved QA snapshot: Frame {i:02d} | {time_str} | Min MSLP: {min_p:.1f} hPa at ({center_lat:.2f}°N, {center_lon:.2f}°E)")

    # Build Animation GIF
    gif_path = output_dir / "amphan_evolution.gif"
    print(f"\nCompiling animation GIF to {gif_path}...")
    images = [imageio.imread(p) for p in image_paths]
    imageio.mimsave(gif_path, images, fps=5)
    print("Animation build complete.")

if __name__ == "__main__":
    nc_file = Path("era5_amphan_72h.nc")
    out_dir = Path("./qa_output")
    run_visual_qa(nc_file, out_dir)