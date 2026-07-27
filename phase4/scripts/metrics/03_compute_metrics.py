"""
03_compute_metrics.py
---------------------
Production execution script for ERA5 Stage B Case Study (Cyclone Amphan).

Key Features:
  - Variable lookup abstraction (z/gh/geopotential, msl/mslp)
  - Dynamic degree-based spatial cropping (independent of grid resolution)
  - Strict boundary checking & spatial dimension consistency checks
  - Physical grid spacing integration for spatial gradients (dy, dx in meters)
  - Automatic frame-by-frame quick-look QA plot generation
  - Complete execution metadata exporting (metadata.json)
"""

import json
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timezone
from scipy.stats import entropy
from scipy.fft import fft2, fftshift

# Import frozen metrics from validated core module
from tracebind.core import compute_tb_v1, compute_tb_v2, compute_moran_i, compute_geary_c

# --- 1. Robust Variable & Coordinate Resolvers ---

def find_variable(ds: xr.Dataset, candidates: list[str]) -> str:
    """Finds first matching variable name present in dataset."""
    for cand in candidates:
        if cand in ds.data_vars or cand in ds.coords:
            return cand
    raise KeyError(f"None of candidate variables {candidates} found in dataset. Available: {list(ds.data_vars)}")

def get_grid_spacing_meters(lat_arr: np.ndarray, lon_arr: np.ndarray) -> tuple[float, float]:
    """Computes mean grid spacing in meters for latitude and longitude."""
    dlat = np.abs(np.mean(np.diff(lat_arr)))
    dlon = np.abs(np.mean(np.diff(lon_arr)))
    mean_lat = np.mean(lat_arr)
    
    # 1 degree lat ~ 111,000 m; lon scaled by cosine of latitude
    dy = dlat * 111_000.0
    dx = dlon * 111_000.0 * np.cos(np.radians(mean_lat))
    return float(dy), float(dx)

# --- 2. Advanced Mathematical Descriptors ---

def compute_grid_aware_gradient_energy(field: np.ndarray, dy: float, dx: float) -> float:
    """Computes mean square physical spatial gradient magnitude (m^2 / m^2)."""
    gy, gx = np.gradient(field, dy, dx)
    return float(np.mean(gx**2 + gy**2))

def compute_freedman_diaconis_entropy(field: np.ndarray) -> float:
    """Computes Shannon entropy using Freedman-Diaconis rule for bin width."""
    data = field.ravel()
    iqr = np.percentile(data, 75) - np.percentile(data, 25)
    n = len(data)
    
    if iqr > 0:
        bin_width = 2 * iqr * (n ** (-1/3))
        bins = max(10, int(np.ceil((np.max(data) - np.min(data)) / bin_width)))
    else:
        bins = 50  # Fallback for zero variance or uniform fields
        
    hist, _ = np.histogram(data, bins=bins, density=True)
    hist = hist[hist > 0]
    return float(entropy(hist))

def compute_experimental_spectral_slope(field: np.ndarray) -> float:
    """
    Experimental Descriptor: Computes 2D radial power spectral density slope.
    Labeled as experimental due to unmasked boundary/Nyquist edge effects.
    """
    ny, nx = field.shape
    # Demean and apply Hann window to reduce edge leakage
    win_y = np.hanning(ny)
    win_x = np.hanning(nx)
    window = np.outer(win_y, win_x)
    
    f_transform = fftshift(fft2((field - np.mean(field)) * window))
    psd = np.abs(f_transform)**2
    
    y, x = np.indices((ny, nx))
    center = (ny // 2, nx // 2)
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(int)
    
    tbin = np.bincount(r.ravel(), psd.ravel())
    nr = np.bincount(r.ravel())
    radial_profile = tbin / np.maximum(nr, 1)
    
    # Mid-frequency fitting mask (10% to 50% of max radius)
    max_r = len(radial_profile) // 2
    r_min, r_max = max(1, int(0.10 * max_r)), int(0.50 * max_r)
    
    k = np.arange(r_min, r_max)
    power = radial_profile[r_min:r_max]
    
    valid = (power > 0) & (k > 0)
    if np.sum(valid) < 3:
        return np.nan
    slope, _ = np.polyfit(np.log(k[valid]), np.log(power[valid]), 1)
    return float(slope)

# --- 3. Quick-Look Visual QA Plotter ---

def save_quicklook_frame(
    full_field: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    center_lat: float,
    center_lon: float,
    half_deg: float,
    frame_idx: int,
    time_str: str,
    output_dir: Path
):
    """Generates and saves a spatial QA diagnostic plot for crop validation."""
    fig, ax = plt.subplots(figsize=(7, 6))
    
    mesh = ax.pcolormesh(lons, lats, full_field, cmap='viridis', shading='auto')
    plt.colorbar(mesh, ax=ax, label="Geopotential Height (m)")
    
    # Plot storm center
    ax.plot(center_lon, center_lat, 'rx', markersize=12, markeredgewidth=2.5, label="Storm Center (Min MSLP)")
    
    # Plot crop bounding box
    bbox_lon = [center_lon - half_deg, center_lon + half_deg, center_lon + half_deg, center_lon - half_deg, center_lon - half_deg]
    bbox_lat = [center_lat - half_deg, center_lat - half_deg, center_lat + half_deg, center_lat + half_deg, center_lat - half_deg]
    ax.plot(bbox_lon, bbox_lat, 'r--', linewidth=1.5, label="20° Target Crop Window")
    
    ax.set_title(f"QA Quick-Look Frame {frame_idx:03d} | {time_str}")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.legend(loc='upper right')
    
    out_path = output_dir / f"frame_{frame_idx:03d}.png"
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()

# --- 4. Main Processing Engine ---

def process_era5_case_study(nc_path: Path, output_dir: Path, window_deg: float = 20.0) -> pd.DataFrame:
    """Full execution engine for ERA5 case study with strict quality guardrails."""
    ds = xr.open_dataset(nc_path)
    
    # Resolve Variable Names
    z_var = find_variable(ds, ['z', 'gh', 'geopotential'])
    msl_var = find_variable(ds, ['msl', 'mslp', 'mean_sea_level_pressure'])
    lat_var = find_variable(ds, ['latitude', 'lat'])
    lon_var = find_variable(ds, ['longitude', 'lon'])
    time_var = find_variable(ds, ['time', 'valid_time'])
    
    lats = ds[lat_var].values
    lons = ds[lon_var].values
    times = pd.to_datetime(ds[time_var].values)
    
    # Extract arrays
    z_data = ds[z_var].values
    msl_data = ds[msl_var].values
    
    # Convert units if required
    if np.nanmean(z_data) > 10000:  # Raw geopotential in m^2/s^2 -> height in meters
        z_data = z_data / 9.80665
    if np.nanmean(msl_data) > 2000: # Pa -> hPa
        msl_data = msl_data / 100.0
        
    dlat = np.abs(np.mean(np.diff(lats)))
    dlon = np.abs(np.mean(np.diff(lons)))
    dy_m, dx_m = get_grid_spacing_meters(lats, lons)
    
    half_deg = window_deg / 2.0
    half_cells_y = int(round(half_deg / dlat))
    half_cells_x = int(round(half_deg / dlon))
    expected_shape = (2 * half_cells_y, 2 * half_cells_x)
    
    frames_dir = output_dir / "quicklook_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    records = []
    print(f"\n{'='*80}\nSTARTING ERA5 CASE STUDY PROCESSING: {nc_path.name}")
    print(f"Grid Res: {dlat:.3f}° x {dlon:.3f}° | Crop Box: {expected_shape[0]}x{expected_shape[1]} cells (~{window_deg}°x{window_deg}°)")
    print(f"{'='*80}\n")
    
    for i, t in enumerate(times):
        frame_z = z_data[i]
        frame_p = msl_data[i]
        time_str = str(t)
        
        # NaN Guard on Full Field
        if np.isnan(frame_z).any() or np.isnan(frame_p).any():
            warnings.warn(f"Frame {i} at {time_str} contains NaNs. Skipping.")
            continue
            
        # Locate Storm Core (Minimum MSLP)
        min_idx = np.unravel_index(np.argmin(frame_p), frame_p.shape)
        cy, cx = min_idx
        center_lat = float(lats[cy])
        center_lon = float(lons[cx])
        min_mslp_val = float(frame_p[cy, cx])
        
        # Calculate Strict Index Bounds
        y_min, y_max = cy - half_cells_y, cy + half_cells_y
        x_min, x_max = cx - half_cells_x, cx + half_cells_x
        
        # Strict Boundary Guard: Skip frame if crop hits grid edge
        if y_min < 0 or y_max > frame_z.shape[0] or x_min < 0 or x_max > frame_z.shape[1]:
            warnings.warn(f"Frame {i} at {time_str}: Storm center ({center_lat:.2f}, {center_lon:.2f}) too close to domain boundary. Skipping frame.")
            continue
            
        sub_z = frame_z[y_min:y_max, x_min:x_max]
        
        # Validate Dimension Consistency
        if sub_z.shape != expected_shape:
            warnings.warn(f"Frame {i} crop shape {sub_z.shape} != expected {expected_shape}. Skipping frame to avoid shape artifact.")
            continue
            
        # Save Diagnostic QA Plot
        save_quicklook_frame(frame_z, lats, lons, center_lat, center_lon, half_deg, i, time_str, frames_dir)
        
        # Compute Metrics
        tb1 = compute_tb_v1(sub_z)
        tb2 = compute_tb_v2(sub_z)
        moran = compute_moran_i(sub_z)
        geary = compute_geary_c(sub_z)
        grad_e = compute_grid_aware_gradient_energy(sub_z, dy_m, dx_m)
        sp_ent = compute_freedman_diaconis_entropy(sub_z)
        spec_s = compute_experimental_spectral_slope(sub_z)
        
        rec = {
            "time": time_str,
            "center_lat": center_lat,
            "center_lon": center_lon,
            "min_mslp_hpa": min_mslp_val,
            "tb_v1": tb1,
            "tb_v2": tb2,
            "moran_i": moran,
            "geary_c": geary,
            "gradient_energy": grad_e,
            "spatial_entropy": sp_ent,
            "experimental_spectral_slope": spec_s
        }
        records.append(rec)
        print(f"[{i+1:03d}/{len(times):03d}] {time_str} | Lat: {center_lat:5.2f}°N Lon: {center_lon:5.2f}°E | MSLP: {min_mslp_val:6.1f} hPa | TB-v1: {tb1:.4f} | TB-v2: {tb2:.4f} | Moran: {moran:.4f}")
        
    df = pd.DataFrame(records)
    csv_out = output_dir / "metrics.csv"
    df.to_csv(csv_out, index=False)
    
    # Save Metadata JSON Sidecar
    meta = {
        "storm_name": "Cyclone Amphan",
        "dataset": "ERA5 Reanalysis",
        "analyzed_variable": "500 hPa Geopotential Height",
        "window_size_degrees": window_deg,
        "grid_resolution_degrees_lat": dlat,
        "grid_resolution_degrees_lon": dlon,
        "crop_cell_dimensions": list(expected_shape),
        "spatial_boundary_handling": "Strict Drop / Boundary Guard",
        "metric_definitions_state": "FROZEN",
        "total_frames_processed": len(df),
        "execution_timestamp_utc": datetime.now(timezone.utc).isoformat()
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
        
    print(f"\n[COMPLETE] Successfully generated:")
    print(f"  ├─ Data Table: {csv_out}")
    print(f"  ├─ Provenance: {output_dir / 'metadata.json'}")
    print(f"  └─ QA Frames : {frames_dir}\n")
    return df

if __name__ == "__main__":
    data_file = Path("./data/era5_amphan_72h.nc")
    output_directory = Path("./output/amphan_case_study")
    output_directory.mkdir(parents=True, exist_ok=True)
    
    process_era5_case_study(data_file, output_directory, window_deg=20.0)