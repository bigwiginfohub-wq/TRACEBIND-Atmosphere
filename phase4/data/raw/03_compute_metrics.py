"""
03_compute_metrics.py
---------------------
Core TRACEBIND spatial metric extraction engine for ERA5 cyclone case studies.
Computes frame-by-frame structural and kinematic metrics:
 - TB-v1
 - TB-v2 (Intensity Projection & Cosine Similarity)
 - Moran's I
 - Geary's C
 - Gradient Energy
 - Spectral Slope
 - Spatial Entropy
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

# --- Metric Definitions ---

def compute_tb_v1(field: np.ndarray) -> float:
    """TRACEBIND v1: Quadratic phase/gradient interaction score."""
    gy, gx = np.gradient(field)
    grad_mag = np.sqrt(gx**2 + gy**2)
    valid = grad_mag > 1e-10
    if not np.any(valid):
        return 0.0
    
    # Quadratic curvature proxy
    gyy, gyx = np.gradient(gy)
    gxy, gxx = np.gradient(gx)
    curvature = np.abs(gxx * gyy - gxy * gyx)
    
    tb1_val = np.mean(curvature[valid] / (grad_mag[valid] + 1e-6))
    return float(tb1_val)

def compute_tb_v2_variants(field: np.ndarray) -> tuple[float, float]:
    """
    Computes both TRACEBIND v2 variants:
     1. Unbounded Intensity Projection (tb_v2_intensity)
     2. Normalized Cosine Similarity (tb_v2_cosine)
    """
    gy, gx = np.gradient(field)
    grad_mag = np.sqrt(gx**2 + gy**2)
    
    valid_grad = grad_mag > 1e-10
    if not np.any(valid_grad):
        return 0.0, 0.0

    padded = np.pad(field, pad_width=1, mode='edge')
    dy_res = 0.5 * (padded[2:, 1:-1] - padded[:-2, 1:-1])
    dx_res = 0.5 * (padded[1:-1, 2:] - padded[1:-1, :-2])
    
    # 1. Unbounded Intensity Projection
    grad_mag_center = grad_mag[1:-1, 1:-1]
    v2_intensity = float(np.sum(coherence_map * grad_mag_center) / (np.sum(grad_mag_center) + 1e-8))

    # 2. Fully Normalized Cosine Similarity
    res_mag = np.sqrt(dx_res**2 + dy_res**2)
    valid_both = valid_grad & (res_mag > 1e-10)
    
    if not np.any(valid_both):
        tb_v2_cosine = 0.0
    else:
        coherence_cosine = (gx * dx_res + gy * dy_res) / (grad_mag * res_mag + 1e-12)
        tb_v2_cosine = float(np.mean(coherence_cosine[valid_both]))

    return tb_v2_intensity, tb_v2_cosine

def compute_tb_v2(field: np.ndarray) -> float:
    """Legacy alias returning the primary intensity variant for backwards compatibility."""
    intensity, _ = compute_tb_v2_variants(field)
    return intensity

def compute_morans_i(field: np.ndarray) -> float:
    """Computes global Moran's I spatial autocorrelation."""
    z = field - np.mean(field)
    s0 = z.shape[0] * z.shape[1]
    
    # Simple 4-neighbor spatial weights
    padded = np.pad(z, pad_width=1, mode='constant', constant_values=0)
    neighbors = padded[2:, 1:-1] + padded[:-2, 1:-1] + padded[1:-1, 2:] + padded[1:-1, :-2]
    
    numerator = np.sum(z * neighbors)
    denominator = np.sum(z**2) + 1e-12
    return float((s0 / (4.0 * s0)) * (numerator / denominator))

def compute_gearys_c(field: np.ndarray) -> float:
    """Computes Geary's C spatial association metric."""
    z = field
    n = z.size
    padded = np.pad(z, pad_width=1, mode='edge')
    
    diff_sq = (
        (z - padded[2:, 1:-1])**2 + 
        (z - padded[:-2, 1:-1])**2 + 
        (z - padded[1:-1, 2:])**2 + 
        (z - padded[1:-1, :-2])**2
    )
    
    s0 = 4.0 * n
    var_z = np.var(z) + 1e-12
    return float(((n - 1) * np.sum(diff_sq)) / (2.0 * s0 * n * var_z))

def compute_gradient_energy(field: np.ndarray) -> float:
    """Computes total mean spatial gradient energy."""
    gy, gx = np.gradient(field)
    return float(np.mean(gx**2 + gy**2))

def compute_spectral_slope(field: np.ndarray) -> float:
    """Computes isotropic 2D FFT radial power spectrum log-log slope."""
    fft = np.fft.fftshift(np.fft.fft2(field))
    power = np.abs(fft)**2
    
    ny, nx = field.shape
    cy, cx = ny // 2, nx // 2
    y, x = np.ogrid[-cy:ny-cy, -cx:nx-cx]
    r = np.hypot(x, y).astype(int)
    
    radial_sum = np.bincount(r.ravel(), power.ravel())
    radial_count = np.bincount(r.ravel())
    radial_prof = radial_sum / (radial_count + 1e-12)
    
    # Fit slope in inertial subrange
    k = np.arange(1, len(radial_prof))
    valid_k = (k > 2) & (k < len(k) // 2) & (radial_prof[1:] > 0)
    
    if not np.any(valid_k):
        return 0.0
        
    log_k = np.log(k[valid_k])
    log_p = np.log(radial_prof[1:][valid_k])
    slope, _ = np.polyfit(log_k, log_p, 1)
    return float(slope)

def compute_spatial_entropy(field: np.ndarray, bins: int = 32) -> float:
    """Computes Shannon spatial field intensity entropy."""
    hist, _ = np.histogram(field, bins=bins, density=True)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))

# --- Robust Coordinate & Variable Helpers ---

def find_variable(ds: xr.Dataset, candidates: list[str]) -> str:
    """Locates matching dataset variable or coordinate name."""
    for var in candidates:
        if var in ds.data_vars or var in ds.coords:
            return var
    # Case-insensitive fallback
    all_names = list(ds.data_vars.keys()) + list(ds.coords.keys())
    for var in candidates:
        for name in all_names:
            if var.lower() in name.lower():
                return name
    return None

def squeeze_to_2d(data_arr: np.ndarray) -> np.ndarray:
    """Squeezes multidimensional arrays down to 2D (lat, lon)."""
    data = np.squeeze(data_arr)
    while data.ndim > 2:
        data = data[0]
    return data

# --- ERA5 Pipeline Runner ---

def process_era5_case_study(nc_file_path: Path, output_dir: Path, window_deg: float = 20.0) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    ds = xr.open_dataset(nc_file_path)

    # Resolve time coordinate
    time_var = find_variable(ds, ['valid_time', 'time'])
    if not time_var:
        raise KeyError(f"Could not find time variable in dataset. Available: {list(ds.coords.keys())}")
    times = ds[time_var].values

    # Resolve MSLP and Geopotential variables
    mslp_var = find_variable(ds, ['msl', 'mslp', 'sp', 'mean_sea_level_pressure'])
    z_var = find_variable(ds, ['z', 'z500', 'geopotential'])

    if not mslp_var:
        raise KeyError(f"Could not find MSLP variable. Available variables: {list(ds.data_vars.keys())}")
    if not z_var:
        raise KeyError(f"Could not find Geopotential variable. Available variables: {list(ds.data_vars.keys())}")

    records = []

    print(f"\n================================================================================")
    print(f"RUNNING FROZEN ERA5 METRIC ENGINE: {nc_file_path.name}")
    print(f"Detected MSLP Var: '{mslp_var}' | Geopotential Var: '{z_var}'")
    print(f"Target Window: {window_deg}°x{window_deg}° | Total Frames: {len(times)}")
    print(f"================================================================================\n")

    for idx, t in enumerate(times):
        ds_frame = ds.sel({time_var: t})
        
        # Extract and squeeze MSLP field
        mslp_data = squeeze_to_2d(ds_frame[mslp_var].values)
        if np.nanmax(mslp_data) > 50000:  # Convert Pa to hPa
            mslp_hpa = mslp_data / 100.0
        else:
            mslp_hpa = mslp_data

        # Find center minimum MSLP coordinate
        min_idx = np.unravel_index(np.nanargmin(mslp_hpa), mslp_hpa.shape)
        min_mslp_val = float(mslp_hpa[min_idx])

        # Extract and squeeze 500 hPa Geopotential Height
        z_data = squeeze_to_2d(ds_frame[z_var].values)

        # Crop centered sub-grid for spatial analysis
        lat_len, lon_len = z_data.shape
        cy, cx = min_idx
        half_w = int((window_deg / 0.25) / 2)  # 0.25 deg resolution default

        y_min, y_max = max(0, cy - half_w), min(lat_len, cy + half_w)
        x_min, x_max = max(0, cx - half_w), min(lon_len, cx + half_w)

        sub_z = z_data[y_min:y_max, x_min:x_max]

        # Compute metric suite
        tb1 = compute_tb_v1(sub_z)
        tb2_intensity, tb2_cosine = compute_tb_v2_variants(sub_z)
        moran = compute_morans_i(sub_z)
        geary = compute_gearys_c(sub_z)
        grad_energy = compute_gradient_energy(sub_z)
        spectral_slope = compute_spectral_slope(sub_z)
        entropy = compute_spatial_entropy(sub_z)

        records.append({
            'time': pd.to_datetime(t),
            'min_mslp': min_mslp_val,
            'tb_v1': tb1,
            'tb_v2_intensity': tb2_intensity,
            'tb_v2_cosine': tb2_cosine,
            'tb_v2': tb2_intensity,  # Standard alias
            'morans_i': moran,
            'gearys_c': geary,
            'gradient_energy': grad_energy,
            'spectral_slope': spectral_slope,
            'spatial_entropy': entropy
        })

    ds.close()
    
    df = pd.DataFrame(records)
    csv_out = output_dir / "metrics.csv"
    df.to_csv(csv_out, index=False)
    print(f"[✓] Processing complete. Saved {len(df)} frames to: {csv_out}")
    return df