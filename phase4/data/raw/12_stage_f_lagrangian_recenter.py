"""
12_stage_f_lagrangian_recenter.py
---------------------------------
Stage F: Lagrangian Storm-Relative Spatial Extraction
1. Locates minimum MSLP (cyclone center) dynamically at each hourly timestep.
2. Extracts a circular spatial disk (R = 500 km) around the dynamic center.
3. Computes geostrophic vorticity from Geopotential Height ('z').
4. Computes spatial metrics strictly within the dynamic storm frame.
5. Outputs Eulerian vs. Lagrangian Lead Shift comparison: Delta(Delta t).
"""

from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr

RAW_DATA_DIR = Path(r"C:\TRACEBIND-Atmosphere\phase4\data\raw")
OUTPUT_COHORT_DIR = RAW_DATA_DIR / "output_cohort"
SUMMARY_DIR = OUTPUT_COHORT_DIR / "summary"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

STORMS = ["Amphan", "Fani", "Mocha", "Yaas", "Sidr", "Nargis"]
RADIUS_KM = 500.0
OMEGA = 7.2921e-5  # Earth's rotation rate (rad/s)

def resolve_nc_path(storm_name: str) -> Path | None:
    """Robustly searches for the storm's NetCDF file across possible naming schemes."""
    name_low = storm_name.lower()
    
    candidates = [
        RAW_DATA_DIR / f"era5_{name_low}_72h.nc",
        RAW_DATA_DIR / f"era5_{name_low}.nc",
        RAW_DATA_DIR / f"{name_low}_era5.nc",
        RAW_DATA_DIR / f"{storm_name}_era5.nc",
        OUTPUT_COHORT_DIR / storm_name / f"{name_low}_era5.nc",
        OUTPUT_COHORT_DIR / storm_name / f"era5_{name_low}_72h.nc",
    ]
    
    for p in candidates:
        if p.exists():
            return p
            
    glob_matches = list(RAW_DATA_DIR.glob(f"*{name_low}*.nc"))
    if glob_matches:
        return glob_matches[0]
        
    return None

def haversine_mask(lats, lons, center_lat, center_lon, radius_km):
    """Generates a boolean radial spatial mask (True within radius_km)."""
    R_earth = 6371.0
    lat_rad = np.radians(lats[:, None])
    lon_rad = np.radians(lons[None, :])
    clat_rad = np.radians(center_lat)
    clon_rad = np.radians(center_lon)

    dlat = lat_rad - clat_rad
    dlon = lon_rad - clon_rad

    a = np.sin(dlat / 2.0)**2 + np.cos(lat_rad) * np.cos(clat_rad) * np.sin(dlon / 2.0)**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    dist = R_earth * c
    return dist <= radius_km

def compute_geostrophic_vorticity(z_frame, lats, lons):
    """Computes geostrophic vorticity from geopotential height (z)."""
    dy = 111000.0  # ~111 km per deg lat
    mean_lat = np.mean(lats)
    dx = 111000.0 * np.cos(np.radians(mean_lat))

    f = 2.0 * OMEGA * np.sin(np.radians(lats))
    f = np.where(np.abs(f) < 1e-5, 1e-5, f)
    f_2d = f[:, None]

    dz_dy, dz_dx = np.gradient(z_frame, dy, dx)
    u_g = -(1.0 / f_2d) * dz_dy
    v_g = (1.0 / f_2d) * dz_dx

    dvg_dx = np.gradient(v_g, dx, axis=1)
    dug_dy = np.gradient(u_g, dy, axis=0)
    vorticity = dvg_dx - dug_dy
    return vorticity

def compute_lagrangian_tb2(field, mask):
    """Computes TB-v2 Intensity within the spatial mask."""
    vals = field[mask]
    if len(vals) == 0 or np.all(np.isnan(vals)):
        return 0.0, 0.0
    
    gy, gx = np.gradient(field)
    grad_mag = np.sqrt(gx**2 + gy**2)[mask]
    
    grad_energy = float(np.mean(grad_mag**2))
    tb_v2_intensity = float(np.sqrt(grad_energy) * np.std(vals))
    return grad_energy, tb_v2_intensity

def process_lagrangian_storm(storm_name: str):
    nc_path = resolve_nc_path(storm_name)
    if nc_path is None or not nc_path.exists():
        print(f"[-] Could not find NetCDF file for storm: {storm_name}")
        return None

    print(f"[+] Processing Lagrangian extraction for {storm_name} using: {nc_path.name}")
    ds = xr.open_dataset(nc_path)
    
    # Dynamic dimension detection for lat/lon/time
    lat_col = next((c for c in ['latitude', 'lat', 'y'] if c in ds or c in ds.coords), None)
    lon_col = next((c for c in ['longitude', 'lon', 'x'] if c in ds or c in ds.coords), None)
    time_col = next((c for c in ['valid_time', 'time', 'valid_time_utc', 'date'] if c in ds or c in ds.coords), None)

    if not all([lat_col, lon_col, time_col]):
        print(f"[-] Could not identify lat/lon/time coordinates in {nc_path.name}.")
        print(f"    Available coordinates/dims: {list(ds.coords.keys())} / {list(ds.dims.keys())}")
        ds.close()
        return None

    msl_var = next((v for v in ['msl', 'mean_sea_level_pressure', 'mslp'] if v in ds), None)
    z_var = next((v for v in ['z', 'geopotential'] if v in ds), None)

    if not all([msl_var, z_var]):
        print(f"[-] Missing required variables (msl, z) in {nc_path.name}. Found: {list(ds.keys())}")
        ds.close()
        return None

    lats = ds[lat_col].values
    lons = ds[lon_col].values
    times = ds[time_col].values

    records = []

    for t_idx, t in enumerate(times):
        msl_frame = np.squeeze(ds[msl_var].isel({time_col: t_idx}).values)
        z_frame = np.squeeze(ds[z_var].isel({time_col: t_idx}).values)
        
        # Dynamic storm center (min MSLP)
        min_idx = np.unravel_index(np.argmin(msl_frame), msl_frame.shape)
        c_lat, c_lon = lats[min_idx[0]], lons[min_idx[1]]
        
        raw_min = msl_frame[min_idx]
        min_mslp_val = raw_min / 100.0 if raw_min > 50000 else raw_min

        mask = haversine_mask(lats, lons, c_lat, c_lon, RADIUS_KM)

        # Geostrophic vorticity from geopotential height
        vorticity = compute_geostrophic_vorticity(z_frame, lats, lons)

        ge_lag, tb2_lag = compute_lagrangian_tb2(vorticity, mask)

        records.append({
            'time': pd.to_datetime(t),
            'center_lat': c_lat,
            'center_lon': c_lon,
            'min_mslp_hpa': min_mslp_val,
            'ge_lagrangian': ge_lag,
            'tb2_lagrangian': tb2_lag
        })

    ds.close()
    return pd.DataFrame(records)

def run_stage_f_comparison():
    e3_path = SUMMARY_DIR / "stage_e3_dimensionless_features.csv"
    if not e3_path.exists():
        print("[-] E-3 summary file missing. Ensure 11_stage_e3_dimensionless.py has run.")
        return
    
    df_e3 = pd.read_csv(e3_path)
    records = []

    print("\n=========================================================================================")
    print("             STAGE F: LAGRANGIAN (STORM-RELATIVE) RE-CENTERING AUDIT                    ")
    print("=========================================================================================")

    for storm in STORMS:
        df_lag = process_lagrangian_storm(storm)
        if df_lag is None:
            continue

        df_lag['dp_dt'] = np.gradient(df_lag['min_mslp_hpa'], 1.0)
        idx_max_deep = df_lag['dp_dt'].idxmin()
        t_max_deep = df_lag.loc[idx_max_deep, 'time']

        for m_name, col_lag in [('gradient_energy', 'ge_lagrangian'), ('tb_v2_intensity', 'tb2_lagrangian')]:
            df_lag[f'd_{m_name}_dt'] = np.gradient(df_lag[col_lag], 1.0)
            idx_max_growth = df_lag[f'd_{m_name}_dt'].idxmax()
            t_max_growth = df_lag.loc[idx_max_growth, 'time']
            
            lead_lagrangian = (t_max_deep - t_max_growth).total_seconds() / 3600.0

            # Fetch Eulerian lead from Stage E-3
            e3_match = df_e3[(df_e3['Storm'] == storm) & (df_e3['Metric'] == m_name)]
            lead_eulerian = e3_match['Lead_vs_MaxDeepening_h'].values[0] if len(e3_match) > 0 else np.nan

            delta_shift = lead_lagrangian - lead_eulerian

            records.append({
                'Storm': storm,
                'Metric': m_name,
                'Lead_Eulerian_h': lead_eulerian,
                'Lead_Lagrangian_h': lead_lagrangian,
                'Delta_Shift_h': delta_shift
            })

    results_df = pd.DataFrame(records)
    results_df.to_csv(SUMMARY_DIR / "stage_f_lagrangian_vs_eulerian.csv", index=False)

    print("\n" + results_df.to_string(index=False))
    print("=========================================================================================\n")

if __name__ == "__main__":
    run_stage_f_comparison()