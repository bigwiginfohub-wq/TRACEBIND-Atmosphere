"""
20_phase5A_compute_gradient_energy.py
-------------------------------------
TRACEBIND Phase 5A: Spatial Gradient Energy Feature Extractor (N=13 Cohort)

Algorithm Version: TRACEBIND-P5A-v1.1 (Fixed Spherical Spacing Bug)
"""

import sys
import os
import json
from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd

# Base Directories
BASE_DIR = Path(r"C:\TRACEBIND-Atmosphere\phase5")
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EARTH_RADIUS_M = 6371000.0  # Mean Earth radius in meters

def compute_spherical_gradients(msl_field: np.ndarray, lats: np.ndarray, lons: np.ndarray):
    """
    Computes spatial pressure gradients (Gx, Gy) accounting for spherical latitude convergence.
    """
    # Latitude step size in meters (dy)
    dlat_deg = abs(float(lats[1] - lats[0])) if len(lats) > 1 else 0.25
    dy = np.radians(dlat_deg) * EARTH_RADIUS_M
    
    # Gy along axis 0 (lat)
    gy = np.gradient(msl_field, dy, axis=0)
    
    # Longitude step size in meters (dx) per row
    dlon_deg = abs(float(lons[1] - lons[0])) if len(lons) > 1 else 0.25
    dlon_rad = np.radians(dlon_deg)
    
    gx = np.zeros_like(msl_field)
    for i, lat_deg in enumerate(lats):
        lat_rad = np.radians(lat_deg)
        dx_meters = dlon_rad * EARTH_RADIUS_M * np.cos(lat_rad)
        gx[i, :] = np.gradient(msl_field[i, :], dx_meters)
        
    grad_sq = gx**2 + gy**2
    return grad_sq, gx, gy

def run_phase5A():
    print("=========================================================================================")
    print("      TRACEBIND PHASE 5A: SPHERICAL GRADIENT ENERGY FEATURE EXTRACTOR (N=13)            ")
    print("=========================================================================================\n")

    nc_files = sorted(list(DATA_DIR.glob("era5_*_72h.nc")))
    if not nc_files:
        print("[-] No NetCDF files found in data directory!")
        return

    phase5A_results = []

    for nc_path in nc_files:
        storm_id = nc_path.stem.replace("era5_", "").replace("_72h", "")
        print(f"[*] Processing Storm: {storm_id.upper()}")

        try:
            ds = xr.open_dataset(nc_path)
        except Exception as e:
            print(f"    [-] Error loading {nc_path.name}: {e}")
            continue

        lat_coord = "latitude" if "latitude" in ds.coords else "lat"
        lon_coord = "longitude" if "longitude" in ds.coords else "lon"
        msl_var = "msl" if "msl" in ds.data_vars else list(ds.data_vars.keys())[0]

        lats = ds[lat_coord].values
        lons = ds[lon_coord].values
        msl_data = ds[msl_var].values  # Shape: (T, Lat, Lon)

        t_steps = msl_data.shape[0]
        n_cells = len(lats) * len(lons)

        ge_mean_series = []
        ge_total_series = []
        p95_grad_series = []

        for t in range(t_steps):
            grad_sq, gx, gy = compute_spherical_gradients(msl_data[t], lats, lons)
            
            ge_tot = float(np.sum(grad_sq))
            ge_m = float(np.mean(grad_sq))
            grad_mag = np.sqrt(grad_sq)
            p95_mag = float(np.percentile(grad_mag, 95))

            ge_total_series.append(ge_tot)
            ge_mean_series.append(ge_m)
            p95_grad_series.append(p95_mag)

        ge_total_series = np.array(ge_total_series)
        ge_mean_series = np.array(ge_mean_series)
        p95_grad_series = np.array(p95_grad_series)

        peak_frame_idx = int(np.argmax(ge_mean_series))

        storm_record = {
            "algorithm_version": "TRACEBIND-P5A-v1.1",
            "storm_id": storm_id,
            "window_length_hours": t_steps,
            "grid_cells": n_cells,
            "grid_shape": f"{len(lats)}x{len(lons)}",
            "features": {
                "GE_total_integrated": float(np.sum(ge_total_series)),
                "GE_mean_integrated": float(np.mean(ge_mean_series)),
                "GE_peak": float(np.max(ge_mean_series)),
                "GE_median": float(np.median(ge_mean_series)),
                "GE_std": float(np.std(ge_mean_series)),
                "peak_frame_index": peak_frame_idx,
                "p95_gradient_mag_max": float(np.max(p95_grad_series)),
                "p95_gradient_mag_mean": float(np.mean(p95_grad_series))
            }
        }

        phase5A_results.append(storm_record)

        print(f"    -> Window: {t_steps}h | Grid: {len(lats)}x{len(lons)} ({n_cells} cells)")
        print(f"    -> GE_total (Sum): {storm_record['features']['GE_total_integrated']:.4e}")
        print(f"    -> GE_mean Density: {storm_record['features']['GE_mean_integrated']:.4e} Pa^2/m^2")
        print(f"    -> GE_peak Density: {storm_record['features']['GE_peak']:.4e} Pa^2/m^2 (Frame {peak_frame_idx})")
        print(f"    -> P95 Grad Mag (Max): {storm_record['features']['p95_gradient_mag_max']:.4e} Pa/m\n")

        ds.close()

    # 1. Save JSON Output
    json_out = RESULTS_DIR / "phase5A_gradient_energy_results.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(phase5A_results, f, indent=2)

    # 2. Save CSV Output
    csv_rows = []
    for rec in phase5A_results:
        row = {
            "storm_id": rec["storm_id"],
            "version": rec["algorithm_version"],
            "window_hours": rec["window_length_hours"],
            "grid_cells": rec["grid_cells"],
            "grid_shape": rec["grid_shape"],
            "GE_total": rec["features"]["GE_total_integrated"],
            "GE_mean": rec["features"]["GE_mean_integrated"],
            "GE_peak": rec["features"]["GE_peak"],
            "GE_median": rec["features"]["GE_median"],
            "GE_std": rec["features"]["GE_std"],
            "peak_frame": rec["features"]["peak_frame_index"],
            "p95_grad_max_Pam": rec["features"]["p95_gradient_mag_max"],
            "p95_grad_mean_Pam": rec["features"]["p95_gradient_mag_mean"]
        }
        csv_rows.append(row)

    df_csv = pd.DataFrame(csv_rows)
    csv_out = RESULTS_DIR / "phase5A_gradient_energy_results.csv"
    df_csv.to_csv(csv_out, index=False)

    print("=========================================================================================")
    print("                     PHASE 5A FEATURE EXTRACTION SUMMARY                                ")
    print("=========================================================================================")
    summary_cols = ["storm_id", "window_hours", "grid_cells", "GE_total", "GE_mean", "GE_peak", "p95_grad_max_Pam"]
    print(df_csv[summary_cols].to_string(index=False))
    
    print(f"\n[+] Results written to:")
    print(f"    - JSON: {json_out}")
    print(f"    - CSV:  {csv_out}\n")

if __name__ == "__main__":
    run_phase5A()