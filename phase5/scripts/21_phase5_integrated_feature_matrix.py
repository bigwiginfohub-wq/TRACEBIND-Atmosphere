"""
21_phase5_integrated_feature_matrix.py
---------------------------------------
TRACEBIND Phase 5: Integration Layer Matrix Builder (N=13 Cohort)

Pipeline Architecture:
  Measurement Layer (Phase 5A JSON) 
  --> Integration Layer (This Script) 
  --> Inference Layer (Phase 5B Permutation / Testing)

Refinements:
1. Data-Driven Metadata: Merges storm_metadata.csv rather than using hardcoded mappings.
2. Terminology Rigor: Renames min MSL to window_min_msl_hpa to clarify ERA5 spatial window scope.
3. Temporal Dynamics: Computes peak ratio (GE_peak / GE_density) and Coefficient of Variation (CV).
4. Provenance Fields: Attaches algorithm_id, pipeline_version, and generated_utc timestamp.
"""

import sys
import os
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import xarray as xr
import pandas as pd

# Base Directories
BASE_DIR = Path(r"C:\TRACEBIND-Atmosphere\phase5")
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PIPELINE_VERSION = "TRACEBIND-P5-INT-v1.0"
ALGORITHM_ID = "TRACEBIND-P5A-v1.1"

def compute_file_hash(filepath: Path, algorithm="md5") -> str:
    """Computes a hash for dataset tracking and auditability."""
    if not filepath.exists():
        return "N/A"
    hasher = hashlib.md5() if algorithm == "md5" else hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def generate_integrated_matrix():
    print("=========================================================================================")
    print("          TRACEBIND PHASE 5: INTEGRATION LAYER FEATURE MATRIX BUILDER (N=13)             ")
    print("=========================================================================================\n")

    p5a_json = RESULTS_DIR / "phase5A_gradient_energy_results.json"
    meta_csv = DATA_DIR / "storm_metadata.csv"

    if not p5a_json.exists():
        print(f"[-] Missing Phase 5A JSON output at {p5a_json}")
        return

    # Load External Metadata CSV
    metadata_df = None
    if meta_csv.exists():
        metadata_df = pd.read_csv(meta_csv)
        print(f"[+] Loaded external metadata for {len(metadata_df)} storms from {meta_csv.name}")
    else:
        print(f"[!] Warning: {meta_csv.name} not found. Proceeding with grid features only.")

    with open(p5a_json, "r", encoding="utf-8") as f:
        p5a_data = json.load(f)

    integrated_records = []

    for item in p5a_data:
        storm_id = item["storm_id"]
        feats = item["features"]
        nc_file = DATA_DIR / f"era5_{storm_id}_72h.nc"

        # Calculate dataset provenance hash
        dataset_md5 = compute_file_hash(nc_file, "md5")

        # Extract ERA5 window pressure statistics
        window_min_msl_hpa = np.nan
        window_msl_drop_hpa = np.nan

        if nc_file.exists():
            try:
                ds = xr.open_dataset(nc_file)
                msl_var = "msl" if "msl" in ds.data_vars else list(ds.data_vars.keys())[0]
                msl_vals = ds[msl_var].values  # Shape: (T, Lat, Lon)
                
                # Minimum pressure within the spatial grid window (hPa)
                window_min_msl_hpa = float(np.min(msl_vals)) / 100.0
                
                # Spatial-minimum series across time steps
                min_msl_series = np.min(msl_vals, axis=(-2, -1)) / 100.0
                window_msl_drop_hpa = float(np.max(min_msl_series) - np.min(min_msl_series))
                
                ds.close()
            except Exception as e:
                print(f"    [-] Error extracting window MSL stats for {storm_id}: {e}")

        # Compute derived temporal dynamic features
        ge_density = feats["GE_mean_integrated"]
        ge_peak = feats["GE_peak"]
        ge_std = feats["GE_std"]

        # 1. Peak Ratio (Temporal Concentration Factor)
        ge_peak_ratio = float(ge_peak / ge_density) if ge_density > 0 else np.nan

        # 2. Coefficient of Variation (Temporal Variability)
        ge_cv = float(ge_std / ge_density) if ge_density > 0 else np.nan

        record = {
            "storm_id": storm_id,
            "pipeline_version": PIPELINE_VERSION,
            "algorithm_id": ALGORITHM_ID,
            "dataset_md5": dataset_md5,
            "window_hours": item["window_length_hours"],
            "grid_cells": item["grid_cells"],
            "grid_shape": item["grid_shape"],
            "window_min_msl_hpa": round(window_min_msl_hpa, 2),
            "window_msl_drop_hpa": round(window_msl_drop_hpa, 2),
            "GE_density": ge_density,              # Pa^2 / m^2 (Spatial mean density)
            "GE_peak_density": ge_peak,            # Pa^2 / m^2 (Max single-frame density)
            "GE_peak_ratio": round(ge_peak_ratio, 3), # Peak / Density ratio
            "GE_cv": round(ge_cv, 3),                 # Coefficient of Variation over time
            "GE_total": feats["GE_total_integrated"],# Total spatial-temporal integral
            "p95_grad_max_Pam": feats["p95_gradient_mag_max"],
            "peak_frame_index": feats["peak_frame_index"]
        }
        integrated_records.append(record)

    df_integrated = pd.DataFrame(integrated_records)

    # Merge External Metadata if available
    if metadata_df is not None:
        df_integrated = df_integrated.merge(metadata_df, on="storm_id", how="left")

    # Sort descending by GE_density
    df_integrated = df_integrated.sort_values(by="GE_density", ascending=False).reset_index(drop=True)

    # Export Outputs
    csv_out = RESULTS_DIR / "phase5_integrated_feature_matrix.csv"
    json_out = RESULTS_DIR / "phase5_integrated_feature_matrix.json"

    df_integrated.to_csv(csv_out, index=False)

    meta_wrapper = {
        "pipeline_version": PIPELINE_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "cohort_size": len(df_integrated),
        "data": df_integrated.to_dict(orient="records")
    }
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(meta_wrapper, f, indent=2)

    print("=========================================================================================")
    print("                     PHASE 5 INTEGRATED FEATURE MATRIX SUMMARY                          ")
    print("=========================================================================================")
    display_cols = [
        "storm_id", "basin", "window_min_msl_hpa", "GE_density", 
        "GE_peak_ratio", "GE_cv", "p95_grad_max_Pam", "peak_frame_index"
    ]
    # Filter display columns to those present in DataFrame
    display_cols = [c for c in display_cols if c in df_integrated.columns]
    print(df_integrated[display_cols].to_string(index=False))

    print(f"\n[+] Feature Matrix exported successfully:")
    print(f"    - CSV:  {csv_out}")
    print(f"    - JSON: {json_out}\n")

if __name__ == "__main__":
    generate_integrated_matrix()