"""
13a_phase5_implementation_audit.py
----------------------------------
Implementation Audit for TRACEBIND Phase 5:
1. Input Data Isolation & Uniqueness (MD5 Checksum)
2. Meteorological Diversity Check (Cross-correlation vs. previous storm)
3. Direct Sanity Printing (Shape, Mean, Std)
"""

import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

RAW_DATA_DIR = Path(r"C:\TRACEBIND-Atmosphere\phase4\data\raw")
OUTPUT_COHORT_DIR = RAW_DATA_DIR / "output_cohort"
SUMMARY_DIR = OUTPUT_COHORT_DIR / "summary"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

STORMS = ["Amphan", "Fani", "Mocha", "Yaas", "Sidr", "Nargis"]

def resolve_nc_path(storm_name: str) -> Path | None:
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
    return glob_matches[0] if glob_matches else None

def run_audits():
    print("=========================================================================================")
    print("                   TRACEBIND PHASE 5: DATA & INPUT UNIQUENESS AUDIT                     ")
    print("=========================================================================================\n")

    uniqueness_records = []
    previous_field = None

    for storm in STORMS:
        nc_path = resolve_nc_path(storm)
        if nc_path is None or not nc_path.exists():
            print(f"[-] Missing NetCDF file for storm: {storm}")
            continue

        ds = xr.open_dataset(nc_path)
        
        # Identify geopotential / pressure field
        z_var = next((v for v in ['z', 'geopotential', 'msl', 'mean_sea_level_pressure'] if v in ds), None)
        if z_var is None:
            print(f"[-] Geopotential/MSL variable not found in {nc_path.name}")
            ds.close()
            continue

        # Strict dimension check on ds[z_var].dims
        time_dim = next((d for d in ["valid_time", "time", "date"] if d in ds[z_var].dims), None)
        
        if time_dim is None:
            # Fallback to positional axis slicing if dimension name is unmapped
            slice_data = np.squeeze(ds[z_var].values[0]).astype(np.float64)
        else:
            slice_data = np.squeeze(ds[z_var].isel({time_dim: 0}).values).astype(np.float64)

        ds.close()

        # Immediate Sanity Log
        print(f"{storm:8s} | File: {nc_path.name:22s} | Shape: {str(slice_data.shape):14s} | Mean: {slice_data.mean():10.2f} | Std: {slice_data.std():8.2f}")

        # MD5 Checksum
        field_bytes = slice_data.tobytes()
        checksum = hashlib.md5(field_bytes).hexdigest()[:8]

        # Inter-storm Cross-Correlation Check
        if previous_field is not None and previous_field.shape == slice_data.shape:
            corr_prev = float(np.corrcoef(previous_field.ravel(), slice_data.ravel())[0, 1])
        else:
            corr_prev = np.nan

        previous_field = slice_data.copy()

        uniqueness_records.append({
            'Storm': storm,
            'File': nc_path.name,
            'Shape': str(slice_data.shape),
            'MD5_Checksum': checksum,
            'Mean': float(np.mean(slice_data)),
            'Std': float(np.std(slice_data)),
            'Corr_Prev': corr_prev
        })

    print("\n-----------------------------------------------------------------------------------------")
    print("                               UNIQUENESS AUDIT TABLE                                    ")
    print("-----------------------------------------------------------------------------------------")
    df_unique = pd.DataFrame(uniqueness_records)
    print(df_unique.to_string(index=False, formatters={'Corr_Prev': lambda x: f"{x:.4f}" if not np.isnan(x) else "N/A"}))

    summary_path = SUMMARY_DIR / "stage_p5_data_uniqueness_audit.csv"
    df_unique.to_csv(summary_path, index=False)
    print(f"\n[+] Audit summary saved to: {summary_path}")
    print("=========================================================================================\n")

if __name__ == "__main__":
    run_audits()