"""
33_validate_netcdf.py
-----------------------------------------------------
Automated NetCDF Verification Protocol for TRACEBIND Phase 5.
Validates variables, dimensions, spatial grids, NaN absence,
and pressure value plausibility.
"""

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path

# Explicit relative path binding for Phase 5 structure
SCRIPTS_DIR = Path(__file__).parent
PHASE5_DIR = SCRIPTS_DIR.parent
DATA_DIR = PHASE5_DIR / "data"
CATALOG_PATH = DATA_DIR / "storm_catalog.csv"

# Physical Thresholds
MSL_MIN_PA = 85000.0   # 850 hPa
MSL_MAX_PA = 105000.0  # 1050 hPa

def validate_single_netcdf(nc_path: Path) -> bool:
    if not nc_path.exists():
        print(f"  [-] Missing File: {nc_path.name}")
        return False
        
    try:
        ds = xr.open_dataset(nc_path)
        
        # 1. Variable Verification
        if 'msl' not in ds and 'mean_sea_level_pressure' not in ds:
            print(f"  [-] {nc_path.name}: Missing sea level pressure ('msl').")
            ds.close()
            return False
            
        var_name = 'msl' if 'msl' in ds else 'mean_sea_level_pressure'
        data = ds[var_name].values
        
        # 2. NaN / Inf Check
        if np.isnan(data).any() or np.isinf(data).any():
            print(f"  [-] {nc_path.name}: Contains NaNs or Infs.")
            ds.close()
            return False
            
        # 3. Minimum Grid Point Check
        if data.size < 1000:
            print(f"  [-] {nc_path.name}: Grid domain is incomplete ({data.size} points).")
            ds.close()
            return False
            
        # 4. Plausible Pressure Values
        min_p, max_p = np.min(data), np.max(data)
        if min_p < MSL_MIN_PA or max_p > MSL_MAX_PA:
            print(f"  [-] {nc_path.name}: Pressure out of bounds [{min_p:.1f}, {max_p:.1f}] Pa.")
            ds.close()
            return False
            
        ds.close()
        return True

    except Exception as e:
        print(f"  [-] {nc_path.name}: File read error ({e}).")
        return False

def run_validation():
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(f"Catalog missing at: {CATALOG_PATH}")
        
    catalog = pd.read_csv(CATALOG_PATH)
    print(f"[*] Validating {len(catalog)} catalog entries against {DATA_DIR}...\n")
    
    passed, failed = 0, 0
    for idx, row in catalog.iterrows():
        storm_id = str(row['storm_id']).strip()
        
        # Check primary naming convention: era5_<storm_id>_72h.nc
        nc_file = DATA_DIR / f"era5_{storm_id}_72h.nc"
        if not nc_file.exists():
            # Check secondary convention: era5_<storm_id>.nc
            nc_file = DATA_DIR / f"era5_{storm_id}.nc"
            
        is_valid = validate_single_netcdf(nc_file)
        
        if is_valid:
            passed += 1
            print(f"  [✓] {storm_id} ({nc_file.name}): PASS")
        else:
            failed += 1
            print(f"  [✗] {storm_id}: FAIL")
            
    print(f"\n[*] Validation Complete. Passed: {passed} | Failed: {failed}")
    if failed > 0:
        raise RuntimeError(f"Validation failed for {failed} files.")

if __name__ == "__main__":
    run_validation()