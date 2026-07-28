"""
Phase 6B.5: Scientific Input Verification & Dataset Characterization
"""

import os
import json
import glob
import xarray as xr
import numpy as np

def verify_dataset_cohort(data_dir: str = "data/raw/era5_nc"):
    nc_files = glob.glob(os.path.join(data_dir, "*.nc"))
    if not nc_files:
        print("[PHASE 6B.5] No NetCDF artifacts found to verify.")
        return

    print(f"[PHASE 6B.5] Initializing Scientific Input Verification across {len(nc_files)} files...\n")
    
    report = {
        "files_inspected": 0,
        "coordinate_conventions_passed": True,
        "grid_spacing_uniform_0_25": True,
        "units_verified": True,
        "missing_values_detected": False,
        "errors": []
    }

    expected_units = {
        "u10": "m s**-1",
        "v10": "m s**-1",
        "msl": "Pa"
    }

    for filepath in nc_files:
        filename = os.path.basename(filepath)
        report["files_inspected"] += 1
        
        try:
            ds = xr.open_dataset(filepath)
            
            # 1. Coordinate Check: Latitude ordering & Longitude normalization [-180, 180]
            lats = ds.coords.get("latitude") or ds.coords.get("lat")
            lons = ds.coords.get("longitude") or ds.coords.get("lon")

            if lats is None or lons is None:
                report["coordinate_conventions_passed"] = False
                report["errors"].append(f"{filename}: Missing latitude/longitude coordinates.")
                continue

            if np.any(lons.values > 180.0) or np.any(lons.values < -180.0):
                report["coordinate_conventions_passed"] = False
                report["errors"].append(f"{filename}: Longitude values outside normalized [-180, 180] range.")

            # 2. Grid Spacing Validation (Strict 0.25 deg)
            lat_diffs = np.abs(np.diff(lats.values))
            lon_diffs = np.abs(np.diff(lons.values))
            
            if not np.allclose(lat_diffs, 0.25, atol=1e-3) or not np.allclose(lon_diffs, 0.25, atol=1e-3):
                report["grid_spacing_uniform_0_25"] = False
                report["errors"].append(f"{filename}: Non-uniform or non-0.25 degree grid spacing detected.")

            # 3. Unit Consistency Verification
            for var, expected_unit in expected_units.items():
                if var in ds:
                    actual_unit = ds[var].attrs.get("units", "").strip()
                    # Standardize unit representation comparisons
                    if actual_unit not in [expected_unit, "m/s", "m s-1", "Pa", "hPa"]:
                        report["units_verified"] = False
                        report["errors"].append(f"{filename}: Variable '{var}' has unexpected unit '{actual_unit}'.")

            # 4. Null / NaN / Masked Value Audit
            for var in ["u10", "v10", "msl"]:
                if var in ds and ds[var].isnull().any():
                    report["missing_values_detected"] = True
                    report["errors"].append(f"{filename}: Variable '{var}' contains unmasked NaN values.")

            ds.close()

        except Exception as e:
            report["errors"].append(f"{filename}: Failed to parse NetCDF file. Exception: {str(e)}")

    print("=" * 60)
    print("      PHASE 6B.5 DATASET CHARACTERIZATION REPORT")
    print("=" * 60)
    print(f"Total Cohort Files Verified : {report['files_inspected']}")
    print(f"Coordinates Normalized     : {'✅ PASS' if report['coordinate_conventions_passed'] else '❌ FAIL'}")
    print(f"Grid Spacing Uniform (0.25°): {'✅ PASS' if report['grid_spacing_uniform_0_25'] else '❌ FAIL'}")
    print(f"Physical Units Consistent  : {'✅ PASS' if report['units_verified'] else '❌ FAIL'}")
    print(f"Zero NaN/Null Masking      : {'✅ PASS' if not report['missing_values_detected'] else '❌ FAIL'}")
    print("=" * 60)

    if report["errors"]:
        print("\n[WARNING] Errors Detected during Phase 6B.5:")
        for err in report["errors"]:
            print(f"  - {err}")
    else:
        print("\n[PHASE 6B.5 SUCCESS] Cohort datasets are physically verified and ready for Phase 6C wave analysis.")

if __name__ == "__main__":
    verify_dataset_cohort()