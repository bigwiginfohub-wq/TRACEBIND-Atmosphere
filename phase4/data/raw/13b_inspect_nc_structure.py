"""
13b_inspect_nc_structure.py
---------------------------
NetCDF Cohort Structure & Variable Inspector
Prints detailed Dataset metadata, variables, dimensions, and individual
variable checksums for every storm NetCDF file.
"""

import hashlib
from pathlib import Path
import xarray as xr

RAW_DATA_DIR = Path(r"C:\TRACEBIND-Atmosphere\phase4\data\raw")
OUTPUT_COHORT_DIR = RAW_DATA_DIR / "output_cohort"

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

def inspect_cohort():
    print("=========================================================================================")
    print("                NETCDF FILE & VARIABLE STRUCTURE INSPECTION                              ")
    print("=========================================================================================\n")

    for storm in STORMS:
        nc_path = resolve_nc_path(storm)
        print(f"=== STORM: {storm.upper()} ===")
        if nc_path is None or not nc_path.exists():
            print(f"  [-] Path: NOT FOUND\n")
            continue

        print(f"  [+] Resolved File: {nc_path}")
        file_size_mb = nc_path.stat().st_size / (1024 * 1024)
        print(f"  [+] File Size:     {file_size_mb:.2f} MB")

        try:
            ds = xr.open_dataset(nc_path)
            print(f"  [+] Dimensions:    {dict(ds.dims)}")
            print("  [+] Variables & Field Statistics:")

            for var_name in ds.data_vars:
                var_da = ds[var_name]
                vals = var_da.values
                # Compute raw byte MD5 hash of first slice
                slice_val = vals[0] if vals.ndim > 2 else vals
                var_hash = hashlib.md5(slice_val.tobytes()).hexdigest()[:8]

                print(
                    f"      - {var_name:20s} | Shape: {str(var_da.shape):16s} | "
                    f"Mean: {float(vals.mean()):10.2f} | Std: {float(vals.std()):8.2f} | Hash: {var_hash}"
                )

            ds.close()
        except Exception as e:
            print(f"  [-] Error inspecting NetCDF: {e}")
        print("-" * 89 + "\n")

if __name__ == "__main__":
    inspect_cohort()