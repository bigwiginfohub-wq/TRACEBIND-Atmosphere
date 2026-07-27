"""
13c_audit_z_temporal_and_metadata.py
------------------------------------
Audit z-field identity and temporal stationarity across the cyclone cohort.
Checks:
1. Dataset coordinates & ds.sizes
2. Metadata attributes (GRIB shortName, paramId, typeOfLevel, long_name, etc.)
3. Temporal delta max(|z(t_end) - z(t_start)|) with np.isclose safety
"""

from pathlib import Path
import numpy as np
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

def audit_z_field():
    print("=========================================================================================")
    print("               TRACEBIND AUDIT: Z-VARIABLE METADATA & TEMPORAL DELTA                    ")
    print("=========================================================================================\n")

    for storm in STORMS:
        nc_path = resolve_nc_path(storm)
        print(f"=== STORM: {storm.upper()} ===")
        if nc_path is None or not nc_path.exists():
            print("  [-] Path: NOT FOUND\n")
            continue

        try:
            ds = xr.open_dataset(nc_path)
            
            # Use ds.sizes to future-proof against xarray deprecation
            print(f"  [+] Dataset Sizes:   {dict(ds.sizes)}")
            print(f"  [+] Coordinates:     {list(ds.coords.keys())}")

            if "z" not in ds:
                print("  [-] Variable 'z' not found in dataset.\n")
                ds.close()
                continue

            z_da = ds["z"]
            
            print(f"\n  --- Variable 'z' Details ---")
            print(f"  [+] Variable Shape:  {z_da.shape}")
            print(f"  [+] Dimensions:      {z_da.dims}")
            print(f"  [+] Attributes:")
            if z_da.attrs:
                for k, v in z_da.attrs.items():
                    print(f"      - {k:20s}: {v}")
            else:
                print("      (No attributes found)")

            # Identify time dimension
            time_dim = next((d for d in ["valid_time", "time", "date"] if d in z_da.dims), None)

            print(f"\n  --- Temporal Stationarity Check ---")
            if time_dim is None:
                print("  [-] No temporal dimension found on variable 'z'.")
            else:
                first_frame = np.squeeze(z_da.isel({time_dim: 0}).values).astype(np.float64)
                last_frame  = np.squeeze(z_da.isel({time_dim: -1}).values).astype(np.float64)
                
                max_diff = float(np.max(np.abs(last_frame - first_frame)))
                mean_diff = float(np.mean(np.abs(last_frame - first_frame)))

                print(f"  [+] Temporal Delta:  Max Abs Diff = {max_diff:.6f} | Mean Abs Diff = {mean_diff:.6f}")
                
                if np.isclose(max_diff, 0.0, atol=1e-12):
                    print("  [!] DIAGNOSTIC RESULT: STATIC FIELD (Max Abs Diff == 0.0)")
                    print("      Matches surface geopotential / topographical orography behavior.")
                else:
                    print("  [✓] DIAGNOSTIC RESULT: DYNAMIC FIELD (Evolves over time)")

            ds.close()
        except Exception as e:
            print(f"  [-] Audit error during execution: {e}")

        print("-" * 89 + "\n")

if __name__ == "__main__":
    audit_z_field()