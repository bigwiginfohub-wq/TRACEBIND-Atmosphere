"""
34_compute_descriptors.py
-----------------------------------------------------
TRACEBIND Phase 5: Dynamic Row-Wise 5D Extraction Engine
"""

import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from tracebind_core import compute_row_wise_grid_spacing, compute_reduced_vector

SCRIPTS_DIR = Path(__file__).parent
PHASE5_DIR = SCRIPTS_DIR.parent
DATA_DIR = PHASE5_DIR / "data"
RESULTS_DIR = PHASE5_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CATALOG_PATH = DATA_DIR / "storm_catalog.csv"
DESCRIPTORS_OUT = RESULTS_DIR / "storm_descriptors_5d.csv"

def process_all_storms():
    catalog = pd.read_csv(CATALOG_PATH)
    print(f"[*] Extracting 5D vectors using row-wise dx(y) for {len(catalog)} catalog entries...\n")
    
    results = []
    for _, row in catalog.iterrows():
        storm_id = str(row['storm_id']).strip()
        nc_file = DATA_DIR / f"era5_{storm_id}_72h.nc"
        if not nc_file.exists():
            nc_file = DATA_DIR / f"era5_{storm_id}.nc"
            
        ds = xr.open_dataset(nc_file)
        dx_rows, dy = compute_row_wise_grid_spacing(ds)
        
        var_name = 'msl' if 'msl' in ds else 'mean_sea_level_pressure'
        field = ds[var_name].values
        field_2d = field[field.shape[0] // 2, :, :] if field.ndim >= 3 else field

        vec = compute_reduced_vector(field_2d, dx_rows=dx_rows, dy=dy)
        
        p_min = row.get("min_pressure_hpa", float('nan'))
        v_max = row.get("max_wind_kt", float('nan'))
        cohort = str(row.get("cohort_type", "Core")).strip()

        results.append({
            "storm_id": storm_id,
            "cohort_type": cohort,
            "min_pressure_hpa": p_min,
            "max_wind_kt": v_max,
            "mean_dx_meters": float(np.mean(dx_rows)),
            "dy_meters": dy,
            "GE": vec[0],
            "LE": vec[1],
            "C_orient": vec[2],
            "A_radial": vec[3],
            "S_orient": vec[4]
        })
        ds.close()
        print(f"  [✓] {storm_id} ({cohort}) | Mean dx={np.mean(dx_rows)/1000:.1f}km, dy={dy/1000:.1f}km | GE={vec[0]:.3f}, LE={vec[1]:.2e}")

    df_out = pd.DataFrame(results)
    df_out.to_csv(DESCRIPTORS_OUT, index=False)
    print(f"\n[*] Descriptors saved to: {DESCRIPTORS_OUT}")

if __name__ == "__main__":
    process_all_storms()