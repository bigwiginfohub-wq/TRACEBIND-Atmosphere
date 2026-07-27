"""
13a_phase5_implementation_audit.py
----------------------------------
Implementation Audit for TRACEBIND Phase 5 Falsification Engine.
Tests data array uniqueness, field deformation, metric sensitivity, and p-value math.
"""

import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

RAW_DATA_DIR = Path(r"C:\TRACEBIND-Atmosphere\phase4\data\raw")
OUTPUT_COHORT_DIR = RAW_DATA_DIR / "output_cohort"

STORMS = ["Amphan", "Fani", "Mocha", "Yaas", "Sidr", "Nargis"]
OMEGA = 7.2921e-5

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

def compute_geostrophic_vorticity(z_frame, lats):
    dy = 111000.0
    mean_lat = np.mean(lats)
    dx = 111000.0 * np.cos(np.radians(mean_lat))
    f = 2.0 * OMEGA * np.sin(np.radians(lats))
    f = np.where(np.abs(f) < 1e-5, 1e-5, f)[:, None]
    
    dz_dy, dz_dx = np.gradient(z_frame, dy, dx)
    u_g = -(1.0 / f) * dz_dy
    v_g = (1.0 / f) * dz_dx
    
    dvg_dx = np.gradient(v_g, dx, axis=1)
    dug_dy = np.gradient(u_g, dy, axis=0)
    return dvg_dx - dug_dy

def compute_tb_v2(field):
    gy, gx = np.gradient(field)
    grad_mag = np.sqrt(gx**2 + gy**2)
    ge = float(np.mean(grad_mag**2))
    tb2 = float(np.sqrt(ge) * np.std(field))
    return ge, tb2

def generate_synthetic_vortex(shape=(100, 100), sigma=15.0):
    y, x = np.ogrid[:shape[0], :shape[1]]
    center_y, center_x = shape[0] // 2, shape[1] // 2
    r2 = (x - center_x)**2 + (y - center_y)**2
    return 100.0 * np.exp(-r2 / (2.0 * sigma**2))

def run_audits():
    print("=========================================================================================")
    print("                      TRACEBIND PHASE 5: IMPLEMENTATION AUDIT                            ")
    print("=========================================================================================\n")

    # ---------------------------------------------------------------------------------
    # AUDIT 1: DATA UNIQUENESS & SHA-256 HASHES
    # ---------------------------------------------------------------------------------
    print("--- AUDIT 1: Storm Input Uniqueness Check ---")
    storm_hashes = {}
    for storm in STORMS:
        path = resolve_nc_path(storm)
        if path and path.exists():
            ds = xr.open_dataset(path)
            z_var = next(v for v in ['z', 'geopotential'] if v in ds)
            slice_data = np.squeeze(ds[z_var].isel(time=0).values).astype(np.float64)
            data_bytes = slice_data.tobytes()
            data_hash = hashlib.sha256(data_bytes).hexdigest()[:12]
            storm_hashes[storm] = {
                'Path': path.name,
                'Shape': slice_data.shape,
                'Mean_Z': float(np.mean(slice_data)),
                'SHA256': data_hash
            }
            ds.close()
        else:
            storm_hashes[storm] = {'Path': 'MISSING', 'SHA256': 'N/A'}

    df_hash = pd.DataFrame.from_dict(storm_hashes, orient='index')
    print(df_hash.to_string())
    print()

    # ---------------------------------------------------------------------------------
    # AUDIT 2: FIELD SHUFFLE & METRIC SENSITIVITY
    # ---------------------------------------------------------------------------------
    print("--- AUDIT 2: Spatial Shuffle Deformation & Autocorrelation Check ---")
    synth = generate_synthetic_vortex()
    ge_orig, tb2_orig = compute_tb_v2(synth)

    # Real spatial shuffle (flatten, shuffle copy, reshape)
    synth_shuffled = synth.copy().flatten()
    np.random.shuffle(synth_shuffled)
    synth_shuffled = synth_shuffled.reshape(synth.shape)
    ge_shuff, tb2_shuff = compute_tb_v2(synth_shuffled)

    corr_matrix = np.corrcoef(synth.flatten(), synth_shuffled.flatten())[0, 1]

    print(f"Synthetic Vortex Original GE: {ge_orig:.6e} | TB2: {tb2_orig:.6e}")
    print(f"Synthetic Vortex Shuffled GE: {ge_shuff:.6e} | TB2: {tb2_shuff:.6e}")
    print(f"GE Delta: {((ge_shuff - ge_orig) / ge_orig) * 100.0:.2f}%")
    print(f"Spatial Correlation (Orig vs Shuffled): {corr_matrix:.4f}")
    print()

    # ---------------------------------------------------------------------------------
    # AUDIT 3: P-VALUE & Z-SCORE FORMULA INSPECTION
    # ---------------------------------------------------------------------------------
    print("--- AUDIT 3: P-Value Logic & Sign Inspection ---")
    obs_val = 0.0
    null_dist = np.random.normal(loc=51.0, scale=32.0, size=1000)
    
    mean_null = np.mean(null_dist)
    std_null = np.std(null_dist)
    z_calc = (obs_val - mean_null) / std_null

    # Compare formulas
    p_two_sided = np.mean(np.abs(null_dist) >= np.abs(obs_val))
    p_left_tail = np.mean(null_dist <= obs_val)
    p_right_tail = np.mean(null_dist >= obs_val)

    print(f"Observed: {obs_val} | Null Mean: {mean_null:.2f} | Null Std: {std_null:.2f}")
    print(f"Calculated Z-Score: {z_calc:.2f}")
    print(f"p (Two-Sided |null| >= |obs|): {p_two_sided:.4f}  <-- Explains why p was 1.000!")
    print(f"p (Left-Tailed null <= obs):  {p_left_tail:.4f}")
    print(f"p (Right-Tailed null >= obs): {p_right_tail:.4f}")
    print("\n=========================================================================================\n")

if __name__ == "__main__":
    run_audits()