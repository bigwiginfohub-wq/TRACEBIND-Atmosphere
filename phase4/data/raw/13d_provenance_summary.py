"""
13d_provenance_summary.py
-------------------------
Generates a provenance record mapping storm datasets to variable level types,
GRIB param IDs, MD5 hashes, and temporal dynamics.
"""

from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import xarray as xr

RAW_DATA_DIR = Path(r"C:\TRACEBIND-Atmosphere\phase4\data\raw")
STORMS = ["Amphan", "Fani", "Mocha", "Yaas", "Sidr", "Nargis"]

records = []
for storm in STORMS:
    paths = list(RAW_DATA_DIR.glob(f"*{storm.lower()}*.nc"))
    if not paths: continue
    ds = xr.open_dataset(paths[0])
    var_name = 'z' if 'z' in ds else ('msl' if 'msl' in ds else list(ds.data_vars.keys())[0])
    da = ds[var_name]
    time_dim = next((d for d in ["valid_time", "time", "date"] if d in da.dims), None)
    
    first = np.squeeze(da.isel({time_dim: 0}).values if time_dim else da.values).astype(np.float64)
    last = np.squeeze(da.isel({time_dim: -1}).values if time_dim else da.values).astype(np.float64)
    delta = float(np.max(np.abs(last - first)))
    md5 = hashlib.md5(first.tobytes()).hexdigest()[:8]
    
    records.append({
        'Storm': storm, 'Var': var_name,
        'Level_Type': da.attrs.get('GRIB_typeOfLevel', da.attrs.get('long_name', 'N/A')),
        'Param_ID': da.attrs.get('GRIB_paramId', 'N/A'),
        'Temporal_Delta': f"{delta:.2f}",
        'MD5_Frame0': md5
    })
    ds.close()

df = pd.DataFrame(records)
print("\n=== TRACEBIND DATASET PROVENANCE RECORD ===")
print(df.to_string(index=False))