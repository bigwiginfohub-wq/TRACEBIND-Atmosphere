"""
tracebind_manifest.py
---------------------
TRACEBIND Phase 5 Reproducibility & Provenance Engine
Captures runtime environments, dataset MD5 checksums, random seeds,
and SHA-256 cryptographic hashes of analysis scripts.
"""

import sys
import os
import platform
import json
import hashlib
import subprocess
from datetime import datetime
from pathlib import Path
import numpy as np
import scipy
import xarray as xr
import pandas as pd

def compute_file_hash(filepath: Path, algorithm: str = "sha256") -> str:
    """Computes SHA-256 or MD5 hash of any file on disk."""
    if not filepath.exists():
        return "FILE_NOT_FOUND"
    
    hasher = hashlib.sha256() if algorithm == "sha256" else hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def get_git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
    except Exception:
        return "UNKNOWN_OR_NOT_A_GIT_REPO"

def build_reproducibility_manifest(
    storm_name: str,
    nc_filepath: Path,
    scripts_dir: Path = None,
    algorithm_id: str = "TRACEBIND-P5.0-FROZEN",
    n_permutations: int = 1000,
    seed: int = 42
) -> dict:
    """Generates an immutable provenance manifest including code & data hashes."""
    
    # Load dataset metadata
    ds_meta = {}
    if nc_filepath.exists():
        ds = xr.open_dataset(nc_filepath)
        ds_meta = {
            "time_steps": int(ds.sizes.get("valid_time", ds.sizes.get("time", 0))),
            "lat_range": [float(ds.latitude.min()), float(ds.latitude.max())],
            "lon_range": [float(ds.longitude.min()), float(ds.longitude.max())],
            "var_name": "msl" if "msl" in ds else list(ds.data_vars.keys())[0]
        }
        ds.close()

    # Hash critical analysis scripts
    script_hashes = {}
    if scripts_dir and scripts_dir.exists():
        for script_file in sorted(scripts_dir.glob("*.py")):
            script_hashes[script_file.name] = compute_file_hash(script_file, "sha256")

    manifest = {
        "metadata": {
            "algorithm_id": algorithm_id,
            "pipeline_version": "5.0.0",
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "git_commit": get_git_commit_hash(),
        },
        "script_provenance_sha256": script_hashes,
        "dataset": {
            "provider": "Copernicus Climate Data Store (ERA5)",
            "storm": storm_name,
            "input_file": nc_filepath.name,
            "input_md5": compute_file_hash(nc_filepath, "md5"),
            "time_coordinate": "valid_time",
            "grid_resolution": "0.25 deg",
            "dimensions": ds_meta
        },
        "environment": {
            "python_version": sys.version.split()[0],
            "os_platform": platform.platform(),
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "xarray_version": xr.__version__,
            "pandas_version": pd.__version__,
        },
        "random_generator_spec": {
            "explicit_rng": "np.random.default_rng(42)",
            "seed": seed,
            "n_permutations": n_permutations
        }
    }
    return manifest

if __name__ == "__main__":
    script_path = Path(__file__).parent
    dummy_nc = Path(r"C:\TRACEBIND-Atmosphere\phase5\data\era5_amphan_72h.nc")
    m = build_reproducibility_manifest("Amphan", dummy_nc, scripts_dir=script_path)
    print(json.dumps(m, indent=2))