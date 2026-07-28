#!/usr/bin/env python3
"""
TRACEBIND Phase 6C.2 — Kinematic & Vortex Feature Extraction Engine
-------------------------------------------------------------------
Input:  Verified ERA5 NetCDF artifacts (data/harvested/{system_id}.nc)
Output: Compressed feature NetCDF datasets (data/features/{system_id}_features.nc)
        & feature_manifest_6C2.json
"""

import sys
import json
import logging
import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import xarray as xr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Earth radius in meters
R_EARTH = 6371000.0


def compute_sha256(file_path: Path) -> str:
    """Computes SHA-256 hash of a file on disk."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_git_commit() -> str:
    """Safely retrieves current git commit hash if available."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
    except Exception:
        return "git_repo_not_found"


def compute_spherical_gradients_exact(u: np.ndarray, v: np.ndarray, lats: np.ndarray, lons: np.ndarray):
    """
    Computes spatial derivatives on a spherical grid using coordinate-aware gradient computation.
    Clamps cos(lat) to 1e-6 for polar stability.
    """
    lat_rad = np.radians(lats)
    lon_rad = np.radians(lons)

    # Gradient per radian along axes (axis 1 = latitude, axis 2 = longitude)
    du_dlat_rad = np.gradient(u, lat_rad, axis=1, edge_order=2)
    du_dlon_rad = np.gradient(u, lon_rad, axis=2, edge_order=2)
    dv_dlat_rad = np.gradient(v, lat_rad, axis=1, edge_order=2)
    dv_dlon_rad = np.gradient(v, lon_rad, axis=2, edge_order=2)

    # Polar stability clamp
    cos_lat = np.maximum(np.cos(lat_rad), 1e-6)

    # Broadcast shapes for 3D/4D arrays (time, lat, lon)
    if u.ndim == 3:
        cos_lat_bc = cos_lat[None, :, None]
    else:
        cos_lat_bc = cos_lat[:, None]

    # Convert to physical metric derivatives
    du_dy = du_dlat_rad / R_EARTH
    du_dx = du_dlon_rad / (R_EARTH * cos_lat_bc)
    dv_dy = dv_dlat_rad / R_EARTH
    dv_dx = dv_dlon_rad / (R_EARTH * cos_lat_bc)

    return du_dx, du_dy, dv_dx, dv_dy


def extract_features_for_system(nc_file: Path, output_file: Path, system_id: str, overwrite: bool = False) -> dict:
    """Extracts derived kinematic fields and writes a compressed feature NetCDF file."""
    if not nc_file.exists():
        logging.warning(f"Input NetCDF missing: {nc_file}")
        return {"status": "FAIL", "reason": "Missing source file"}

    source_hash = compute_sha256(nc_file)

    # Skip logic if already exists and valid
    if output_file.exists() and not overwrite:
        existing_hash = compute_sha256(output_file)
        logging.info(f"Feature file already exists for {system_id} (SHA: {existing_hash[:8]}...). Skipping.")
        return {
            "status": "SKIPPED",
            "system_id": system_id,
            "source_sha256": source_hash,
            "feature_sha256": existing_hash,
        }

    try:
        with xr.open_dataset(nc_file) as ds:
            u10 = ds["u10"].values
            v10 = ds["v10"].values

            # Safely resolve coordinate arrays
            lats_da = ds.coords.get("latitude", ds.coords.get("lat"))
            lons_da = ds.coords.get("longitude", ds.coords.get("lon"))
            lats = lats_da.values
            lons = lons_da.values

            # 1. Wind Speed Magnitude
            wind_speed = np.sqrt(u10**2 + v10**2)

            # 2. Exact Coordinate-Aware Derivatives
            du_dx, du_dy, dv_dx, dv_dy = compute_spherical_gradients_exact(u10, v10, lats, lons)

            # 3. Derived Fields
            vorticity = dv_dx - du_dy
            divergence = du_dx + dv_dy
            strain_normal = du_dx - dv_dy
            strain_shear = dv_dx + du_dy
            okubo_weiss = (strain_normal**2) + (strain_shear**2) - (vorticity**2)

            # 4. Construct Feature Dataset reusing source coordinates
            feat_ds = xr.Dataset(
                data_vars={
                    "wind_speed": (["time", lats_da.name, lons_da.name], wind_speed.astype(np.float32),
                                   {"units": "m s**-1", "long_name": "10m Wind Speed Magnitude"}),
                    "vorticity": (["time", lats_da.name, lons_da.name], vorticity.astype(np.float32),
                                  {"units": "s**-1", "long_name": "Relative Vorticity (zeta)"}),
                    "divergence": (["time", lats_da.name, lons_da.name], divergence.astype(np.float32),
                                   {"units": "s**-1", "long_name": "Horizontal Divergence (delta)"}),
                    "strain_normal": (["time", lats_da.name, lons_da.name], strain_normal.astype(np.float32),
                                      {"units": "s**-1", "long_name": "Normal Strain Rate (Sn)"}),
                    "strain_shear": (["time", lats_da.name, lons_da.name], strain_shear.astype(np.float32),
                                     {"units": "s**-1", "long_name": "Shear Strain Rate (Ss)"}),
                    "okubo_weiss": (["time", lats_da.name, lons_da.name], okubo_weiss.astype(np.float32),
                                    {"units": "s**-2", "long_name": "Okubo-Weiss Parameter (Q)"}),
                },
                coords=ds.coords,
            )

            # Global Lineage Metadata
            feat_ds.attrs["tracebind_feature_version"] = "6C.2"
            feat_ds.attrs["tracebind_system_id"] = system_id
            feat_ds.attrs["source_artifact_sha256"] = source_hash
            feat_ds.attrs["numerical_method"] = "centered_finite_difference_spherical_edge2"
            feat_ds.attrs["spatial_derivative_model"] = "Local tangent-plane approximation on sphere (metric curvature terms omitted)"
            feat_ds.attrs["creation_utc"] = datetime.now(timezone.utc).isoformat()
            feat_ds.attrs["git_commit"] = get_git_commit()
            feat_ds.attrs["python_version"] = sys.version.split()[0]
            feat_ds.attrs["numpy_version"] = np.__version__
            feat_ds.attrs["xarray_version"] = xr.__version__

            # Compressed NetCDF Output
            encoding = {
                var: {"zlib": True, "complevel": 4}
                for var in feat_ds.data_vars
            }

            output_file.parent.mkdir(parents=True, exist_ok=True)
            feat_ds.to_netcdf(output_file, encoding=encoding)

            feature_hash = compute_sha256(output_file)
            logging.info(f"Generated compressed features for {system_id} -> {output_file.name} (SHA: {feature_hash[:8]}...)")

            return {
                "status": "PASS",
                "system_id": system_id,
                "source_sha256": source_hash,
                "feature_sha256": feature_hash,
            }

    except Exception as e:
        logging.error(f"Failed feature extraction for {system_id}: {str(e)}")
        return {"status": "FAIL", "system_id": system_id, "reason": str(e)}


def locate_paths():
    """Autodetects project paths for TRACEBIND Phase 6."""
    cwd = Path.cwd()
    script_dir = Path(__file__).resolve().parent

    candidates = [cwd, script_dir, cwd.parent, script_dir.parent]

    catalog_path = None
    harvest_dir = None

    # Search for catalog.json
    for p in candidates:
        if (p / "catalog.json").exists():
            catalog_path = p / "catalog.json"
            break

    # Search for harvested raw NetCDF directory
    for p in candidates:
        if (p / "data" / "raw" / "era5_nc").exists():
            harvest_dir = p / "data" / "raw" / "era5_nc"
            break
        elif (p / "data" / "harvested").exists():
            harvest_dir = p / "data" / "harvested"
            break
        elif (p / "harvested").exists():
            harvest_dir = p / "harvested"
            break

    base_dir = catalog_path.parent if catalog_path else (script_dir.parent if script_dir.name == "phase6" else script_dir)
    features_dir = base_dir / "data" / "features"
    manifest_path = script_dir / "feature_manifest_6C2.json"

    return catalog_path, harvest_dir, features_dir, manifest_path

def main():
    logging.info("Starting Refined Phase 6C.2 Feature Extraction Engine...")

    catalog_path, harvest_dir, features_dir, manifest_path = locate_paths()

    if not catalog_path or not catalog_path.exists():
        logging.error("Could not find catalog.json in current or parent directories.")
        return

    if not harvest_dir or not harvest_dir.exists():
        logging.error("Could not find harvested data directory ('data/harvested' or 'harvested').")
        return

    logging.info(f"Using catalog: {catalog_path.absolute()}")
    logging.info(f"Reading harvested files from: {harvest_dir.absolute()}")
    logging.info(f"Writing features to: {features_dir.absolute()}")

    with open(catalog_path, "r") as f:
        catalog = json.load(f)

    systems = catalog.get("systems", [])
    manifest_records = []

    for sys_entry in systems:
        sys_id = sys_entry["system_id"]
        nc_file = harvest_dir / f"{sys_id}.nc"
        feat_file = features_dir / f"{sys_id}_features.nc"

        if nc_file.exists():
            res = extract_features_for_system(nc_file, feat_file, sys_id)
            manifest_records.append(res)
        else:
            logging.warning(f"Harvested NetCDF missing for system {sys_id}: {nc_file.absolute()}")

    manifest_payload = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "total_systems": len(manifest_records),
        "systems": manifest_records,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest_payload, f, indent=2)

    logging.info(f"Phase 6C.2 Feature Extraction Complete! Processed {len(manifest_records)} systems. Manifest saved to {manifest_path.absolute()}")


if __name__ == "__main__":
    main()