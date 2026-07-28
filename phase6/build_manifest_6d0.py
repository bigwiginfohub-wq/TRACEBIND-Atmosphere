import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
import xarray as xr

def compute_sha256(file_path: Path) -> str:
    """Computes SHA-256 hash of a file in 64KB chunks."""
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()

def locate_paths():
    cwd = Path.cwd()
    script_dir = Path(__file__).resolve().parent
    candidates = [cwd, script_dir, cwd.parent, script_dir.parent]

    catalog_path = None
    manifest_6c2_path = None
    features_dir = None

    for p in candidates:
        if (p / "catalog.json").exists():
            catalog_path = p / "catalog.json"
        if (p / "feature_manifest_6C2.json").exists():
            manifest_6c2_path = p / "feature_manifest_6C2.json"
        if (p / "data" / "features").exists():
            features_dir = p / "data" / "features"

    return catalog_path, manifest_6c2_path, features_dir

def build_manifest_6d0():
    print("[INFO] Starting Phase 6D.0 Master Feature Indexing Engine...")
    catalog_path, manifest_6c2_path, features_dir = locate_paths()

    if not catalog_path or not catalog_path.exists():
        raise FileNotFoundError(f"[ERROR] Could not find authoritative catalog.json")
    if not manifest_6c2_path or not manifest_6c2_path.exists():
        raise FileNotFoundError(f"[ERROR] Could not find feature_manifest_6C2.json")
    if not features_dir or not features_dir.exists():
        raise FileNotFoundError(f"[ERROR] Features directory not found at {features_dir}")

    # Load Catalog & Build System Lookup Map
    catalog_sha = compute_sha256(catalog_path)
    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog_data = json.load(f)
    
    catalog_lookup = {
        s["system_id"]: s for s in catalog_data.get("systems", [])
    }

    # Load Phase 6C.2 Manifest for SHA Cross-Verification
    with open(manifest_6c2_path, "r", encoding="utf-8") as f:
        manifest_6c2_data = json.load(f)
    
    expected_shas = {
        item["system_id"]: item for item in manifest_6c2_data.get("features", [])
    }

    systems_index = []
    nc_files = sorted(features_dir.glob("*_features.nc"))

    print(f"[INFO] Catalog SHA-256: {catalog_sha[:12]}... | Processing {len(nc_files)} feature NetCDFs")

    for nc_file in nc_files:
        system_id = nc_file.stem.replace("_features", "")

        # 1. Authoritative Catalog Lookup
        if system_id not in catalog_lookup:
            raise KeyError(f"[CRITICAL ERROR] System {system_id} not found in catalog.json!")
        cat_rec = catalog_lookup[system_id]

        # 2. Cryptographic Integrity Audit
        actual_sha = compute_sha256(nc_file)
        if system_id in expected_shas:
            expected_sha = expected_shas[system_id].get("sha256")
            if expected_sha and actual_sha != expected_sha:
                raise ValueError(
                    f"[CRITICAL INTEGRITY FAILURE] System {system_id} hash mismatch!\n"
                    f" Expected: {expected_sha}\n Actual:   {actual_sha}"
                )

        # 3. Open NetCDF Tensor & Extract Core Standardized Metrics
        with xr.open_dataset(nc_file) as ds:
            # Standardized Vocabulary Extraction
            v_max = float(ds["wind_speed"].max().values) if "wind_speed" in ds else None
            vort_max = float(ds["vorticity"].max().values) if "vorticity" in ds else None
            div_max = float(ds["divergence"].max().values) if "divergence" in ds else None
            ow_min = float(ds["okubo_weiss"].min().values) if "okubo_weiss" in ds else None
            
            # Read MSL from dataset if present, else fallback to catalog/ERA5
            p_min = None
            if "msl" in ds:
                p_min = float(ds["msl"].min().values) / 100.0  # Pa -> hPa
            elif "min_pressure_hpa" in cat_rec:
                p_min = float(cat_rec["min_pressure_hpa"])

        raw_sha = cat_rec.get("raw_sha256") or cat_rec.get("sha256") or "UNKNOWN"

        systems_index.append({
            "system_id": system_id,
            "cohort_id": cat_rec.get("cohort_id", cat_rec.get("cohort", "UNKNOWN")),
            "cohort_name": cat_rec.get("cohort_name", "UNKNOWN"),
            "system_class": cat_rec.get("system_class", "UNKNOWN"),
            "feature_file": nc_file.name,
            "feature_sha256": actual_sha,
            "raw_artifact_sha256": raw_sha,
            "feature_version": "6C.2",
            "algorithm_version": "1.0.0",
            "quick_ref": {
                "wind_max_ms": round(v_max, 2) if v_max is not None else None,
                "pressure_min_hpa": round(p_min, 2) if p_min is not None else None,
                "max_vorticity_s1": float(f"{vort_max:.4e}") if vort_max is not None else None,
                "min_okubo_weiss_s2": float(f"{ow_min:.4e}") if ow_min is not None else None
            }
        })

    master_manifest = {
        "manifest_version": "6D.0",
        "feature_version": "6C.2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "system_count": len(systems_index),
        "generator": "build_manifest_6d0.py",
        "generator_version": "1.0.0",
        "catalog_sha256": catalog_sha,
        "systems": systems_index
    }

    out_path = catalog_path.parent / "feature_manifest.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(master_manifest, f, indent=4)

    print(f"[INFO] Phase 6D.0 Complete! Master manifest written to {out_path}")

if __name__ == "__main__":
    build_manifest_6d0()