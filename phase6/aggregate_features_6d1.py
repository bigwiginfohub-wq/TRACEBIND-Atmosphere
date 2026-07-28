import json
from pathlib import Path
import xarray as xr
import pandas as pd
import numpy as np

VARIABLES = ["wind_speed", "vorticity", "divergence", "strain_normal", "strain_shear", "okubo_weiss"]
PERCENTILES = [5, 25, 50, 75, 95]

def aggregate_features_6d1():
    print("[INFO] Starting Phase 6D.1 Distributional Aggregation Engine...")
    base_dir = Path(__file__).resolve().parent
    manifest_path = base_dir / "feature_manifest.json"
    features_dir = base_dir / "data" / "features"
    out_dir = base_dir / "data" / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        raise FileNotFoundError(f"[ERROR] Could not find {manifest_path}. Run Phase 6D.0 first.")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    summary_rows = []

    for sys_entry in manifest["systems"]:
        sys_id = sys_entry["system_id"]
        nc_path = features_dir / sys_entry["feature_file"]

        row = {
            "system_id": sys_id,
            "cohort_id": sys_entry["cohort_id"],
            "cohort_name": sys_entry["cohort_name"],
            "system_class": sys_entry["system_class"],
            "feature_sha256": sys_entry["feature_sha256"]
        }

        with xr.open_dataset(nc_path) as ds:
            for var in VARIABLES:
                if var not in ds:
                    continue
                
                vals = ds[var].values.flatten()
                vals = vals[~np.isnan(vals)]

                if len(vals) == 0:
                    continue

                row[f"{var}_min"] = float(np.min(vals))
                row[f"{var}_p05"] = float(np.percentile(vals, 5))
                row[f"{var}_p25"] = float(np.percentile(vals, 25))
                row[f"{var}_median"] = float(np.median(vals))
                row[f"{var}_mean"] = float(np.mean(vals))
                row[f"{var}_p75"] = float(np.percentile(vals, 75))
                row[f"{var}_p95"] = float(np.percentile(vals, 95))
                row[f"{var}_max"] = float(np.max(vals))
                row[f"{var}_std"] = float(np.std(vals))

        summary_rows.append(row)

    df = pd.DataFrame(summary_rows)

    csv_path = out_dir / "feature_summary.csv"
    parquet_path = out_dir / "feature_summary.parquet"

    df.to_csv(csv_path, index=False)
    try:
        df.to_parquet(parquet_path, index=False)
        print(f"[INFO] Parquet summary written to {parquet_path}")
    except Exception as e:
        print(f"[WARNING] Could not write Parquet ({e}). CSV exported successfully.")

    print(f"[INFO] Phase 6D.1 Aggregation Complete! Exported {len(df)} system summaries to {csv_path}")

if __name__ == "__main__":
    aggregate_features_6d1()