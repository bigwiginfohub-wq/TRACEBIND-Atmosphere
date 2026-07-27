"""
Phase 6A: One-Way Exporter from catalog.json to cohort_manifest.csv
"""
import json
import pandas as pd

def generate_manifest(catalog_path: str = "catalog.json", manifest_path: str = "cohort_manifest.csv"):
    with open(catalog_path, "r") as f:
        catalog = json.load(f)

    rows = []
    for item in catalog:
        rows.append({
            "system_id": item["system_id"],
            "system_name": item["system_name"],
            "system_class": item["system_class"],
            "basin": item["basin"],
            "event_year": item["event_year"],
            "analysis_time": item["analysis_time"],
            "downloaded": item["status"]["downloaded"],
            "qc_passed": item["status"]["qc_passed"],
            "processed": item["status"]["processed"]
        })

    df = pd.DataFrame(rows)
    df.to_csv(manifest_path, index=False)
    print(f"Generated {manifest_path} from {catalog_path} ({len(df)} entries).")

if __name__ == "__main__":
    generate_manifest()