"""
16_generate_cohort_manifests.py
--------------------------------
TRACEBIND Phase 5 Cohort Provenance Generator
Scans the data directory, computes NetCDF checksums, and outputs 
standalone JSON manifests alongside a combined cohort manifest log.
"""

from pathlib import Path
import json
import pandas as pd
from tracebind_manifest import build_reproducibility_manifest

# Directory Setup
BASE_DIR = Path(r"C:\TRACEBIND-Atmosphere\phase5")
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
MANIFEST_DIR = RESULTS_DIR / "manifests"

def run_cohort_manifest_generator():
    print("=========================================================================================")
    print("         TRACEBIND PHASE 5: COHORT PROVENANCE & MANIFEST GENERATOR                       ")
    print("=========================================================================================\n")

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    
    # Locate all ERA5 storm NetCDF files
    nc_files = sorted(list(DATA_DIR.glob("era5_*.nc")))
    
    if not nc_files:
        print(f"[-] No NetCDF files matching 'era5_*.nc' found in: {DATA_DIR}")
        return

    print(f"[+] Found {len(nc_files)} storm datasets in {DATA_DIR}\n")

    cohort_records = []

    for nc_path in nc_files:
        # Extract storm name (e.g., era5_amphan_72h.nc -> Amphan)
        parts = nc_path.stem.split("_")
        storm_name = parts[1].capitalize() if len(parts) > 1 else nc_path.stem

        print(f"[*] Processing Manifest for: {storm_name} ({nc_path.name})")

        # Build individual manifest
        manifest = build_reproducibility_manifest(
            storm_name=storm_name,
            nc_filepath=nc_path,
            algorithm_id="TRACEBIND-P5.0-FROZEN",
            n_permutations=1000,
            seed=42
        )

        # Save individual JSON manifest
        out_json = MANIFEST_DIR / f"manifest_{storm_name.lower()}.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=4)

        # Log entry for summary table
        cohort_records.append({
            "Storm": storm_name,
            "Filename": manifest["dataset"]["input_file"],
            "MD5_Checksum": manifest["dataset"]["input_md5"],
            "Time_Steps": manifest["dataset"]["dimensions"].get("time_steps", "N/A"),
            "Variable": manifest["dataset"]["dimensions"].get("var_name", "msl"),
            "Algorithm_ID": manifest["metadata"]["algorithm_id"],
            "Timestamp_UTC": manifest["metadata"]["timestamp_utc"]
        })

    # Export master manifest summary CSV
    summary_df = pd.DataFrame(cohort_records)
    summary_csv = RESULTS_DIR / "phase5_cohort_data_provenance.csv"
    summary_df.to_csv(summary_csv, index=False)

    print("\n--- Cohort Data Provenance Summary ---")
    print(summary_df[["Storm", "MD5_Checksum", "Time_Steps", "Algorithm_ID"]].to_string(index=False))
    print(f"\n[+] Master Cohort Manifest saved to: {summary_csv}")
    print(f"[+] Individual JSON manifests written to: {MANIFEST_DIR}\n")

if __name__ == "__main__":
    run_cohort_manifest_generator()