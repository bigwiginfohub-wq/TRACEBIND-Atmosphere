"""
Phase 6A: Batch Quality Control Runner & CSV Exporter
"""
import os
import json
import pandas as pd
from validator import NetCDFValidator
from schema import SystemMetadata

def run_batch_qc(catalog_json_path: str, data_dir: str, output_csv_path: str):
    with open(catalog_json_path, 'r') as f:
        catalog_raw = json.load(f)

    validator = NetCDFValidator()
    results = []

    for item in catalog_raw:
        meta = SystemMetadata.from_dict(item)
        nc_path = os.path.join(data_dir, meta.era5_filename)

        if not os.path.exists(nc_path):
            qc_res = {
                "system_id": meta.system_id,
                "passed": False,
                "errors": f"File not found: {meta.era5_filename}"
            }
        else:
            qc_obj = validator.validate(nc_path, meta.system_id)
            qc_res = qc_obj.to_dict()

        results.append(qc_res)

    df_qc = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df_qc.to_csv(output_csv_path, index=False)
    print(f"✅ QC Batch Completed. Report written to: {output_csv_path}")
    print(f"Pass Rate: {df_qc['passed'].sum()} / {len(df_qc)}")

if __name__ == "__main__":
    run_batch_qc("phase6/catalog.json", "phase6/data", "phase6/results/phase6_qc_report.csv")