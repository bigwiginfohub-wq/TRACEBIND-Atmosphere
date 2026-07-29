import json
import pandas as pd
from pathlib import Path

def generate_unblinded_master():
    script_dir = Path(__file__).resolve().parent
    
    results_csv = script_dir / "extraction" / "c2_cphi_results.csv"
    keycard_json = script_dir / "manifest" / "keycard_c2_access_controlled.json"
    output_csv = script_dir / "unblinded_master_dataset.csv"

    if not results_csv.exists() or not keycard_json.exists():
        raise FileNotFoundError("Missing c2_cphi_results.csv or keycard_c2_access_controlled.json")

    results_df = pd.read_csv(results_csv)

    with open(keycard_json, "r", encoding="utf-8") as f:
        keycard = json.load(f)

    # Extract the mapping dictionary
    unblind_map = keycard.get("unblind_keycard", {})

    # Determine UUID column
    id_col = "uuid" if "uuid" in results_df.columns else results_df.columns[0]

    # Extract case_type and lower-case standard label (cyclone vs control)
    results_df["condition"] = results_df[id_col].map(
        lambda u: unblind_map.get(u, {}).get("case_type", "").strip().lower()
    )
    
    # Also attach the metadata for auditability
    results_df["original_case_id"] = results_df[id_col].map(
        lambda u: unblind_map.get(u, {}).get("original_case_id", "")
    )
    results_df["basin"] = results_df[id_col].map(
        lambda u: unblind_map.get(u, {}).get("basin", "")
    )

    if results_df["condition"].eq("").any() or results_df["condition"].isnull().any():
        unmapped = results_df[results_df["condition"].isin(["", None])][id_col].tolist()
        raise ValueError(f"Unblinding failed! Unmapped UUIDs found: {unmapped}")

    # Output immutable master dataset
    results_df.to_csv(output_csv, index=False)
    print(f"SUCCESS: Immutable master dataset generated -> {output_csv.resolve()}")

if __name__ == "__main__":
    generate_unblinded_master()