import json
import pandas as pd
from pathlib import Path

def audit_unblinding_integrity():
    script_dir = Path(__file__).resolve().parent
    master_csv = script_dir / "unblinded_master_dataset.csv"
    keycard_json = script_dir / "manifest" / "keycard_c2_access_controlled.json"

    df = pd.read_csv(master_csv)
    with open(keycard_json, "r", encoding="utf-8") as f:
        keycard = json.load(f).get("unblind_keycard", {})

    print("================ C2 UNBLINDING AUDIT VERIFICATION ================\n")

    # 1. Check Total Sample Count & Balance
    total_cases = len(df)
    counts = df["condition"].value_counts().to_dict()
    print(f"[CHECK 1] Total Cases: {total_cases}")
    print(f"          Cohort Split: {counts}")
    
    balanced = counts.get("cyclone", 0) == 10 and counts.get("control", 0) == 10 and total_cases == 20
    print(f"          Status: {'PASS (10 Cyclone vs 10 Control)' if balanced else 'FAIL (Imbalanced split)'}\n")

    # 2. Line-by-Line Keycard Mapping Integrity Check
    print("[CHECK 2] Line-by-Line Keycard Cross-Reference:")
    mapping_errors = 0

    for idx, row in df.iterrows():
        blinded_id = row["blinded_id"]
        keycard_entry = keycard.get(blinded_id)

        if not keycard_entry:
            print(f"  ❌ MISMATCH: {blinded_id} not found in keycard!")
            mapping_errors += 1
            continue

        expected_case_id = keycard_entry.get("original_case_id")
        expected_type = keycard_entry.get("case_type", "").lower()
        actual_condition = str(row["condition"]).lower()
        actual_case_id = row.get("original_case_id")

        if expected_type != actual_condition or expected_case_id != actual_case_id:
            print(f"  ❌ MISMATCH for {blinded_id}: Keycard ({expected_case_id}, {expected_type}) != CSV ({actual_case_id}, {actual_condition})")
            mapping_errors += 1
        else:
            print(f"  ✓ {blinded_id} -> {actual_case_id:<20} | Type: {actual_condition.upper():<7} | C_phi: {row['c_phi']:.6f}")

    print(f"\n          Mapping Integrity Status: {'PASS (0 Mismatches)' if mapping_errors == 0 else f'FAIL ({mapping_errors} Mismatches Found)'}\n")

    # 3. Label Correctness & Case ID Inspection
    cyclones = df[df["condition"] == "cyclone"]["original_case_id"].tolist()
    controls = df[df["condition"] == "control"]["original_case_id"].tolist()

    print("[CHECK 3] Cohort Label Validation:")
    print("  Cyclone Cases Identified:")
    for c in cyclones:
        print(f"    - {c}")
    print("\n  Control Cases Identified:")
    for ctrl in controls:
        print(f"    - {ctrl}")

    all_passed = balanced and (mapping_errors == 0)
    print("\n================================================================")
    print(f"FINAL AUDIT VERIFICATION: {'PASSED - DATASET INTEGRITY CONFIRMED' if all_passed else 'FAILED'}")
    print("================================================================")

if __name__ == "__main__":
    audit_unblinding_integrity()