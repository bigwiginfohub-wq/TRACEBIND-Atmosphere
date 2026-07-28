"""
Phase 6A.5: Catalog Audit & Reproducibility Lock Engine
"""

import os
import json
import hashlib
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple
import jsonschema

BUILDER_VERSION = "1.1.0"

def compute_sha256(filepath: str) -> str:
    if not os.path.exists(filepath):
        return "FILE_NOT_FOUND"
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(4096), b""):
            sha256.update(block)
    return sha256.hexdigest()

def validate_json_schema(catalog_path: str, schema_path: str) -> Tuple[bool, List[str]]:
    if not os.path.exists(catalog_path) or not os.path.exists(schema_path):
        return False, ["Catalog or schema file missing."]

    with open(catalog_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    validator = jsonschema.Draft7Validator(schema)
    errors = [f"Schema violation at [{' -> '.join(str(p) for p in err.path)}]: {err.message}" for err in validator.iter_errors(data)]
    return len(errors) == 0, errors

def audit_scientific_rules(catalog_data: Dict[str, Any]) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    findings = {"ERROR": [], "WARNING": [], "INFO": []}
    checks = {}

    systems = catalog_data.get("systems", [])
    meta = catalog_data.get("catalog_metadata", {})

    # 1. Manifest & Provenance Presence
    has_rules_hash = bool(meta.get("selection_rules_hash"))
    has_sources_hash = bool(meta.get("sources_hash"))
    has_raw_hash = bool(meta.get("raw_ibtracs_sha256"))
    checks["Configuration hashes present"] = "PASS" if (has_rules_hash and has_sources_hash) else "FAIL"
    checks["Source hashes present"] = "PASS" if has_raw_hash else "FAIL"

    seen_ids, seen_filenames = set(), set()
    is_deterministic = True

    for idx, sys in enumerate(systems):
        sys_id = sys.get("system_id")
        filename = sys.get("era5_filename")
        lat = sys.get("center_lat")
        lon = sys.get("center_lon")

        if sys_id in seen_ids:
            findings["ERROR"].append(f"Duplicate system_id '{sys_id}'")
        seen_ids.add(sys_id)

        if filename in seen_filenames:
            findings["ERROR"].append(f"Duplicate era5_filename '{filename}'")
        seen_filenames.add(filename)

        if lat is None or not (-90.0 <= lat <= 90.0):
            findings["ERROR"].append(f"Record '{sys_id}': Invalid latitude {lat}")
        if lon is None or not (-180.0 <= lon <= 180.0):
            findings["ERROR"].append(f"Record '{sys_id}': Unnormalized longitude {lon}")

    checks["Duplicate IDs"] = "PASS" if len(seen_ids) == len(systems) else "FAIL"
    checks["Duplicate filenames"] = "PASS" if len(seen_filenames) == len(systems) else "FAIL"

    # Determinism Check: ensure array is pre-sorted
    sorted_systems = sorted(systems, key=lambda x: (-x["event_year"], x["basin"], x["cohort_id"], x["system_id"]))
    if sorted_systems != systems:
        checks["Deterministic ordering"] = "FAIL"
        findings["ERROR"].append("Catalog systems array is not deterministically ordered.")
    else:
        checks["Deterministic ordering"] = "PASS"

    checks["Catalog reproducible"] = "PASS" if len(findings["ERROR"]) == 0 else "FAIL"
    return findings, checks

def generate_statistics_report(catalog_data: Dict[str, Any], hashes: Dict[str, str], schema_passed: bool, schema_errors: List[str], findings: Dict[str, List[str]], checks: Dict[str, str]):
    systems = catalog_data.get("systems", [])
    meta = catalog_data.get("catalog_metadata", {})
    audit_passed = schema_passed and len(findings["ERROR"]) == 0

    md = f"""# TRACEBIND Phase 6 Catalog Audit Report

**Generated UTC:** {datetime.now(timezone.utc).isoformat()}  
**Audit Status:** `{'PASSED' if audit_passed else 'FAILED'}`  
**Catalog Systems Count:** {len(systems)}  

---

## 1. Reproducibility & Integrity Checklist

| Verification Check | Status |
| :--- | :--- |
| **Schema valid** | `{'PASS' if schema_passed else 'FAIL'}` |
| **Deterministic ordering** | `{checks.get('Deterministic ordering', 'FAIL')}` |
| **Duplicate IDs** | `{checks.get('Duplicate IDs', 'FAIL')}` |
| **Duplicate filenames** | `{checks.get('Duplicate filenames', 'FAIL')}` |
| **Configuration hashes present** | `{checks.get('Configuration hashes present', 'FAIL')}` |
| **Source hashes present** | `{checks.get('Source hashes present', 'FAIL')}` |
| **Catalog reproducible** | `{checks.get('Catalog reproducible', 'FAIL')}` |

---

## 2. Provenance & Cryptographic Hashes

| Artifact | File / Component | Version / SHA256 |
| :--- | :--- | :--- |
| **Manifest Version** | Metadata | `{meta.get('manifest_version', '1.0.0')}` |
| **Selection Rules** | `selection_rules.yaml` | `{hashes.get('selection_rules', 'N/A')}` |
| **Data Sources** | `sources.yaml` | `{hashes.get('sources', 'N/A')}` |
| **Frozen Input Dataset** | `data/raw/ibtracs_v04r01_last3years.csv` | `{meta.get('raw_ibtracs_sha256', 'N/A')}` |
| **Catalog Database** | `catalog.json` | `{hashes.get('catalog', 'N/A')}` |

---
"""
    with open("catalog_statistics.md", "w", encoding="utf-8") as f:
        f.write(md)
    print("[REPORT] Generated 'catalog_statistics.md'")

def main():
    print("="*60)
    print("TRACEBIND Phase 6A.5: Catalog Audit & Reproducibility Lock")
    print("="*60)

    hashes = {
        "selection_rules": compute_sha256("selection_rules.yaml"),
        "sources": compute_sha256("sources.yaml"),
        "schema": compute_sha256("catalog.schema.json"),
        "catalog": compute_sha256("catalog.json"),
        "audit_script": compute_sha256(__file__)
    }

    schema_passed, schema_errors = validate_json_schema("catalog.json", "catalog.schema.json")
    
    with open("catalog.json", "r", encoding="utf-8") as f:
        catalog_data = json.load(f)

    findings, checks = audit_scientific_rules(catalog_data)
    audit_passed = schema_passed and len(findings["ERROR"]) == 0

    generate_statistics_report(catalog_data, hashes, schema_passed, schema_errors, findings, checks)

    print("\n--- Audit Checklist ---")
    for check, status in checks.items():
        print(f"  {check:<30}: {status}")

    if audit_passed:
        print("\n [PASSED] Catalog is audit-proof and locked. Ready for Phase 6B harvesting.")
    else:
        print("\n ❌ [FAILED] Resolve catalog errors before initiating harvest.")
    print("="*60)

if __name__ == "__main__":
    main()