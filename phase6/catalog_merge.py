"""
Phase 6A.5: Catalog Merger & Manifest Stamping
"""

import json
import hashlib
from datetime import datetime, timezone

BUILDER_VERSION = "1.1.0"
MANIFEST_VERSION = "1.0.0"

def compute_sha256(filepath: str) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(4096), b""):
            sha256.update(block)
    return sha256.hexdigest()

def merge_and_stamp():
    with open("temp_tc_catalog.json", "r", encoding="utf-8") as f:
        tc_systems = json.load(f)
    with open("temp_random_catalog.json", "r", encoding="utf-8") as f:
        rand_systems = json.load(f)

    all_systems = tc_systems + rand_systems

    # Strict Deterministic Sort: Year (desc), Basin (asc), Cohort (asc), System ID (asc)
    all_systems.sort(key=lambda x: (-x["event_year"], x["basin"], x["cohort_id"], x["system_id"]))

    rules_hash = compute_sha256("selection_rules.yaml")
    sources_hash = compute_sha256("sources.yaml")
    raw_ibtracs_hash = compute_sha256("data/raw/ibtracs_v04r01_last3years.csv")

    catalog_data = {
        "catalog_metadata": {
            "catalog_version": "1.0",
            "manifest_version": MANIFEST_VERSION,
            "builder_version": BUILDER_VERSION,
            "selection_rules_hash": rules_hash,
            "sources_hash": sources_hash,
            "raw_ibtracs_sha256": raw_ibtracs_hash,
            "generated_utc": "DETERMINISTIC_LOCK" # Frozen for hash verification comparisons
        },
        "systems": all_systems
    }

    with open("catalog.json", "w", encoding="utf-8") as f:
        json.dump(catalog_data, f, indent=2)

    print(f"[CATALOG MERGE] Wrote {len(all_systems)} systems to 'catalog.json'.")

if __name__ == "__main__":
    merge_and_stamp()