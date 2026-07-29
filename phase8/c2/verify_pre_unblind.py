import sys
import json
import hashlib
from pathlib import Path

def compute_sha256(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def verify_stage_gate(manifest_path: Path, snapshot_path: Path) -> bool:
    if not manifest_path.exists() or not snapshot_path.exists():
        print("FAIL: Missing manifest or snapshot file.")
        return False

    with open(snapshot_path, "r", encoding="utf-8") as f:
        snapshot = json.load(f)

    actual_manifest_hash = compute_sha256(manifest_path)

    # Access hash from snapshot dictionary layout
    artifacts = snapshot.get("artifacts_sha256", {})
    manifest_info = artifacts.get("c2_cohort_manifest_blinded.json", {})
    expected_hash = manifest_info.get("sha256")

    if expected_hash is None:
        print("FAIL: Could not find 'c2_cohort_manifest_blinded.json' entry inside 'artifacts_sha256'.")
        return False

    if expected_hash != actual_manifest_hash:
        print(f"FAIL: Hash mismatch!")
        print(f"  Snapshot expects : {expected_hash}")
        print(f"  Actual manifest  : {actual_manifest_hash}")
        return False

    # Check consistency flag
    consistency = snapshot.get("consistency_checks", {})
    if not consistency.get("manifest_consistency_passed", False):
        print("FAIL: manifest_consistency_passed is False in audit snapshot.")
        return False

    print("PASS")
    return True

if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).resolve().parent
    
    # Resolves paths whether run from root or inside phase8/c2
    if (SCRIPT_DIR / "manifest" / "c2_cohort_manifest_blinded.json").exists():
        MANIFEST = SCRIPT_DIR / "manifest" / "c2_cohort_manifest_blinded.json"
        SNAPSHOT = SCRIPT_DIR / "diagnostics" / "pre_unblinding_audit_snapshot.json"
    else:
        MANIFEST = Path("phase8/c2/manifest/c2_cohort_manifest_blinded.json")
        SNAPSHOT = Path("phase8/c2/diagnostics/pre_unblinding_audit_snapshot.json")

    success = verify_stage_gate(MANIFEST, SNAPSHOT)
    sys.exit(0 if success else 1)