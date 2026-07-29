"""
TRACEBIND Phase 8 - Freeze Phase C0 v1.0 Release
=================================================
Creates an immutable release snapshot of the Phase C0 QC auditor,
report configurations, and parameter thresholds.
"""

import json
import shutil
import hashlib
from pathlib import Path

PILOT_DIR = Path(__file__).resolve().parents[1]
RELEASE_DIR = PILOT_DIR / "releases" / "phase_c0_v1.0"
RELEASE_DIR.mkdir(parents=True, exist_ok=True)

# 1. Copy Artifacts & Reports
qc_report_src = PILOT_DIR / "reports" / "phase_c0_qc_report.json"
if qc_report_src.exists():
    shutil.copy(qc_report_src, RELEASE_DIR / "phase_c0_qc_report.json")

mask_src_dir = PILOT_DIR / "artifacts" / "qc_masks"
if mask_src_dir.exists():
    mask_dst_dir = RELEASE_DIR / "example_masks"
    if mask_dst_dir.exists():
        shutil.rmtree(mask_dst_dir)
    shutil.copytree(mask_src_dir, mask_dst_dir)

# 2. Compute Auditor Implementation Hash
qc_script_path = PILOT_DIR / "src" / "qc_report.py"
hasher = hashlib.sha256()
with open(qc_script_path, "rb") as f:
    hasher.update(f.read())
auditor_sha256 = hasher.hexdigest()

# 3. Create center_detection.md
center_detection_md = """# TRACEBIND Phase C0: Center Detection Specification (v1.0)

## Overview
Phase C0 provides spatial alignment and quality control auditing prior to metric calculation.

## Algorithm Sequence
1. **Coordinate Alignment**: Verify descending/ascending latitudes and enforce standard ordering.
2. **Geodesic Spacing Calculation**: Calculate localized grid spacing $(\\Delta x, \\Delta y)$ using WGS84 geodesics evaluated at domain midpoints.
3. **Primary Center**: Identify local $P_{min}$ (MSLP Minimum).
4. **Local Search Mask**: Construct geodesic search radius ($r \\le \\text{max\\_center\\_sep\\_km}$) centered on $P_{min}$.
5. **Local Vorticity Center**: Find maximum relative vorticity $\\zeta$ strictly within candidate cells ($N > 0$).
6. **Stability Audit**: Measure geodesic distance separation between $P_{min}$ and local $\\zeta_{max}$. Reject systems with separation $> \\text{max\\_center\\_sep\\_km}$.
"""
with open(RELEASE_DIR / "center_detection.md", "w") as f:
    f.write(center_detection_md)

# 4. Create accepted_thresholds.md
accepted_thresholds_md = """# TRACEBIND Phase C0: Accepted Thresholds (v1.0)

| Parameter | Value | Unit | Description |
|---|---|---|---|
| `max_center_sep_km` | 100.0 | km | Maximum allowed separation between MSLP min and local vorticity max. |
| `missing_values` | 0 | cells | Maximum allowed null/NaN fields in primary data variables. |
| `min_candidate_cells` | 1 | cells | Minimum required grid points in local search radius mask. |
| `r_inner_km` | 30.0 | km | Inner radius bound for annular eyewall shell mask. |
| `r_outer_km` | 150.0 | km | Outer radius bound for annular eyewall shell mask. |
"""
with open(RELEASE_DIR / "accepted_thresholds.md", "w") as f:
    f.write(accepted_thresholds_md)

# 5. Create algorithm_version.json
version_meta = {
    "version": "v1.0",
    "phase": "Phase C0 Gate Auditor",
    "auditor_script": str(qc_script_path.relative_to(PILOT_DIR)),
    "auditor_sha256": auditor_sha256,
    "status": "FROZEN",
    "frozen_at_utc": "2026-07-28T17:45:00Z"
}
with open(RELEASE_DIR / "algorithm_version.json", "w") as f:
    json.dump(version_meta, f, indent=2)

print(f"[✓] Phase C0 v1.0 Frozen successfully -> {RELEASE_DIR}")