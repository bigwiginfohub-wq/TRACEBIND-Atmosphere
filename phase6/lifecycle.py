"""
lifecycle.py - Decoupled Dual-State Machine, Event Logging, & Audit Engine
"""

from enum import Enum
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional


class HarvestState(str, Enum):
    DEFERRED = "deferred"
    PENDING = "pending"
    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
    QC_PASSED = "qc_passed"
    READY_FOR_ANALYSIS = "ready_for_analysis"
    FAILED_TRANSIENT = "failed_transient"  # Network timeouts, CDS HTTP 502/503
    FAILED_PERMANENT = "failed_permanent"  # Corrupt NetCDF, missing variables
    RETRY_PENDING = "retry_pending"


class AnalysisState(str, Enum):
    NOT_STARTED = "not_started"
    FEATURE_EXTRACTION = "feature_extraction"
    FEATURES_COMPLETE = "features_complete"
    SPECTRAL_ANALYSIS = "spectral_analysis"
    SPECTRAL_COMPLETE = "spectral_complete"
    VALIDATION = "validation"
    COMPLETE = "complete"
    FAILED = "failed"


VALID_HARVEST_TRANSITIONS = {
    HarvestState.DEFERRED: {HarvestState.PENDING},
    HarvestState.PENDING: {HarvestState.DOWNLOADING},
    HarvestState.DOWNLOADING: {HarvestState.DOWNLOADED, HarvestState.FAILED_TRANSIENT, HarvestState.FAILED_PERMANENT},
    HarvestState.FAILED_TRANSIENT: {HarvestState.RETRY_PENDING, HarvestState.PENDING},
    HarvestState.RETRY_PENDING: {HarvestState.DOWNLOADING},
    HarvestState.DOWNLOADED: {HarvestState.QC_PASSED, HarvestState.FAILED_PERMANENT},
    HarvestState.QC_PASSED: {HarvestState.READY_FOR_ANALYSIS},
    HarvestState.READY_FOR_ANALYSIS: set(), # Terminal state for acquisition phase
    HarvestState.FAILED_PERMANENT: set(),    # Terminal state for hard failures
}


def transition_harvest_state(
    record: Dict[str, Any],
    new_state: HarvestState,
    actor: str,
    reason: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes a valid harvest state transition, increments catalog_revision,
    and appends an entry to the immutable transition log.
    """
    status = record.setdefault("status", {})
    current_state_str = status.get("harvest_state", HarvestState.PENDING.value)
    current_state = HarvestState(current_state_str)

    # State transition guardrail
    if new_state not in VALID_HARVEST_TRANSITIONS.get(current_state, set()) and current_state != new_state:
        raise ValueError(
            f"[LIFECYCLE ERROR] Invalid transition from '{current_state.value}' to '{new_state.value}'"
        )

    now_utc = datetime.now(timezone.utc)
    now_iso = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Update state fields
    status["harvest_state"] = new_state.value
    if new_state == HarvestState.DEFERRED:
        status["deferred_reason"] = reason
    elif new_state == HarvestState.PENDING:
        status["deferred_reason"] = None

    # Optimistic locking revision bump
    record["catalog_revision"] = record.get("catalog_revision", 0) + 1

    # Append-only transition history
    history: List[Dict[str, Any]] = record.setdefault("transition_history", [])
    
    # Verify monotonic time ordering
    if history:
        last_timestamp_str = history[-1]["utc"]
        last_dt = datetime.strptime(last_timestamp_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        if now_utc < last_dt:
            raise ValueError(f"[AUDIT FAILURE] Timestamp regression detected! {now_iso} < {last_timestamp_str}")

    history.append({
        "phase": "harvest",
        "state": new_state.value,
        "utc": now_iso,
        "actor": actor,
        "revision": record["catalog_revision"],
        "reason": reason
    })

    return record