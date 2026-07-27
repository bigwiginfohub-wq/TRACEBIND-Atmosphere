# tracebind/metrics/metric_result.py
import hashlib
from enum import Enum
from typing import Dict, Any, Optional, List
from types import MappingProxyType


class MetricStatus(Enum):
    SUCCESS = "SUCCESS"
    DEGENERATE_VARIANCE = "DEGENERATE_VARIANCE"
    INSUFFICIENT_NODES = "INSUFFICIENT_NODES"


class MetricResult:
    """
    Immutable container holding statistical measurement outputs,
    versioning, status codes, and cryptographically verified provenance tracking.
    """
    def __init__(self,
                 r_observed: float,
                 z_score: float,
                 p_value: float,
                 r_null_mean: float,
                 r_null_std: float,
                 metric_name: str,
                 metric_version: str,
                 status: MetricStatus,
                 collection_fingerprint: str,
                 null_fingerprint: str,
                 warnings: Optional[List[str]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        
        self._r_observed = float(r_observed)
        self._z_score = float(z_score)
        self._p_value = float(p_value)
        self._r_null_mean = float(r_null_mean)
        self._r_null_std = float(r_null_std)
        self._metric_name = str(metric_name)
        self._metric_version = str(metric_version)
        self._status = status
        self._collection_fp = str(collection_fingerprint)
        self._null_fp = str(null_fingerprint)
        self._warnings = tuple(warnings or [])

        # Provenance Tracking
        provenance = {
            "metric_name": self._metric_name,
            "metric_version": self._metric_version,
            "status": self._status.value,
            "warnings": list(self._warnings),
            "collection_fingerprint": self._collection_fp,
            "null_fingerprint": self._null_fp,
            "contract_version": "1.1.0",
            **(metadata or {})
        }
        self._metadata = MappingProxyType(provenance)

        # Robust Identity Fingerprint
        fp_bytes = f"{self._collection_fp}_{self._null_fp}_{self._metric_name}_{self._metric_version}_{self._status.value}".encode('utf-8')
        self._fingerprint = hashlib.sha256(fp_bytes).hexdigest()[:16]

    @property
    def fingerprint(self) -> str: return self._fingerprint
    @property
    def r_observed(self) -> float: return self._r_observed
    @property
    def z_score(self) -> float: return self._z_score
    @property
    def p_value(self) -> float: return self._p_value
    @property
    def r_null_mean(self) -> float: return self._r_null_mean
    @property
    def r_null_std(self) -> float: return self._r_null_std
    @property
    def status(self) -> MetricStatus: return self._status
    @property
    def warnings(self) -> Tuple[str, ...]: return self._warnings
    @property
    def metadata(self) -> MappingProxyType: return self._metadata

    def __repr__(self) -> str:
        return (f"MetricResult(metric='{self._metric_name}', status='{self._status.value}', "
                f"R_obs={self._r_observed:.4f}, Z={self._z_score:.2f}, p={self._p_value:.4f}, fp='{self._fingerprint}')")