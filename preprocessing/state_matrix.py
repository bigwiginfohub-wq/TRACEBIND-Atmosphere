# C:\TRACEBIND-Atmosphere\preprocessing\state_matrix.py
import numpy as np
import hashlib
import threading
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
from types import MappingProxyType
from abc import ABC, abstractmethod

class CoordinateTransform(ABC):
    """Abstract spatial transformer interface separating file projection physics from graph math."""
    @abstractmethod
    def to_canonical(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """Transforms 2D raster row/col arrays into an (N, 2) metric coordinate space."""
        pass


class AffineCoordinateTransform(CoordinateTransform):
    """Standard 6-parameter affine geometric mapping implementation."""
    def __init__(self, transform_vector: Tuple[float, float, float, float, float, float]):
        if len(transform_vector) != 6:
            raise ValueError("Transform Error: Affine matrix must contain exactly 6 parameters.")
        self._vector = transform_vector

    def to_canonical(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        c, a, b, f, d, e = self._vector
        c1 = c + cols * a + rows * b
        c2 = f + cols * d + rows * e
        return np.column_stack((c1.ravel(), c2.ravel()))


class ObservationView:
    """An immutable, flat matrix index container optimizing active data for graph pipelines."""
    def __init__(self, coordinates: np.ndarray, values: np.ndarray):
        self._coordinates = np.asarray(coordinates, dtype=np.float64)
        self._values = np.asarray(values, dtype=np.float64)
        
        self._coordinates.flags.writeable = False
        self._values.flags.writeable = False

    @property
    def coordinates(self) -> np.ndarray: return self._coordinates

    @property
    def values(self) -> np.ndarray: return self._values

    @property
    def n_points(self) -> int: return self._coordinates.shape[0]


class RawStateMatrix:
    """Immutable data container preserving native multidimensional topology and metadata bounds."""
    def __init__(self, 
                 timestamp: datetime, 
                 values: np.ndarray, 
                 mask: np.ndarray, 
                 transform: CoordinateTransform, 
                 coord_system: str, 
                 metadata: Dict[str, Any]):
        
        self._timestamp = timestamp
        self._values = np.asarray(values, dtype=np.float64).copy()
        self._values.flags.writeable = False
        
        self._mask = np.asarray(mask, dtype=bool).copy()
        self._mask.flags.writeable = False
        
        self._transform = transform
        self._coord_system = str(coord_system)
        self._metadata = MappingProxyType(dict(metadata))
        
        # Thread-safe synchronization properties
        self._cached_view: Optional[ObservationView] = None
        self._lock = threading.Lock()

        # Efficient identity fingerprint generation leveraging topology properties
        fingerprint_input = (
            f"{self._timestamp.isoformat()}_{self._values.shape}_{self._coord_system}_"
            f"{self._metadata.get('contract_version', '1.0.0')}"
        ).encode('utf-8')
        self._fingerprint = hashlib.sha256(fingerprint_input).hexdigest()[:16]

    @property
    def timestamp(self) -> datetime: return self._timestamp
    @property
    def values(self) -> np.ndarray: return self._values
    @property
    def mask(self) -> np.ndarray: return self._mask
    @property
    def transform(self) -> CoordinateTransform: return self._transform
    @property
    def coord_system(self) -> str: return self._coord_system
    @property
    def metadata(self) -> MappingProxyType: return self._metadata
    @property
    def fingerprint(self) -> str: return self._fingerprint
    @property
    def height(self) -> int: return self._values.shape[0]
    @property
    def width(self) -> int: return self._values.shape[1]

    def to_observation_view(self) -> ObservationView:
        """Thread-safe lazy evaluator generating flat canonical observation spaces."""
        with self._lock:
            if self._cached_view is not None:
                return self._cached_view

            rows, cols = np.meshgrid(np.arange(self.height), np.arange(self.width), indexing='ij')
            
            # Extract coordinates using mapped matrix mechanics
            flat_coords = self._transform.to_canonical(rows, cols)
            flat_vals = self._values.ravel()
            flat_mask = self._mask.ravel()

            self._cached_view = ObservationView(
                coordinates=flat_coords[flat_mask],
                values=flat_vals[flat_mask]
            )
            return self._cached_view

    def __repr__(self) -> str:
        return (f"RawStateMatrix(contract=v{self._metadata.get('contract_version', '1.0.0')}, "
                f"grid=[{self.height}x{self.width}], CRS='{self._coord_system}', "
                f"fp='{self._fingerprint}', time={self._timestamp.isoformat()})")


class RawStateMatrixBuilder:
    """Rigorous contract gatekeeper auditing input variables before freezing."""
    def __init__(self):
        self._timestamp = None
        self._values = None
        self._mask = None
        self._transform = None
        self._coord_system = None
        self._metadata = {}

    def setup_dimensions(self, timestamp: datetime, transform: CoordinateTransform, coord_system: str) -> 'RawStateMatrixBuilder':
        if not isinstance(timestamp, datetime):
            raise TypeError("Data Contract Violation: Timestamp must be a datetime object.")
        if not isinstance(transform, CoordinateTransform):
            raise TypeError("Data Contract Violation: Transform must implement the CoordinateTransform interface.")
        
        self._timestamp = timestamp
        self._transform = transform
        self._coord_system = str(coord_system)
        return self

    def inject_tensors(self, values: Any, mask: Any) -> 'RawStateMatrixBuilder':
        self._values = np.asarray(values, dtype=np.float64)
        self._mask = np.asarray(mask, dtype=bool)
        return self

    def inject_metadata(self, sensor_name: str, processing_level: str, **kwargs) -> 'RawStateMatrixBuilder':
        self._metadata = {
            "contract_version": "1.0.0",
            "sensor": str(sensor_name),
            "processing_level": str(processing_level),
            "missing_data_policy": "STRICT_EXCLUDE",
            **kwargs
        }
        return self

    def validate_and_freeze(self) -> RawStateMatrix:
        if any(v is None for v in [self._timestamp, self._values, self._mask, self._transform, self._coord_system]):
            raise ValueError("Validation Error: Cannot freeze an incomplete structural payload assembly.")

        if self._values.ndim != 2 or self._mask.ndim != 2:
            raise ValueError("Validation Error: Values and mask properties must be exact 2D arrays matching topology matrices.")

        if self._values.shape != self._mask.shape:
            raise ValueError(f"Validation Error: Structural shape mismatch. Values={self._values.shape}, Mask={self._mask.shape}.")

        if self._values.shape[0] <= 0 or self._values.shape[1] <= 0:
            raise ValueError("Validation Error: Matrix grids must possess positive dimensional coordinates.")

        if np.sum(self._mask) == 0:
            raise ValueError("Data Contract Violation: Target grid is fully masked; no active data samples exist.")

        # Non-finite values tracking inside isolated valid bounds
        if not np.all(np.isfinite(self._values[self._mask])):
            raise ValueError("Data Contract Violation: Active data vectors contain non-finite numeric states (NaN/Inf).")

        return RawStateMatrix(
            timestamp=self._timestamp,
            values=self._values,
            mask=self._mask,
            transform=self._transform,
            coord_system=self._coord_system,
            metadata=self._metadata
        )