# tracebind/core/region_of_interest.py
from enum import Enum
from functools import cached_property
from datetime import datetime, timezone
import hashlib
import numpy as np
from typing import Dict, Any, Tuple, Optional
from types import MappingProxyType


class ROIStrategy(Enum):
    BOUNDING_BOX = "bounding_box"
    POLYGON = "polygon"
    LABEL = "label"
    MASK = "mask"


class CoordinateTransform:
    """Interface for mapping pixel/raster grid coordinates to canonical spatial coordinates."""
    def to_canonical(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class AffineCoordinateTransform(CoordinateTransform):
    def __init__(self, transform_vector: Tuple[float, float, float, float, float, float]):
        if len(transform_vector) != 6:
            raise ValueError("Transform Error: Affine matrix must contain exactly 6 parameters.")
        self._vector = transform_vector

    def to_canonical(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        c, a, b, f, d, e = self._vector
        c1 = c + cols * a + rows * b
        c2 = f + cols * d + rows * e
        return np.column_stack((c1.ravel(), c2.ravel()))


class PointCloud:
    """
    Universal immutable spatial representation.
    Holds (N, D) coordinates and (N,) values decoupled from grid raster topologies.
    """
    def __init__(self, coordinates: np.ndarray, values: np.ndarray, metadata: Optional[Dict[str, Any]] = None):
        self._coordinates = np.asarray(coordinates, dtype=np.float64)
        self._values = np.asarray(values, dtype=np.float64)
        
        self._coordinates.flags.writeable = False
        self._values.flags.writeable = False
        self._metadata = MappingProxyType(metadata or {})

    @property
    def coordinates(self) -> np.ndarray: return self._coordinates

    @property
    def values(self) -> np.ndarray: return self._values

    @property
    def n_points(self) -> int: return self._coordinates.shape[0]

    @property
    def metadata(self) -> MappingProxyType: return self._metadata

    def __repr__(self) -> str:
        return f"PointCloud(n_points={self.n_points}, dim={self._coordinates.shape[1] if self.n_points > 0 else 0})"


class RegionOfInterest:
    """
    Immutable, provenance-tracked spatial subset of a parent RawStateMatrix.
    Preserves parent pixel offsets and bounding box for complete reconstruction.
    """
    def __init__(self,
                 parent_fingerprint: str,
                 roi_id: str,
                 values: np.ndarray,
                 mask: np.ndarray,
                 row_offset: int,
                 col_offset: int,
                 transform: CoordinateTransform,
                 coord_system: str,
                 strategy: ROIStrategy,
                 extraction_params: Dict[str, Any],
                 parent_metadata: Dict[str, Any]):
        
        self._parent_fingerprint = str(parent_fingerprint)
        self._roi_id = str(roi_id)
        
        self._values = np.asarray(values, dtype=np.float64).copy()
        self._values.flags.writeable = False
        
        self._mask = np.asarray(mask, dtype=bool).copy()
        self._mask.flags.writeable = False
        
        self._row_offset = int(row_offset)
        self._col_offset = int(col_offset)
        self._transform = transform
        self._coord_system = str(coord_system)
        self._strategy = strategy
        
        # Provenance Tracking
        parent_bbox = (
            self._row_offset, 
            self._row_offset + self._values.shape[0],
            self._col_offset, 
            self._col_offset + self._values.shape[1]
        )
        
        provenance = {
            "parent_fingerprint": self._parent_fingerprint,
            "roi_id": self._roi_id,
            "extraction_strategy": self._strategy.value,
            "extraction_params": extraction_params,
            "parent_bbox": parent_bbox,
            "row_offset": self._row_offset,
            "col_offset": self._col_offset,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "contract_version": "1.1.0",
            **parent_metadata
        }
        self._metadata = MappingProxyType(provenance)

        # Deterministic Identity Fingerprint
        fp_bytes = f"{self._parent_fingerprint}_{self._roi_id}_{parent_bbox}_{np.sum(self._mask)}".encode('utf-8')
        self._fingerprint = hashlib.sha256(fp_bytes).hexdigest()[:16]

    @property
    def parent_fingerprint(self) -> str: return self._parent_fingerprint
    @property
    def roi_id(self) -> str: return self._roi_id
    @property
    def fingerprint(self) -> str: return self._fingerprint
    @property
    def values(self) -> np.ndarray: return self._values
    @property
    def mask(self) -> np.ndarray: return self._mask
    @property
    def row_offset(self) -> int: return self._row_offset
    @property
    def col_offset(self) -> int: return self._col_offset
    @property
    def transform(self) -> CoordinateTransform: return self._transform
    @property
    def coord_system(self) -> str: return self._coord_system
    @property
    def strategy(self) -> ROIStrategy: return self._strategy
    @property
    def metadata(self) -> MappingProxyType: return self._metadata
    @property
    def height(self) -> int: return self._values.shape[0]
    @property
    def width(self) -> int: return self._values.shape[1]
    @property
    def n_valid(self) -> int: return int(np.sum(self._mask))

    @cached_property
    def point_cloud(self) -> PointCloud:
        """
        Thread-safe cached evaluation generating a universal PointCloud representation.
        Calculates absolute canonical coordinates leveraging parent pixel offsets.
        """
        rows, cols = np.meshgrid(
            np.arange(self._row_offset, self._row_offset + self.height),
            np.arange(self._col_offset, self._col_offset + self.width),
            indexing='ij'
        )
        
        flat_coords = self._transform.to_canonical(rows, cols)
        flat_vals = self._values.ravel()
        flat_mask = self._mask.ravel()

        return PointCloud(
            coordinates=flat_coords[flat_mask],
            values=flat_vals[flat_mask],
            metadata=dict(self._metadata)
        )

    def __repr__(self) -> str:
        return (f"RegionOfInterest(id='{self._roi_id}', strategy='{self._strategy.value}', "
                f"grid=[{self.height}x{self.width}], offset=({self._row_offset}, {self._col_offset}), "
                f"valid_points={self.n_valid}, fp='{self._fingerprint}')")