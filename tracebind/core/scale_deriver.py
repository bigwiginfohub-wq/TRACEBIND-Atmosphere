# tracebind/core/scale_deriver.py
import numpy as np
import hashlib
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional
from tracebind.core.region_of_interest import RegionOfInterest, CoordinateTransform


class ScaledCoordinateTransform(CoordinateTransform):
    """Wraps a parent CoordinateTransform to scale normalized grid indices back to canonical coordinates."""
    def __init__(self, parent_transform: CoordinateTransform, row_factor: float, col_factor: float):
        self._parent_transform = parent_transform
        self._row_factor = float(row_factor)
        self._col_factor = float(col_factor)

    def to_canonical(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        # Scale coarse grid indices back to parent raster coordinates
        parent_rows = rows * self._row_factor
        parent_cols = cols * self._col_factor
        return self._parent_transform.to_canonical(parent_rows, parent_cols)


class PoolingStrategy(ABC):
    """Abstract strategy interface for spatial grid aggregation."""
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def pool(self, vals_4d: np.ndarray, mask_4d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pools a 4D array of shape (out_h, block_h, out_w, block_w).
        Returns (new_values, new_mask).
        """
        pass


class MeanPooling(PoolingStrategy):
    @property
    def name(self) -> str: return "mean"

    def pool(self, vals_4d: np.ndarray, mask_4d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        out_h, _, out_w, _ = vals_4d.shape
        valid_counts = np.sum(mask_4d, axis=(1, 3))
        new_mask = valid_counts > 0

        vals_masked = np.where(mask_4d, vals_4d, 0.0)
        sum_vals = np.sum(vals_masked, axis=(1, 3))
        
        new_values = np.zeros((out_h, out_w), dtype=np.float64)
        np.divide(sum_vals, valid_counts, out=new_values, where=new_mask)
        return new_values, new_mask


class MedianPooling(PoolingStrategy):
    @property
    def name(self) -> str: return "median"

    def pool(self, vals_4d: np.ndarray, mask_4d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # TODO: Replace nested iteration with vectorized strided kernel if performance demands
        out_h, _, out_w, _ = vals_4d.shape
        valid_counts = np.sum(mask_4d, axis=(1, 3))
        new_mask = valid_counts > 0
        new_values = np.zeros((out_h, out_w), dtype=np.float64)

        for r in range(out_h):
            for c in range(out_w):
                if new_mask[r, c]:
                    block_vals = vals_4d[r, :, c, :][mask_4d[r, :, c, :]]
                    new_values[r, c] = np.median(block_vals)
        return new_values, new_mask


class MaxPooling(PoolingStrategy):
    @property
    def name(self) -> str: return "max"

    def pool(self, vals_4d: np.ndarray, mask_4d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        out_h, _, out_w, _ = vals_4d.shape
        valid_counts = np.sum(mask_4d, axis=(1, 3))
        new_mask = valid_counts > 0
        new_values = np.zeros((out_h, out_w), dtype=np.float64)

        vals_masked = np.where(mask_4d, vals_4d, -np.inf)
        max_vals = np.max(vals_masked, axis=(1, 3))
        new_values = np.where(new_mask, max_vals, 0.0)
        return new_values, new_mask


class ScaleDeriver:
    """
    Executes non-destructive resolution transformations on RegionOfInterest entities.
    Preserves absolute parent offsets while tracking scale levels and aggregation statistics.
    """

    @classmethod
    def derive(cls, 
               roi: RegionOfInterest, 
               target_shape: Tuple[int, int], 
               strategy: Optional[PoolingStrategy] = None) -> RegionOfInterest:
        """
        Derives a scaled representation of the target RegionOfInterest.

        Args:
            roi: The source RegionOfInterest.
            target_shape: Desired output grid dimensions (out_h, out_w).
            strategy: Strategy instance implementing PoolingStrategy (defaults to MeanPooling).
        """
        if strategy is None:
            strategy = MeanPooling()

        out_h, out_w = target_shape
        h, w = roi.height, roi.width

        if out_h <= 0 or out_w <= 0:
            raise ValueError(f"Scale Error: Target dimensions [{out_h}x{out_w}] must be positive integers.")
        
        if out_h > h or out_w > w:
            raise ValueError(f"Scale Error: Upsampling to target [{out_h}x{out_w}] from [{h}x{w}] is not supported.")

        # Identity return optimization
        if (out_h, out_w) == (h, w):
            return roi

        # Calculate exact block scaling factors
        block_h = h // out_h
        block_w = w // out_w

        if block_h < 1 or block_w < 1:
            raise ValueError(f"Scale Error: Target shape {target_shape} incompatible with source shape ({h}, {w}).")

        crop_h = out_h * block_h
        crop_w = out_w * block_w

        # Quantify discarded perimeter pixels due to block truncation
        discarded_pixels = (h * w) - (crop_h * crop_w)

        vals_cropped = roi.values[:crop_h, :crop_w]
        mask_cropped = roi.mask[:crop_h, :crop_w]

        # Reshape into 4D tensor for block aggregation: (out_h, block_h, out_w, block_w)
        vals_4d = vals_cropped.reshape(out_h, block_h, out_w, block_w)
        mask_4d = mask_cropped.reshape(out_h, block_h, out_w, block_w)

        # Delegate pooling calculation to Strategy
        new_values, new_mask = strategy.pool(vals_4d, mask_4d)

        # Scientific Metadata Audit
        total_valid_source = np.sum(roi.mask)
        total_valid_target = np.sum(new_mask)
        valid_fraction = float(total_valid_target / (out_h * out_w)) if (out_h * out_w) > 0 else 0.0

        current_level = roi.metadata.get("scale_level", 0) + 1
        row_factor = float(h / out_h)
        col_factor = float(w / out_w)

        scaled_transform = ScaledCoordinateTransform(roi.transform, row_factor, col_factor)

        # Unique Scale Fingerprint incorporating method and target resolution
        fp_bytes = f"{roi.fingerprint}_{strategy.name}_{out_h}x{out_w}_{current_level}".encode('utf-8')
        scale_id = f"{roi.roi_id}-S{current_level}-{strategy.name[:3]}"

        scale_params = {
            "source_roi_id": roi.roi_id,
            "source_roi_fingerprint": roi.fingerprint,
            "scale_level": current_level,
            "pooling_method": strategy.name,
            "target_shape": (out_h, out_w),
            "row_factor": row_factor,
            "col_factor": col_factor,
            "valid_fraction": valid_fraction,
            "discarded_pixels": discarded_pixels,
            "source_valid_points": int(total_valid_source),
            "scaled_valid_points": int(total_valid_target)
        }

        return RegionOfInterest(
            parent_fingerprint=roi.parent_fingerprint,
            roi_id=scale_id,
            values=new_values,
            mask=new_mask,
            row_offset=roi.row_offset, # Provenance invariant: preserved relative to original parent
            col_offset=roi.col_offset, # Provenance invariant: preserved relative to original parent
            transform=scaled_transform,
            coord_system=roi.coord_system,
            strategy=roi.strategy,
            extraction_params=scale_params,
            parent_metadata=dict(roi.metadata)
        )