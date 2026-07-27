# tracebind/core/roi_extractor.py
import numpy as np
import hashlib
from typing import Any, Dict, Optional
from tracebind.core.region_of_interest import RegionOfInterest, ROIStrategy


class ROIExtractor:
    """
    Single unified entry point for non-destructive spatial feature extraction.
    Does not modify, resample, or interpolate the parent observation grid.
    """

    @classmethod
    def extract(cls, 
                parent: Any, 
                strategy: ROIStrategy, 
                roi_id: Optional[str] = None, 
                **kwargs) -> RegionOfInterest:
        """
        Public Extraction Gateway.
        Dispatches execution based on the requested ROIStrategy.
        """
        if strategy == ROIStrategy.BOUNDING_BOX:
            return cls._extract_bounding_box(parent, roi_id=roi_id, **kwargs)
        elif strategy == ROIStrategy.LABEL:
            return cls._extract_label(parent, roi_id=roi_id, **kwargs)
        elif strategy == ROIStrategy.MASK:
            return cls._extract_mask(parent, roi_id=roi_id, **kwargs)
        else:
            raise NotImplementedError(f"Extraction strategy '{strategy}' is not yet supported.")

    @staticmethod
    def _extract_bounding_box(parent: Any, 
                              y_min: int, y_max: int, 
                              x_min: int, x_max: int, 
                              roi_id: Optional[str] = None) -> RegionOfInterest:
        if y_min < 0 or y_max > parent.height or x_min < 0 or x_max > parent.width:
            raise ValueError(f"Bounds [{y_min}:{y_max}, {x_min}:{x_max}] exceed parent grid [{parent.height}x{parent.width}].")
        if y_min >= y_max or x_min >= x_max:
            raise ValueError("Minimum slice indices must be strictly less than maximum indices.")

        sub_values = parent.values[y_min:y_max, x_min:x_max]
        sub_mask = parent.mask[y_min:y_max, x_min:x_max]

        if np.sum(sub_mask) == 0:
            raise ValueError("Requested Bounding Box contains zero valid observations.")

        if roi_id is None:
            param_hash = hashlib.sha256(f"{y_min}_{y_max}_{x_min}_{x_max}".encode('utf-8')).hexdigest()[:8]
            roi_id = f"ROI-BBOX-{param_hash}"

        return RegionOfInterest(
            parent_fingerprint=parent.fingerprint,
            roi_id=roi_id,
            values=sub_values,
            mask=sub_mask,
            row_offset=y_min,
            col_offset=x_min,
            transform=parent.transform,
            coord_system=parent.coord_system,
            strategy=ROIStrategy.BOUNDING_BOX,
            extraction_params={"y_min": y_min, "y_max": y_max, "x_min": x_min, "x_max": x_max},
            parent_metadata=dict(parent.metadata)
        )

    @staticmethod
    def _extract_label(parent: Any, 
                       label_matrix: np.ndarray, 
                       target_id: int, 
                       roi_id: Optional[str] = None) -> RegionOfInterest:
        if label_matrix.shape != parent.values.shape:
            raise ValueError(f"Label shape {label_matrix.shape} does not match parent shape {parent.values.shape}.")

        object_mask = (label_matrix == target_id) & parent.mask
        if np.sum(object_mask) == 0:
            raise ValueError(f"Label component ID={target_id} has no valid pixels in parent matrix.")

        rows, cols = np.where(object_mask)
        y_min, y_max = int(np.min(rows)), int(np.max(rows)) + 1
        x_min, x_max = int(np.min(cols)), int(np.max(cols)) + 1

        sub_values = parent.values[y_min:y_max, x_min:x_max]
        sub_mask = object_mask[y_min:y_max, x_min:x_max]

        if roi_id is None:
            roi_id = f"ROI-LABEL-OBJ{target_id}"

        return RegionOfInterest(
            parent_fingerprint=parent.fingerprint,
            roi_id=roi_id,
            values=sub_values,
            mask=sub_mask,
            row_offset=y_min,
            col_offset=x_min,
            transform=parent.transform,
            coord_system=parent.coord_system,
            strategy=ROIStrategy.LABEL,
            extraction_params={"target_id": target_id, "bbox_cropped": (y_min, y_max, x_min, x_max)},
            parent_metadata=dict(parent.metadata)
        )

    @staticmethod
    def _extract_mask(parent: Any, 
                     boolean_mask: np.ndarray, 
                     roi_id: Optional[str] = None) -> RegionOfInterest:
        if boolean_mask.shape != parent.values.shape:
            raise ValueError(f"Mask shape {boolean_mask.shape} does not match parent shape {parent.values.shape}.")

        combined_mask = boolean_mask.astype(bool) & parent.mask
        if np.sum(combined_mask) == 0:
            raise ValueError("Provided binary mask produces zero valid points within parent matrix bounds.")

        rows, cols = np.where(combined_mask)
        y_min, y_max = int(np.min(rows)), int(np.max(rows)) + 1
        x_min, x_max = int(np.min(cols)), int(np.max(cols)) + 1

        sub_values = parent.values[y_min:y_max, x_min:x_max]
        sub_mask = combined_mask[y_min:y_max, x_min:x_max]

        if roi_id is None:
            roi_id = f"ROI-MASK-{hashlib.sha256(boolean_mask.tobytes()).hexdigest()[:8]}"

        return RegionOfInterest(
            parent_fingerprint=parent.fingerprint,
            roi_id=roi_id,
            values=sub_values,
            mask=sub_mask,
            row_offset=y_min,
            col_offset=x_min,
            transform=parent.transform,
            coord_system=parent.coord_system,
            strategy=ROIStrategy.MASK,
            extraction_params={"bbox_cropped": (y_min, y_max, x_min, x_max)},
            parent_metadata=dict(parent.metadata)
        )