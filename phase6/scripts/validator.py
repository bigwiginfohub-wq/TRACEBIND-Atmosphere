"""
Phase 6A: Strict NetCDF Grid, Coordinate, and Physical Property Validator
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np
import xarray as xr
from config import LAT_KEYS, LON_KEYS, TIME_KEYS, PHYSICAL_LIMITS

@dataclass
class QCResult:
    system_id: str
    passed: bool
    grid_ok: bool
    variables_ok: bool
    shape_ok: bool
    physical_range_ok: bool
    monotonic_ok: bool
    spacing_ok: bool
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "system_id": self.system_id,
            "passed": self.passed,
            "grid_ok": self.grid_ok,
            "variables_ok": self.variables_ok,
            "shape_ok": self.shape_ok,
            "physical_range_ok": self.physical_range_ok,
            "monotonic_ok": self.monotonic_ok,
            "spacing_ok": self.spacing_ok,
            "warnings_count": len(self.warnings),
            "errors": " | ".join(self.errors)
        }

class NetCDFValidator:
    def __init__(self, expected_shape: Optional[Tuple[int, int]] = (121, 121), grid_res: float = 0.25):
        self.expected_shape = expected_shape
        self.grid_res = grid_res

    def validate(self, nc_path: str, system_id: str) -> QCResult:
        qc = QCResult(
            system_id=system_id,
            passed=False,
            grid_ok=False,
            variables_ok=False,
            shape_ok=False,
            physical_range_ok=False,
            monotonic_ok=False,
            spacing_ok=False
        )

        try:
            ds = xr.open_dataset(nc_path)
        except Exception as e:
            qc.errors.append(f"Failed to open NetCDF file: {str(e)}")
            return qc

        # 1. Coordinate Name Resolution
        lat_var = next((k for k in ds.coords if k.lower() in LAT_KEYS), None)
        lon_var = next((k for k in ds.coords if k.lower() in LON_KEYS), None)
        time_var = next((k for k in ds.coords if k.lower() in TIME_KEYS), None)

        if not lat_var or not lon_var or not time_var:
            qc.errors.append(f"Missing essential coordinates. Found: {list(ds.coords.keys())}")
            ds.close()
            return qc

        qc.grid_ok = True

        # 2. Monotonicity & Spacing Check
        lats = ds[lat_var].values
        lons = ds[lon_var].values

        lat_diffs = np.diff(lats)
        lon_diffs = np.diff(lons)

        is_lat_mono = np.all(lat_diffs > 0) or np.all(lat_diffs < 0)
        is_lon_mono = np.all(lon_diffs > 0) or np.all(lon_diffs < 0)

        if is_lat_mono and is_lon_mono:
            qc.monotonic_ok = True
        else:
            qc.errors.append("Coordinate grid is not monotonic.")

        # Uniform Spacing Check
        lat_spacing_ok = np.allclose(np.abs(lat_diffs), self.grid_res, atol=1e-3)
        lon_spacing_ok = np.allclose(np.abs(lon_diffs), self.grid_res, atol=1e-3)

        if lat_spacing_ok and lon_spacing_ok:
            qc.spacing_ok = True
        else:
            qc.warnings.append(f"Grid spacing deviates from expected {self.grid_res} degrees.")

        # 3. Shape Verification
        spatial_shape = (len(lats), len(lons))
        if self.expected_shape is None or spatial_shape == self.expected_shape:
            qc.shape_ok = True
        else:
            qc.errors.append(f"Shape mismatch: Got {spatial_shape}, expected {self.expected_shape}")

        # 4. Variables & Physical Value Range Check
        data_vars = list(ds.data_vars.keys())
        if not data_vars:
            qc.errors.append("No data variables found in dataset.")
        else:
            qc.variables_ok = True

        has_nan_or_inf = False
        phys_ok = True

        for var_name in data_vars:
            arr = ds[var_name].values
            if np.isnan(arr).any() or np.isinf(arr).any():
                has_nan_or_inf = True
                qc.errors.append(f"NaN or Inf values detected in variable: {var_name}")

            # MSLP Unit Check & Sanity (Convert Pa to hPa if needed)
            if "msl" in var_name.lower() or "pressure" in var_name.lower():
                mean_val = np.mean(arr)
                if mean_val > 10000:  # Data in Pascals
                    arr_hpa = arr / 100.0
                else:
                    arr_hpa = arr

                min_p, max_p = np.min(arr_hpa), np.max(arr_hpa)
                if min_p < PHYSICAL_LIMITS["MSLP_HPA_MIN"] or max_p > PHYSICAL_LIMITS["MSLP_HPA_MAX"]:
                    phys_ok = False
                    qc.errors.append(f"MSLP value out of physical range: [{min_p:.1f}, {max_p:.1f}] hPa")

        if not has_nan_or_inf and phys_ok:
            qc.physical_range_ok = True

        # Overall Status Determination
        qc.passed = (
            qc.grid_ok and 
            qc.variables_ok and 
            qc.shape_ok and 
            qc.physical_range_ok and 
            qc.monotonic_ok
        )

        ds.close()
        return qc