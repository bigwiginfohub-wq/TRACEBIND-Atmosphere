#!/usr/bin/env python3
"""
TRACEBIND Phase 7B - Layered ERA5 Observational Validator
=========================================================
File: phase7/validation/validate_observation.py
"""

import sys
import json
import numpy as np
import xarray as xr
from pathlib import Path
import matplotlib.pyplot as plt

VALIDATION_DIR = Path(__file__).resolve().parent
SANDBOX_DIR = VALIDATION_DIR.parent / "sandbox"
if str(SANDBOX_DIR) not in sys.path:
    sys.path.insert(0, str(SANDBOX_DIR))


class ERA5ObservationValidator:
    """3-Layer Observational Validator enforcing audit gate progression."""

    def __init__(self, nc_path: Path, output_dir: Path, storm_id: str):
        self.nc_path = Path(nc_path)
        self.output_dir = Path(output_dir) / storm_id
        self.storm_id = storm_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fig_dir = self.output_dir / "figures"
        self.fig_dir.mkdir(exist_ok=True)
        
        self.ds = None
        self.audit = {
            "storm_id": storm_id,
            "layer0_dataset_integrity": {},
            "layer1_physical_validation": {},
            "layer2_tracebind_metrics": {},
            "overall_status": "PENDING"
        }

    def execute_validation_pipeline(self) -> bool:
        """Executes Layer 0, Layer 1, and Layer 2 in strict sequence."""
        print(f"\n==================================================")
        print(f" EXECUTING VALIDATION PIPELINE: {self.storm_id}")
        print(f"==================================================")

        try:
            # LAYER 0: DATASET INTEGRITY
            self._layer_0_dataset_integrity()
            
            # LAYER 1: PHYSICAL VALIDATION
            self._layer_1_physical_validation()

            # LAYER 2: TRACEBIND METRICS
            self._layer_2_tracebind_metrics()

            self.audit["overall_status"] = "PASSED"
            self._write_audit_reports()
            print(f"[✓] SUCCESS: {self.storm_id} PASSED all validation layers.")
            return True

        except Exception as err:
            self.audit["overall_status"] = "FAILED"
            self.audit["failure_reason"] = str(err)
            self._write_audit_reports()
            print(f"[❌] FAILURE: {self.storm_id} rejected at validation stage: {err}")
            return False

    # -------------------------------------------------------------------------
    # LAYER 0: DATASET INTEGRITY
    # -------------------------------------------------------------------------
    def _layer_0_dataset_integrity(self):
        print("[Layer 0] Checking Dataset Integrity...")
        if not self.nc_path.exists():
            raise FileNotFoundError(f"File not found: {self.nc_path}")

        self.ds = xr.open_dataset(self.nc_path)

        # 0.1 Variable & Coordinate Presence
        for var in ["u", "v"]:
            if var not in self.ds:
                raise KeyError(f"Missing required wind component: '{var}'")
        for coord in ["latitude", "longitude"]:
            if coord not in self.ds:
                raise KeyError(f"Missing spatial coordinate: '{coord}'")

        # 0.2 Grid Uniformity & Step Delta Checks
        lats = self.ds.latitude.values
        lons = self.ds.longitude.values
        
        d_lats = np.abs(np.diff(lats))
        d_lons = np.abs(np.diff(lons))

        lat_step_var = np.max(d_lats) - np.min(d_lats)
        lon_step_var = np.max(d_lons) - np.min(d_lons)

        if lat_step_var > 1e-4 or lon_step_var > 1e-4:
            raise ValueError(f"Irregular grid detected! Lat delta var: {lat_step_var:.2e}, Lon delta var: {lon_step_var:.2e}")

        # 0.3 Coordinate Conventions & Radius Logging
        lon_convention = "0_to_360" if np.max(lons) > 180.0 else "-180_to_180"
        lat_ordering = "descending" if lats[0] > lats[-1] else "ascending"

        self.audit["layer0_dataset_integrity"] = {
            "status": "PASS",
            "dimensions": {k: int(v) for k, v in self.ds.dims.items()},
            "grid_resolution_deg": float(np.mean(d_lats)),
            "lon_convention": lon_convention,
            "lat_ordering": lat_ordering,
            "earth_radius_m": 6371000.0,
            "projection": "Equirectangular / WGS84"
        }

    # -------------------------------------------------------------------------
    # LAYER 1: PHYSICAL VALIDATION
    # -------------------------------------------------------------------------
    def _layer_1_physical_validation(self):
        print("[Layer 1] Checking Physical Bounds & Mask Topology...")
        u_raw = self.ds.u.values
        v_raw = self.ds.v.values

        # 1.1 Metadata Unit Attributes
        u_unit = self.ds.u.attrs.get("units", "unspecified")
        v_unit = self.ds.v.attrs.get("units", "unspecified")
        if u_unit not in ["m s**-1", "m s-1", "m/s"] or v_unit not in ["m s**-1", "m s-1", "m/s"]:
            raise ValueError(f"Unrecognized wind unit attributes: u='{u_unit}', v='{v_unit}'")

        # 1.2 Speed Bounds
        speed = np.sqrt(u_raw**2 + v_raw**2)
        max_speed = float(np.nanmax(speed))
        if max_speed > 120.0 or max_speed < 0.0:
            raise ValueError(f"Unphysical wind magnitude: {max_speed:.1f} m/s")

        # 1.3 Missing Value Topology (NaN Analysis)
        total_pixels = u_raw.size
        nan_pixels = int(np.isnan(u_raw).sum())
        nan_pct = (nan_pixels / total_pixels) * 100.0

        # Boundary vs. Interior NaNs
        edge_mask = np.ones_like(u_raw, dtype=bool)
        edge_mask[1:-1, 1:-1] = False
        edge_nans = int(np.isnan(u_raw)[edge_mask].sum())
        interior_nans = nan_pixels - edge_nans

        self.audit["layer1_physical_validation"] = {
            "status": "PASS",
            "u_unit": u_unit,
            "v_unit": v_unit,
            "max_wind_speed_ms": max_speed,
            "missing_value_topology": {
                "total_nans": nan_pixels,
                "nan_percentage": float(nan_pct),
                "edge_nans": edge_nans,
                "interior_nans": interior_nans
            }
        }

    # -------------------------------------------------------------------------
    # LAYER 2: TRACEBIND METRICS & MULTI-CENTER TRACKING
    # -------------------------------------------------------------------------
    def _layer_2_tracebind_metrics(self):
        print("[Layer 2] Evaluating Multi-Center Candidate Ensemble & Coherence...")
        u = np.nan_to_num(self.ds.u.values)
        v = np.nan_to_num(self.ds.v.values)
        lats = self.ds.latitude.values
        lons = self.ds.longitude.values

        speed = np.sqrt(u**2 + v**2)

        # Spatial Derivatives for Vorticity
        dy_m = np.abs(lats[1] - lats[0]) * 111000.0
        dx_m = np.abs(lons[1] - lons[0]) * 111000.0 * np.cos(np.radians(np.mean(lats)))

        du_dy = np.gradient(u, dy_m, axis=0)
        dv_dx = np.gradient(v, dx_m, axis=1)
        vorticity = dv_dx - du_dy

        # 2.1 Multi-Candidate Center Ensemble
        # Center A: Wind Speed Minimum
        idx_min_speed = np.unravel_index(np.argmin(speed), speed.shape)
        center_min_speed = (float(lats[idx_min_speed[0]]), float(lons[idx_min_speed[1]]))

        # Center B: Vorticity Maximum
        idx_max_vort = np.unravel_index(np.argmax(vorticity), vorticity.shape)
        center_max_vort = (float(lats[idx_max_vort[0]]), float(lons[idx_max_vort[1]]))

        # Center C: Vorticity Top-5% Centroid
        vort_95th = np.percentile(vorticity, 95)
        top_vort_mask = vorticity >= vort_95th
        grid_lons, grid_lats = np.meshgrid(lons, lats)
        center_vort_centroid = (
            float(np.mean(grid_lats[top_vort_mask])),
            float(np.mean(grid_lons[top_vort_mask]))
        )

        # Operational Center Assignment (Vorticity Centroid)
        op_lat, op_lon = center_vort_centroid

        # 2.2 Tangential & Radial Decomposition
        dx_grid = (grid_lons - op_lon) * 111000.0 * np.cos(np.radians(op_lat))
        dy_grid = (grid_lats - op_lat) * 111000.0
        R = np.sqrt(dx_grid**2 + dy_grid**2) + 1e-12

        # Unit vectors
        tx, ty = -dy_grid / R, dx_grid / R
        rx, ry = dx_grid / R, dy_grid / R

        u_tangential = u * tx + v * ty
        u_radial = u * rx + v * ry

        mean_tangential = float(np.mean(u_tangential))
        mean_radial = float(np.mean(u_radial))

        # Coherence (C_phi)
        u_speed = speed + 1e-12
        proj_tangential = np.abs((u / u_speed) * tx + (v / u_speed) * ty)
        c_phi = float(np.mean(proj_tangential))

        if not (0.0 <= c_phi <= 1.0):
            raise ValueError(f"Coherence metric out of bounds [0, 1]: C_phi = {c_phi}")

        self.audit["layer2_tracebind_metrics"] = {
            "status": "PASS",
            "candidate_centers": {
                "min_wind_speed": {"lat": center_min_speed[0], "lon": center_min_speed[1]},
                "max_vorticity": {"lat": center_max_vort[0], "lon": center_max_vort[1]},
                "vorticity_top5pct_centroid": {"lat": center_vort_centroid[0], "lon": center_vort_centroid[1]}
            },
            "operational_center": {"lat": op_lat, "lon": op_lon},
            "kinematic_components": {
                "mean_tangential_velocity_ms": mean_tangential,
                "mean_radial_velocity_ms": mean_radial,
                "phase_coherence_c_phi": c_phi
            }
        }

        # 2.3 Comprehensive 6-Panel Diagnostic Figure
        self._generate_diagnostic_figure(speed, u, v, vorticity, proj_tangential, op_lat, op_lon)

    def _generate_diagnostic_figure(self, speed, u, v, vorticity, coherence_map, op_lat, op_lon):
        """Generates a 6-panel diagnostic figure for instant visual audit."""
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        
        # Panel 1: Wind Speed
        im0 = axes[0, 0].imshow(speed, cmap='viridis', origin='upper')
        fig.colorbar(im0, ax=axes[0, 0], label='m/s')
        axes[0, 0].set_title(f'{self.storm_id} - Wind Speed')

        # Panel 2: Zonal Wind (U)
        im1 = axes[0, 1].imshow(u, cmap='RdBu_r', origin='upper')
        fig.colorbar(im1, ax=axes[0, 1], label='m/s')
        axes[0, 1].set_title('U Component')

        # Panel 3: Meridional Wind (V)
        im2 = axes[0, 2].imshow(v, cmap='RdBu_r', origin='upper')
        fig.colorbar(im2, ax=axes[0, 2], label='m/s')
        axes[0, 2].set_title('V Component')

        # Panel 4: Relative Vorticity
        im3 = axes[1, 0].imshow(vorticity, cmap='magma', origin='upper')
        fig.colorbar(im3, ax=axes[1, 0], label='s^-1')
        axes[1, 0].set_title('Relative Vorticity (ζ)')

        # Panel 5: Local Coherence Map
        im4 = axes[1, 1].imshow(coherence_map, cmap='plasma', origin='upper', vmin=0, vmax=1)
        fig.colorbar(im4, ax=axes[1, 1], label='C_phi')
        axes[1, 1].set_title('Local Coherence Map')

        # Panel 6: Center Candidates Map
        axes[1, 2].imshow(speed, cmap='gray', origin='upper', alpha=0.5)
        centers = self.audit["layer2_tracebind_metrics"]["candidate_centers"]
        axes[1, 2].plot(centers["min_wind_speed"]["lon"], centers["min_wind_speed"]["lat"], 'blue', marker='o', label='Min Wind')
        axes[1, 2].plot(centers["max_vorticity"]["lon"], centers["max_vorticity"]["lat"], 'red', marker='x', label='Max Vort')
        axes[1, 2].plot(op_lon, op_lat, 'lime', marker='*', markersize=12, label='Op Centroid')
        axes[1, 2].legend(loc='upper right', fontsize=8)
        axes[1, 2].set_title('Center Candidate Ensemble')

        plt.tight_layout()
        fig_path = self.fig_dir / "full_diagnostic_panel.png"
        plt.savefig(fig_path, dpi=300)
        plt.close()

    def _write_audit_reports(self):
        """Outputs metadata.json and validation_report.md."""
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self.audit, f, indent=2)

        report_path = self.output_dir / "validation_report.md"
        with open(report_path, "w") as f:
            f.write(f"# Phase 7B Validation Report: {self.storm_id}\n\n")
            f.write(f"* **Overall Audit Status:** `{self.audit.get('overall_status')}`\n\n")
            f.write("## Layer 0: Dataset Integrity\n")
            f.write(f"```json\n{json.dumps(self.audit.get('layer0_dataset_integrity'), indent=2)}\n```\n\n")
            f.write("## Layer 1: Physical Validation\n")
            f.write(f"```json\n{json.dumps(self.audit.get('layer1_physical_validation'), indent=2)}\n```\n\n")
            f.write("## Layer 2: TRACEBIND Metrics\n")
            f.write(f"```json\n{json.dumps(self.audit.get('layer2_tracebind_metrics'), indent=2)}\n```\n")