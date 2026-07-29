#!/usr/bin/env python3
"""
TRACEBIND Phase 7 - Composite Superposition Fingerprint Benchmark
================================================================
File: phase7/sandbox/tests/verify_composite_superposition.py
"""

import sys
import csv
from pathlib import Path

SANDBOX_DIR = Path(__file__).resolve().parent.parent
if str(SANDBOX_DIR) not in sys.path:
    sys.path.insert(0, str(SANDBOX_DIR))

import numpy as np
from fields.synthetic_vortex import make_grid, lamb_oseen_vortex, inject_gaussian_noise


def compute_normalized_coherence(X: np.ndarray, Y: np.ndarray, u: np.ndarray, v: np.ndarray) -> float:
    """Computes bounded Phase Coherence C_phi around domain centroid."""
    R = np.sqrt(X**2 + Y**2) + 1e-12
    tx = -Y / R
    ty = X / R
    speed = np.sqrt(u**2 + v**2) + 1e-12
    proj = np.abs((u / speed) * tx + (v / speed) * ty)
    return float(np.mean(proj))


def run_composite_benchmark():
    X, Y = make_grid(nx=201, ny=201, dx=2500.0, dy=2500.0)
    r_core = 50000.0
    gamma0 = 1e5
    s_0 = gamma0 / (2.0 * np.pi * (r_core**2))

    # Parameter sweeps
    shear_ratios = [0.0, 0.2, 0.5, 0.8, 1.0]
    snr_levels_db = [-10.0, 0.0, 10.0, 20.0, 30.0]
    d_star_separations = [1.5, 2.5, 3.5, 5.0]

    reports_dir = SANDBOX_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / "composite_superposition_fingerprint.csv"

    records = []

    for d_star in d_star_separations:
        sep = d_star * r_core
        # Co-rotating pair base
        v1 = lamb_oseen_vortex(X, Y, gamma=gamma0, r_core=r_core, center=(-sep / 2.0, 0.0))
        v2 = lamb_oseen_vortex(X, Y, gamma=gamma0, r_core=r_core, center=(sep / 2.0, 0.0))
        u_base = v1["u"] + v2["u"]
        v_base = v1["v"] + v2["v"]

        for s_ratio in shear_ratios:
            u_sheared = u_base + (s_ratio * s_0 * Y)
            v_sheared = v_base

            for snr in snr_levels_db:
                u_comp, v_comp = inject_gaussian_noise(u_sheared, v_sheared, snr_db=snr, seed=42)
                c_val = compute_normalized_coherence(X, Y, u_comp, v_comp)
                records.append([
                    f"{d_star:.1f}", 
                    f"{s_ratio:.2f}", 
                    f"{snr:.1f}", 
                    f"{c_val:.6f}"
                ])

    # Export quantitative CSV fingerprint
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["d_star", "shear_ratio", "snr_db", "c_phi"])
        writer.writerows(records)

    print(f"[✓] Generated composite fingerprint with {len(records)} parameter states.")
    print(f"[✓] Benchmark CSV exported to: {csv_path}")


if __name__ == "__main__":
    run_composite_benchmark()