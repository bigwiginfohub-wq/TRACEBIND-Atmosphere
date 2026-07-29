#!/usr/bin/env python3
"""
TRACEBIND Phase 7 - Binary Vortex Interaction Experiment & CSV Export
===================================================================
File: phase7/sandbox/tests/verify_binary_interaction.py
"""

import sys
import csv
from pathlib import Path

SANDBOX_DIR = Path(__file__).resolve().parent.parent
if str(SANDBOX_DIR) not in sys.path:
    sys.path.insert(0, str(SANDBOX_DIR))

import numpy as np
import matplotlib.pyplot as plt
from fields.synthetic_vortex import make_grid, lamb_oseen_vortex


def compute_normalized_coherence(X: np.ndarray, Y: np.ndarray, u: np.ndarray, v: np.ndarray) -> float:
    """Computes bounded Phase Coherence C_phi around domain centroid."""
    R = np.sqrt(X**2 + Y**2) + 1e-12
    tx = -Y / R
    ty = X / R
    speed = np.sqrt(u**2 + v**2) + 1e-12
    proj = np.abs((u / speed) * tx + (v / speed) * ty)
    return float(np.mean(proj))


def run_binary_vortex_experiment():
    X, Y = make_grid(nx=301, ny=301, dx=2000.0, dy=2000.0)
    r_core = 40000.0
    gamma0 = 1e5

    # Separation distance range relative to core radius: d* = d / r_core
    d_star_range = np.linspace(1.0, 6.0, 35)

    c_corotating = []
    c_counterrotating = []

    for d_star in d_star_range:
        sep = d_star * r_core
        
        # 1. Co-rotating pair (Gamma1 = +Gamma0, Gamma2 = +Gamma0)
        v1 = lamb_oseen_vortex(X, Y, gamma=gamma0, r_core=r_core, center=(-sep / 2.0, 0.0))
        v2_co = lamb_oseen_vortex(X, Y, gamma=gamma0, r_core=r_core, center=(sep / 2.0, 0.0))
        u_co = v1["u"] + v2_co["u"]
        v_co = v1["v"] + v2_co["v"]
        c_corotating.append(compute_normalized_coherence(X, Y, u_co, v_co))

        # 2. Counter-rotating pair (Gamma1 = +Gamma0, Gamma2 = -Gamma0)
        v2_cnt = lamb_oseen_vortex(X, Y, gamma=-gamma0, r_core=r_core, center=(sep / 2.0, 0.0))
        u_cnt = v1["u"] + v2_cnt["u"]
        v_cnt = v1["v"] + v2_cnt["v"]
        c_counterrotating.append(compute_normalized_coherence(X, Y, u_cnt, v_cnt))

    # --- CSV Export ---
    reports_dir = SANDBOX_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / "binary_vortex_coherence_benchmark.csv"

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["d_star", "separation_m", "c_phi_corotating", "c_phi_counterrotating"])
        for d_s, c_co, c_cnt in zip(d_star_range, c_corotating, c_counterrotating):
            writer.writerow([f"{d_s:.4f}", f"{d_s * r_core:.1f}", f"{c_co:.6f}", f"{c_cnt:.6f}"])

    print(f"[✓] Quantitative benchmark CSV exported to: {csv_path}")

    # --- Plotting (Raw String Literals r'...' To Avoid Escape Warnings) ---
    plt.figure(figsize=(8, 5))
    plt.plot(d_star_range, c_corotating, 'b-o', lw=2, ms=4, label=r'Co-Rotating Pair ($\Gamma_1 = \Gamma_2$)')
    plt.plot(d_star_range, c_counterrotating, 'r-s', lw=2, ms=4, label=r'Counter-Rotating Pair ($\Gamma_1 = -\Gamma_2$)')
    plt.axhline(2 / np.pi, color='k', linestyle='--', label=r'Asymptotic Limit ($2/\pi \approx 0.6366$)')
    
    plt.title(r'Phase Coherence $C_\phi(d^*)$ vs. Binary Vortex Separation')
    plt.xlabel(r'Normalized Separation Distance $d^* = d / r_{\text{core}}$')
    plt.ylabel(r'Phase Coherence ($C_\phi$)')
    plt.ylim(0.5, 1.0)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    output_path = reports_dir / "binary_vortex_coherence_curve.png"
    plt.savefig(output_path, dpi=300)
    print(f"[✓] Clean plot generated and saved to: {output_path}")


if __name__ == "__main__":
    run_binary_vortex_experiment()