#!/usr/bin/env python3
"""
TRACEBIND Phase 7 - Complete 6-Gate Synthetic Sandbox Suite
==========================================================
File: phase7/sandbox/tests/verify_suite.py
"""

import sys
from pathlib import Path

SANDBOX_DIR = Path(__file__).resolve().parent.parent
if str(SANDBOX_DIR) not in sys.path:
    sys.path.insert(0, str(SANDBOX_DIR))

import numpy as np
from fields.synthetic_vortex import make_grid, lamb_oseen_vortex, inject_gaussian_noise
from metrics.coherence import compute_phase_coherence
from tests.verify_convergence import compute_vorticity_2nd_order, verify_grid_convergence


def run_scale_invariance_test() -> dict:
    """Verifies that EOC >= 1.80 holds across multiple vortex core radii r_c."""
    core_radii = [25_000.0, 50_000.0, 100_000.0, 200_000.0]
    eoc_results = []

    for r_core in core_radii:
        grid_sizes = [64, 128]
        errors = []
        h_vals = []

        for N in grid_sizes:
            # Scale domain relative to vortex core size
            domain_extent = r_core * 20.0
            dx = dy = domain_extent / N
            x = np.linspace(-domain_extent / 2.0 + dx / 2.0, domain_extent / 2.0 - dx / 2.0, N)
            y = np.linspace(-domain_extent / 2.0 + dy / 2.0, domain_extent / 2.0 - dy / 2.0, N)
            X, Y = np.meshgrid(x, y)

            R = np.sqrt(X**2 + Y**2) + 1e-12
            gamma = 1e5
            v_theta = (gamma / (2.0 * np.pi * R)) * (1.0 - np.exp(-(R / r_core)**2))
            u = -v_theta * (Y / R)
            v = v_theta * (X / R)
            zeta_exact = (gamma / (np.pi * r_core**2)) * np.exp(-(R / r_core)**2)

            zeta_num = compute_vorticity_2nd_order(u, v, dx, dy)
            margin = int(N * 0.10)
            inner = slice(margin, -margin)

            err_l2 = np.sqrt(np.mean((zeta_num[inner, inner] - zeta_exact[inner, inner])**2))
            ref_l2 = np.sqrt(np.mean(zeta_exact[inner, inner]**2))
            errors.append(err_l2 / ref_l2)
            h_vals.append(dx)

        eoc = np.log(errors[0] / errors[1]) / np.log(h_vals[0] / h_vals[1])
        eoc_results.append(float(eoc))

    all_pass = all(e >= 1.80 for e in eoc_results)
    return {"core_radii_m": core_radii, "observed_eocs": eoc_results, "pass": bool(all_pass)}


def run_vortex_shear_ordering_test() -> dict:
    """Verifies monotonic coherence degradation under increasing environmental shear strain."""
    X, Y = make_grid(nx=201, ny=201, dx=2500.0, dy=2500.0)
    vortex = lamb_oseen_vortex(X, Y, gamma=1e5, r_core=50000.0)

    # Reference shear scale S_0 based on maximum vortex shear strain
    s_0 = 1e5 / (2.0 * np.pi * (50000.0**2))
    shear_ratios = [0.0, 0.1, 0.2, 0.4, 0.8]
    coherence_vals = []

    for ratio in shear_ratios:
        s_rate = ratio * s_0
        u_shear = vortex["u"] + s_rate * Y
        v_shear = vortex["v"]
        c_val = compute_phase_coherence(X, Y, u_shear, v_shear)["C_phi"]
        coherence_vals.append(float(c_val))

    # Coherence must strictly decrease as shear increases
    is_monotonic = all(x > y for x, y in zip(coherence_vals, coherence_vals[1:]))
    return {"shear_ratios": shear_ratios, "coherence": coherence_vals, "pass": bool(is_monotonic)}


if __name__ == "__main__":
    print("==========================================================================")
    print("          TRACEBIND PHASE 7 - ADDITIONAL REGRESSION SUITES               ")
    print("==========================================================================")

    scale_res = run_scale_invariance_test()
    print(f"[{'✓' if scale_res['pass'] else '✗'}] TEST_005_SCALE_INVARIANCE:")
    for r, e in zip(scale_res['core_radii_m'], scale_res['observed_eocs']):
        print(f"    Radius {r/1000:5.0f} km -> Observed EOC: {e:.4f}")

    shear_res = run_vortex_shear_ordering_test()
    print(f"\n[{'✓' if shear_res['pass'] else '✗'}] TEST_006_SHEAR_STRAIN_ORDERING:")
    for sr, c in zip(shear_res['shear_ratios'], shear_res['coherence']):
        print(f"    Shear Ratio {sr:3.1f}x -> Phase Coherence C_phi: {c:.5f}")

    print("==========================================================================")