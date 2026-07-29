#!/usr/bin/env python3
"""
TRACEBIND Phase 7B - MetPy Derivative Error Spectrum & Cross-Validation
========================================================================
File: phase7/validation/verify_metpy_crossval.py
"""

import sys
from pathlib import Path
import numpy as np
import metpy.calc as mpcalc
from metpy.units import units
import matplotlib.pyplot as plt

VALIDATION_DIR = Path(__file__).resolve().parent
SANDBOX_DIR = VALIDATION_DIR.parent / "sandbox"
if str(SANDBOX_DIR) not in sys.path:
    sys.path.insert(0, str(SANDBOX_DIR))

from fields.synthetic_vortex import make_grid, lamb_oseen_vortex


def run_error_spectrum_validation():
    # 1. Grid setup
    dx_m, dy_m = 2500.0, 2500.0
    nx, ny = 201, 201
    X, Y = make_grid(nx=nx, ny=ny, dx=dx_m, dy=dy_m)
    vortex = lamb_oseen_vortex(X, Y, gamma=1e5, r_core=50000.0)
    
    u_raw, v_raw = vortex["u"], vortex["v"]

    # 2. Internal 2nd-order Central Difference Vorticity
    du_dy = np.gradient(u_raw, dy_m, axis=0)
    dv_dx = np.gradient(v_raw, dx_m, axis=1)
    vort_internal = dv_dx - du_dy

    # 3. MetPy Vorticity Implementation
    u_metpy = u_raw * units("m/s")
    v_metpy = v_raw * units("m/s")
    vort_metpy = mpcalc.vorticity(u_metpy, v_metpy, dx=dx_m * units.meter, dy=dy_m * units.meter).magnitude

    # 4. Error Field & Domain Masking (Exclude 5-cell boundary halo)
    diff_full = vort_internal - vort_metpy
    
    interior_mask = np.zeros((ny, nx), dtype=bool)
    halo = 5
    interior_mask[halo:-halo, halo:-halo] = True

    diff_interior = diff_full[interior_mask]
    vort_ref_interior = vort_metpy[interior_mask]

    # 5. Metrics (Absolute and Relative Scale-Tied)
    vort_scale = float(np.max(np.abs(vort_ref_interior)))
    
    rmse = float(np.sqrt(np.mean(diff_interior**2)))
    rel_rmse = rmse / vort_scale

    l_infinity = float(np.max(np.abs(diff_interior)))
    rel_l_infinity = l_infinity / vort_scale

    corr_coef = float(np.corrcoef(vort_internal[interior_mask], vort_metpy[interior_mask])[0, 1])
    msb = float(np.mean(diff_interior))
    rel_bias = msb / vort_scale

    # 6. Display Summary Table
    print("=" * 65)
    print(" METPY KINEMATIC CROSS-VALIDATION ERROR SPECTRUM")
    print("=" * 65)
    print(f"  * Reference Scale (Max |ζ|)    : {vort_scale:.6e} s^-1")
    print(f"  * RMSE                         : {rmse:.6e} s^-1  (Rel: {rel_rmse:.3e})")
    print(f"  * L_infinity (Max Abs Error)   : {l_infinity:.6e} s^-1  (Rel: {rel_l_infinity:.3e})")
    print(f"  * Pearson Correlation (r)      : {corr_coef:.12f}")
    print(f"  * Mean Signed Bias (MSB)       : {msb:.6e} s^-1  (Rel: {rel_bias:.3e})")
    print(f"  * Boundary Halo Excluded       : {halo} grid cells")
    print("-" * 65)

    # 7. Generate Diagnostic Error Plots
    artifacts_dir = VALIDATION_DIR / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    im = axes[0].imshow(diff_full, cmap="coolwarm", origin="lower")
    fig.colorbar(im, ax=axes[0], label="Difference (s^-1)")
    axes[0].set_title("Vorticity Difference Field (Internal - MetPy)")
    
    axes[1].hist(diff_interior.flatten(), bins=50, color="navy", alpha=0.7, edgecolor="black")
    axes[1].set_title("Interior Error Distribution (5-cell halo excluded)")
    axes[1].set_xlabel("Error (s^-1)")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(artifacts_dir / "metpy_crossval_error_diagnostics.png", dpi=300)
    plt.close()

    # 8. Scale-Tied Scale Invariant Assertion Gates
    assert rel_rmse < 1e-6, f"Relative RMSE {rel_rmse:.3e} exceeds tolerance 1e-6"
    assert rel_l_infinity < 1e-5, f"Relative L_infinity {rel_l_infinity:.3e} exceeds tolerance 1e-5"
    assert corr_coef > 0.999999, f"Correlation {corr_coef:.8f} below threshold 0.999999"
    assert abs(rel_bias) < 1e-6, f"Relative bias {rel_bias:.3e} exceeds threshold 1e-6"

    print("[✓] PASS: Derivative engine validated against MetPy baseline.\n")


if __name__ == "__main__":
    run_error_spectrum_validation()