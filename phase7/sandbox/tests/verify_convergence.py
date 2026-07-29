#!/usr/bin/env python3
"""
TRACEBIND Phase 7 - Convergence Verification with Explicit Central Differences
================================================================================
File: phase7/sandbox/tests/verify_convergence.py
"""

import numpy as np
from typing import Dict, List, Tuple


def compute_vorticity_2nd_order(u: np.ndarray, v: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Explicit 2nd-order interior central difference:
    zeta = dv/dx - du/dy
    """
    vorticity = np.zeros_like(u)
    
    # Central difference on interior points (2nd-order accurate)
    dv_dx = (v[1:-1, 2:] - v[1:-1, :-2]) / (2.0 * dx)
    du_dy = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2.0 * dy)
    
    vorticity[1:-1, 1:-1] = dv_dx - du_dy
    return vorticity


def verify_grid_convergence() -> Dict[str, float]:
    """Runs h-refinement series (64, 128, 256, 512) and checks EOC threshold."""
    # Fixed physical domain extent: L = 1,000 km
    domain_extent = 1_000_000.0  
    grid_sizes = [64, 128, 256, 512]
    errors = []
    h_values = []

    for N in grid_sizes:
        # Cell-centered grid coordinates to avoid placing node directly at r=0
        dx = domain_extent / N
        dy = domain_extent / N
        x = np.linspace(-domain_extent / 2.0 + dx / 2.0, domain_extent / 2.0 - dx / 2.0, N)
        y = np.linspace(-domain_extent / 2.0 + dy / 2.0, domain_extent / 2.0 - dy / 2.0, N)
        X, Y = np.meshgrid(x, y)

        # Analytical Lamb-Oseen velocity and vorticity
        R = np.sqrt(X**2 + Y**2) + 1e-12
        gamma = 1e5
        r_core = 50_000.0
        
        v_theta = (gamma / (2.0 * np.pi * R)) * (1.0 - np.exp(-(R / r_core)**2))
        u = -v_theta * (Y / R)
        v = v_theta * (X / R)
        zeta_exact = (gamma / (np.pi * r_core**2)) * np.exp(-(R / r_core)**2)

        # Numerical vorticity via explicit 2nd-order stencil
        zeta_num = compute_vorticity_2nd_order(u, v, dx, dy)

        # Mask boundary (10% domain margin) to evaluate pure interior error
        margin = int(N * 0.10)
        inner = slice(margin, -margin)
        
        # L2 Norm Relative Error
        err_l2 = np.sqrt(np.mean((zeta_num[inner, inner] - zeta_exact[inner, inner])**2))
        ref_l2 = np.sqrt(np.mean(zeta_exact[inner, inner]**2))
        rel_err = err_l2 / ref_l2

        errors.append(rel_err)
        h_values.append(dx)

    # Calculate Empirical Order of Convergence (EOC) between highest resolution steps
    eoc = np.log(errors[-2] / errors[-1]) / np.log(h_values[-2] / h_values[-1])
    
    return {
        "EOC": float(eoc),
        "err_64": float(errors[0]),
        "err_128": float(errors[1]),
        "err_256": float(errors[2]),
        "err_512": float(errors[3]),
        "pass": bool(eoc >= 1.80)
    }


if __name__ == "__main__":
    res = verify_grid_convergence()
    print("==========================================================================")
    print("                 TRACEBIND CONVERGENCE RE-EVALUATION                     ")
    print("==========================================================================")
    print(f" Observed EOC        : {res['EOC']:.4f}  (Required: >= 1.80)")
    print(f" Relative Error N=64 : {res['err_64']*100:.5f}%")
    print(f" Relative Error N=128: {res['err_128']*100:.5f}%")
    print(f" Relative Error N=256: {res['err_256']*100:.5f}%")
    print(f" Relative Error N=512: {res['err_512']*100:.5f}%")
    print(f" Promotion Gate Pass : {res['pass']}")
    print("==========================================================================")