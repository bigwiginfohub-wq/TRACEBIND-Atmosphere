#!/usr/bin/env python3
"""
TRACEBIND Phase 7 - Corrected Sensitivity & 2D Response Surface Generator
========================================================================
File: phase7/sandbox/tests/plot_sensitivity.py
"""

import sys
from pathlib import Path

SANDBOX_DIR = Path(__file__).resolve().parent.parent
if str(SANDBOX_DIR) not in sys.path:
    sys.path.insert(0, str(SANDBOX_DIR))

import numpy as np
import matplotlib.pyplot as plt
from fields.synthetic_vortex import make_grid, lamb_oseen_vortex, inject_gaussian_noise


def compute_normalized_coherence(X: np.ndarray, Y: np.ndarray, u: np.ndarray, v: np.ndarray) -> float:
    """Computes strictly bounded Phase Coherence C_phi in [0, 1]."""
    R = np.sqrt(X**2 + Y**2) + 1e-12
    # Tangential unit vectors
    tx = -Y / R
    ty = X / R
    
    speed = np.sqrt(u**2 + v**2) + 1e-12
    u_hat = u / speed
    v_hat = v / speed
    
    # Absolute projection onto tangential direction
    proj = np.abs(u_hat * tx + v_hat * ty)
    return float(np.mean(proj))


def generate_diagnostics():
    X, Y = make_grid(nx=201, ny=201, dx=2500.0, dy=2500.0)
    vortex_base = lamb_oseen_vortex(X, Y, gamma=1e5, r_core=50000.0)
    s_0 = 1e5 / (2.0 * np.pi * (50000.0**2))

    # -------------------------------------------------------------------------
    # 1. Corrected 1D Sensitivity Curves
    # -------------------------------------------------------------------------
    shears = np.linspace(0.0, 1.5, 30)
    c_shear = [compute_normalized_coherence(X, Y, vortex_base["u"] + s * s_0 * Y, vortex_base["v"]) for s in shears]

    eccentricities = np.linspace(0.0, 0.95, 30)
    c_ecc = []
    for ecc in eccentricities:
        v_ecc = lamb_oseen_vortex(X, Y, gamma=1e5, r_core=50000.0, eccentricity=ecc)
        c_ecc.append(compute_normalized_coherence(X, Y, v_ecc["u"], v_ecc["v"]))

    snrs = np.linspace(-10.0, 30.0, 30)
    c_snr = []
    for snr in snrs:
        u_n, v_n = inject_gaussian_noise(vortex_base["u"], vortex_base["v"], snr_db=snr, seed=42)
        c_snr.append(compute_normalized_coherence(X, Y, u_n, v_n))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].plot(shears, c_shear, 'b-o', lw=2, ms=4)
    axes[0].axhline(2/np.pi, color='r', linestyle='--', label=r'Theoretical Bound ($2/\pi \approx 0.6366$)')
    axes[0].set_title(r'Phase Coherence vs. Shear Ratio ($S/S_0$)')
    axes[0].set_xlabel(r'Relative Shear Strain Rate ($S/S_0$)')
    axes[0].set_ylabel(r'Phase Coherence ($C_\phi$)')
    axes[0].set_ylim(0.5, 1.05)
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(eccentricities, c_ecc, 'g-s', lw=2, ms=4)
    axes[1].set_title(r'Phase Coherence vs. Core Ellipticity ($e$)')
    axes[1].set_xlabel(r'Ellipticity ($e = \sqrt{1 - b^2/a^2}$)')
    axes[1].set_ylabel(r'Phase Coherence ($C_\phi$)')
    axes[1].set_ylim(0.5, 1.05)  # Fixed bounded Y-axis
    axes[1].grid(True)

    axes[2].plot(snrs, c_snr, 'm-^', lw=2, ms=4)
    axes[2].axhline(2/np.pi, color='r', linestyle='--', label=r'Noise Floor ($2/\pi$)')
    axes[2].set_title(r'Phase Coherence vs. Signal Noise (SNR dB)')
    axes[2].set_xlabel('SNR (dB)')
    axes[2].set_ylabel(r'Phase Coherence ($C_\phi$)')
    axes[2].set_ylim(0.5, 1.05)
    axes[2].grid(True)

    plt.tight_layout()
    output_1d = SANDBOX_DIR / "reports" / "metric_sensitivity_curves_corrected.png"
    plt.savefig(output_1d, dpi=300)
    print(f"[✓] Corrected 1D sensitivity curves saved to: {output_1d}")

    # -------------------------------------------------------------------------
    # 2. 2D Response Surface Maps: C_phi = f(Shear, SNR) & f(Ellipticity, Shear)
    # -------------------------------------------------------------------------
    n_res = 25
    shear_grid = np.linspace(0.0, 1.2, n_res)
    snr_grid = np.linspace(-5.0, 25.0, n_res)
    ecc_grid = np.linspace(0.0, 0.9, n_res)

    S_mat, SNR_mat = np.meshgrid(shear_grid, snr_grid)
    C_shear_snr = np.zeros_like(S_mat)

    for i in range(n_res):
        for j in range(n_res):
            s_val = S_mat[i, j]
            snr_val = SNR_mat[i, j]
            u_s = vortex_base["u"] + s_val * s_0 * Y
            v_s = vortex_base["v"]
            u_n, v_n = inject_gaussian_noise(u_s, v_s, snr_db=snr_val, seed=42)
            C_shear_snr[i, j] = compute_normalized_coherence(X, Y, u_n, v_n)

    E_mat, S_mat2 = np.meshgrid(ecc_grid, shear_grid)
    C_ecc_shear = np.zeros_like(E_mat)

    for i in range(n_res):
        for j in range(n_res):
            e_val = E_mat[i, j]
            s_val = S_mat2[i, j]
            v_e = lamb_oseen_vortex(X, Y, gamma=1e5, r_core=50000.0, eccentricity=e_val)
            u_s = v_e["u"] + s_val * s_0 * Y
            v_s = v_e["v"]
            C_ecc_shear[i, j] = compute_normalized_coherence(X, Y, u_s, v_s)

    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5.5))

    cp1 = axes2[0].contourf(S_mat, SNR_mat, C_shear_snr, levels=15, cmap='viridis')
    fig2.colorbar(cp1, ax=axes2[0], label=r'Phase Coherence ($C_\phi$)')
    axes2[0].set_title(r'$C_\phi(S, \text{SNR})$ Response Surface')
    axes2[0].set_xlabel(r'Relative Shear ($S/S_0$)')
    axes2[0].set_ylabel('SNR (dB)')

    cp2 = axes2[1].contourf(E_mat, S_mat2, C_ecc_shear, levels=15, cmap='magma')
    fig2.colorbar(cp2, ax=axes2[1], label=r'Phase Coherence ($C_\phi$)')
    axes2[1].set_title(r'$C_\phi(e, S)$ Response Surface')
    axes2[1].set_xlabel(r'Ellipticity ($e$)')
    axes2[1].set_ylabel(r'Relative Shear ($S/S_0$)')

    plt.tight_layout()
    output_2d = SANDBOX_DIR / "reports" / "metric_response_surface_2d.png"
    plt.savefig(output_2d, dpi=300)
    print(f"[✓] 2D response surfaces saved to: {output_2d}")


if __name__ == "__main__":
    generate_diagnostics()