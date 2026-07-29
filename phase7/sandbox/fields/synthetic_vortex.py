#!/usr/bin/env python3
"""
TRACEBIND Phase 7 - Advanced Synthetic Field Generators
======================================================
File: phase7/sandbox/fields/synthetic_vortex.py
"""

import numpy as np
from typing import Tuple, Dict


def make_grid(nx: int = 101, ny: int = 101, dx: float = 5000.0, dy: float = 5000.0) -> Tuple[np.ndarray, np.ndarray]:
    """Generates a 2D spatial coordinate grid centered at (0, 0)."""
    x = np.linspace(-(nx // 2) * dx, (nx // 2) * dx, nx)
    y = np.linspace(-(ny // 2) * dy, (ny // 2) * dy, ny)
    return np.meshgrid(x, y)


def lamb_oseen_vortex(
    X: np.ndarray, 
    Y: np.ndarray, 
    gamma: float = 1e5, 
    r_core: float = 50000.0,
    center: Tuple[float, float] = (0.0, 0.0),
    eccentricity: float = 0.0,
    angle_rad: float = 0.0
) -> Dict[str, np.ndarray]:
    """
    Computes analytical velocity (u, v) and vorticity fields for a generalized 
    Lamb-Oseen vortex supporting off-center placement and elliptical distortion.
    """
    # Shift to center
    Xc = X - center[0]
    Yc = Y - center[1]

    # Rotate coordinates for elliptical orientation
    if angle_rad != 0.0:
        X_rot = Xc * np.cos(angle_rad) + Yc * np.sin(angle_rad)
        Y_rot = -Xc * np.sin(angle_rad) + Yc * np.cos(angle_rad)
        Xc, Yc = X_rot, Y_rot

    # Elliptical core transformation: e = sqrt(1 - b^2/a^2)
    aspect_ratio = np.sqrt(1.0 - np.clip(eccentricity, 0.0, 0.99)**2)
    R_eff = np.sqrt((Xc / aspect_ratio)**2 + (Yc * aspect_ratio)**2) + 1e-12

    # Velocity magnitude
    v_theta = (gamma / (2.0 * np.pi * R_eff)) * (1.0 - np.exp(-(R_eff / r_core)**2))
    
    u = -v_theta * (Yc / R_eff)
    v = v_theta * (Xc / R_eff)
    vorticity = (gamma / (np.pi * r_core**2)) * np.exp(-(R_eff / r_core)**2)

    return {"u": u, "v": v, "vorticity": vorticity}


def binary_vortex_pair(
    X: np.ndarray, 
    Y: np.ndarray, 
    separation: float = 150000.0, 
    gamma: float = 1e5, 
    r_core: float = 50000.0
) -> Dict[str, np.ndarray]:
    """Generates a co-rotating binary vortex system to test multi-center interference."""
    v1 = lamb_oseen_vortex(X, Y, gamma=gamma, r_core=r_core, center=(-separation / 2.0, 0.0))
    v2 = lamb_oseen_vortex(X, Y, gamma=gamma, r_core=r_core, center=(separation / 2.0, 0.0))
    
    return {
        "u": v1["u"] + v2["u"],
        "v": v1["v"] + v2["v"],
        "vorticity": v1["vorticity"] + v2["vorticity"]
    }


def inject_gaussian_noise(u: np.ndarray, v: np.ndarray, snr_db: float, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Injects zero-mean Gaussian noise calibrated to a target SNR (dB)."""
    rng = np.random.default_rng(seed)
    signal_power = np.mean(u**2 + v**2)
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise_std = np.sqrt(noise_power / 2.0)
    
    return u + rng.normal(0.0, noise_std, size=u.shape), v + rng.normal(0.0, noise_std, size=v.shape)