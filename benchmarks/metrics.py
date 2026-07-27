"""
Unified Metrics Wrappers for Spatial Benchmarking
Provides Moran's I, Geary's C, Semivariogram Range, and TRACEBIND metric functions.
"""

import numpy as np
from scipy import ndimage


def compute_morans_i(grid: np.ndarray) -> float:
    """Computes global Moran's I for a 2D scalar field (queen contiguity)."""
    nx, ny = grid.shape
    z = grid - np.mean(grid)

    dx = z[:-1, :] * z[1:, :]
    dy = z[:, :-1] * z[:, :-1]
    d_diag1 = z[:-1, :-1] * z[1:, 1:]
    d_diag2 = z[:-1, 1:] * z[1:, :-1]

    numerator = np.sum(dx) + np.sum(dy) + np.sum(d_diag1) + np.sum(d_diag2)
    s0 = (nx - 1) * ny + nx * (ny - 1) + 2 * (nx - 1) * (ny - 1)

    denominator = np.sum(z**2)
    if denominator == 0:
        return 0.0
    return float((nx * ny / s0) * (numerator / denominator))


def compute_gearys_c(grid: np.ndarray) -> float:
    """Computes Geary's C for a 2D scalar field."""
    nx, ny = grid.shape
    z = grid - np.mean(grid)

    diff_x = (grid[:-1, :] - grid[1:, :]) ** 2
    diff_y = (grid[:, :-1] - grid[:, 1:]) ** 2

    sum_sq_diff = np.sum(diff_x) + np.sum(diff_y)
    w0 = (nx - 1) * ny + nx * (ny - 1)

    var_z = np.sum(z**2)
    if var_z == 0:
        return 1.0
    return float(((nx * ny - 1) * sum_sq_diff) / (2 * w0 * var_z))


def compute_variogram_range(grid: np.ndarray, max_lags: int = 10) -> float:
    """Estimates empirical semivariogram correlation length proxy."""
    nx, ny = grid.shape
    sill = np.var(grid)
    if sill == 0:
        return 0.0

    gamma = []
    for h in range(1, max_lags + 1):
        diff_x = (grid[:-h, :] - grid[h:, :]) ** 2 if nx > h else np.array([0])
        diff_y = (grid[:, :-h] - grid[:, h:]) ** 2 if ny > h else np.array([0])
        gamma.append(0.5 * (np.mean(diff_x) + np.mean(diff_y)))

    gamma = np.array(gamma)
    idx = np.where(gamma >= 0.95 * sill)[0]
    return float((idx[0] + 1) if len(idx) > 0 else max_lags)


def compute_tracebind_metric(grid: np.ndarray, k: int = 8, radius: int = 3) -> float:
    """Computes TRACEBIND kinematic phase response metric."""
    dx = ndimage.sobel(grid, axis=0)
    dy = ndimage.sobel(grid, axis=1)
    grad_mag = np.hypot(dx, dy)
    phase = np.arctan2(dy, dx)

    local_phase_var = ndimage.generic_filter(phase, np.var, size=radius)
    metric = np.mean(grad_mag * np.exp(-local_phase_var)) * (1.0 / (k**0.5))
    return float(metric)


def evaluate_all_metrics(grid: np.ndarray) -> dict:
    """Evaluates all four spatial metrics on a single grid array."""
    return {
        "Moran's I": compute_morans_i(grid),
        "Geary's C": compute_gearys_c(grid),
        "Semivariogram Range": compute_variogram_range(grid),
        "TRACEBIND": compute_tracebind_metric(grid),
    }