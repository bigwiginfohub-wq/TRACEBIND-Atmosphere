import numpy as np
from .quality import check_nan

def compute_morans_i(grid: np.ndarray) -> float:
    grid_clean = check_nan(grid)
    z = grid_clean - np.mean(grid_clean)
    diff_h = z[:, :-1] * z[:, 1:]
    diff_v = z[:-1, :] * z[1:, :]
    numerator = np.sum(diff_h) + np.sum(diff_v)
    s0 = diff_h.size + diff_v.size
    denominator = np.sum(z**2)
    if denominator == 0:
        return 0.0
    return float((grid.size / s0) * (numerator / denominator))

def compute_gearys_c(grid: np.ndarray) -> float:
    grid_clean = check_nan(grid)
    z = grid_clean - np.mean(grid_clean)
    diff_h = (grid_clean[:, :-1] - grid_clean[:, 1:]) ** 2
    diff_v = (grid_clean[:-1, :] - grid_clean[1:, :]) ** 2
    numerator = np.sum(diff_h) + np.sum(diff_v)
    s0 = diff_h.size + diff_v.size
    denominator = np.sum(z**2)
    if denominator == 0:
        return 1.0
    return float(((grid.size - 1) / (2 * s0)) * (numerator / denominator))