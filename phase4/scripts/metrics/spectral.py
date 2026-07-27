import numpy as np
from .quality import check_nan

def compute_gradient_energy(grid: np.ndarray) -> float:
    grid_clean = check_nan(grid)
    gy, gx = np.gradient(grid_clean)
    return float(np.mean(gx**2 + gy**2))