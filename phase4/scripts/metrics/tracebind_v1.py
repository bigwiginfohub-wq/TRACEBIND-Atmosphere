import numpy as np
from .quality import check_nan

def compute_tracebind_v1(grid: np.ndarray, mode: str = "coherence") -> float:
    """
    TRACEBIND v1 (Frozen Baseline Formulation).
    Global scalar interaction coupling mean gradient magnitude with standard deviation.
    """
    grid_clean = check_nan(grid)
    if grid_clean.size == 0 or grid_clean.ndim != 2:
        return np.nan
    
    std_val = np.std(grid_clean)
    if std_val < 1e-8:
        return np.nan
        
    norm_grid = (grid_clean - np.mean(grid_clean)) / std_val
    gy, gx = np.gradient(norm_grid)
    grad_mag = np.sqrt(gx**2 + gy**2)
    
    raw_div = float(np.mean(grad_mag) * std_val)
    
    if mode == "coherence":
        return float(1.0 / (1.0 + raw_div))
    return raw_div