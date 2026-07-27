import numpy as np
from .quality import check_nan, normalize

def compute_tracebind(grid: np.ndarray, mode: str = "coherence") -> float:
    """
    TRACEBIND Kinematic-Phase Interaction Metric (v1 Production).
    
    Parameters:
    -----------
    grid : np.ndarray
        2D spatial atmospheric field frame.
    mode : str
        'coherence' : Returns [0, 1] index where 1.0 = perfect spatial coherence, 0.0 = total chaos.
        'divergence': Returns raw gradient-coupling value (higher = higher spatial disorder).
    """
    grid_clean = check_nan(grid)
    if grid_clean.size == 0 or grid_clean.ndim != 2:
        return 0.0
    
    std_val = np.std(grid_clean)
    if std_val < 1e-8:
        return 1.0 if mode == "coherence" else 0.0
        
    norm_grid = (grid_clean - np.mean(grid_clean)) / std_val
    gy, gx = np.gradient(norm_grid)
    grad_mag = np.sqrt(gx**2 + gy**2)
    
    raw_divergence = float(np.mean(grad_mag) * std_val)
    
    if mode == "coherence":
        # Bounded [0, 1] Coherence Metric
        return float(1.0 / (1.0 + raw_divergence))
    return raw_divergence