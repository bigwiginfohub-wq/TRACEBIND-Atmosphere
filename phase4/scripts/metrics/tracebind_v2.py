import numpy as np
from .quality import check_nan

def compute_tracebind_v2(grid: np.ndarray, mode: str = "coherence") -> float:
    """
    TRACEBIND v2 (Experimental Local Kinematic Gradient Alignment).
    
    Corrected Features:
    - True 8-neighbor stencil using interior slicing (no periodic roll boundary artifacts).
    - Linear rescaling [-1, 1] -> [0, 1] preserving anti-parallel vector physics.
    - Flat fields return NaN (undefined gradient directional coherence).
    """
    grid_clean = check_nan(grid)
    if grid_clean.size == 0 or grid_clean.ndim != 2:
        return np.nan
    
    std_val = np.std(grid_clean)
    if std_val < 1e-8:
        return np.nan  # Undefined gradient orientation for flat fields
        
    norm_grid = (grid_clean - np.mean(grid_clean)) / std_val
    gy, gx = np.gradient(norm_grid)
    grad_mag = np.sqrt(gx**2 + gy**2)
    
    eps = 1e-8
    u_x = gx / (grad_mag + eps)
    u_y = gy / (grad_mag + eps)
    
    # Extract interior grid [1:-1, 1:-1] to avoid boundary artifacts
    center_x, center_y = u_x[1:-1, 1:-1], u_y[1:-1, 1:-1]
    
    # 8-Neighbor Stencil Directions
    neighbors = [
        (u_x[:-2, 1:-1], u_y[:-2, 1:-1]),   # N
        (u_x[2:, 1:-1],  u_y[2:, 1:-1]),    # S
        (u_x[1:-1, 2:],  u_y[1:-1, 2:]),    # E
        (u_x[1:-1, :-2], u_y[1:-1, :-2]),   # W
        (u_x[:-2, 2:],   u_y[:-2, 2:]),     # NE
        (u_x[:-2, :-2],  u_y[:-2, :-2]),    # NW
        (u_x[2:, 2:],    u_y[2:, 2:]),      # SE
        (u_x[2:, :-2],   u_y[2:, :-2])      # SW
    ]
    
    # Compute Cosine Similarity across all 8 neighbors
    cos_sim_sum = np.zeros_like(center_x)
    for nx, ny in neighbors:
        cos_sim_sum += (center_x * nx + center_y * ny)
        
    mean_cos_sim = cos_sim_sum / 8.0  # Range [-1, 1]
    
    # Preserves physical distinction between orthogonal (0.5) and anti-parallel (0.0)
    coherence_map = (mean_cos_sim + 1.0) / 2.0  # Rescaled to [0, 1]
    
    v2_coherence = float(np.mean(coherence_map))
    
    if mode == "coherence":
        return v2_coherence
    return float(1.0 - v2_coherence)