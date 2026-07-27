import numpy as np

def check_nan(grid: np.ndarray) -> np.ndarray:
    """Fills NaNs/infs with zeros and returns a clean array."""
    return np.nan_to_num(grid, nan=0.0, posinf=0.0, neginf=0.0)

def normalize(grid: np.ndarray) -> np.ndarray:
    """Z-score normalizes a 2D grid frame."""
    grid_clean = check_nan(grid)
    std = np.std(grid_clean)
    if std < 1e-8:
        return grid_clean - np.mean(grid_clean)
    return (grid_clean - np.mean(grid_clean)) / std

def validate_grid(grid: np.ndarray) -> bool:
    """Validates grid dimensions and content."""
    if grid is None or grid.ndim != 2:
        return False
    if grid.size == 0 or np.all(np.isnan(grid)):
        return False
    return True