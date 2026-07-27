import numpy as np
from scipy.stats import entropy
from .quality import check_nan

def compute_texture_entropy(grid: np.ndarray, bins: int = 32) -> float:
    grid_clean = check_nan(grid)
    counts, _ = np.histogram(grid_clean, bins=bins)
    probs = counts / np.sum(counts) if np.sum(counts) > 0 else np.zeros_like(counts)
    probs = probs[probs > 0]
    return float(entropy(probs, base=2)) if len(probs) > 0 else 0.0