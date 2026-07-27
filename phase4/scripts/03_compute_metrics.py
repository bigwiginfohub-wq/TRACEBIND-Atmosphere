import os
import sys
import numpy as np
import pandas as pd
from scipy.ndimage import sobelfilter, gaussian_filter
from scipy.stats import entropy

# Add parent directory if necessary to import custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

try:
    from metrics import compute_tracebind  # Your core TRACEBIND function
except ImportError:
    # Fallback mock/placeholder if metrics module is elsewhere
    def compute_tracebind(grid: np.ndarray) -> float:
        return float(np.std(grid) / (np.mean(grid) + 1e-8))


def compute_morans_i(grid: np.ndarray) -> float:
    """Computes global Moran's I for a 2D spatial grid (4-neighbor connectivity)."""
    grid_clean = np.nan_to_num(grid)
    z = grid_clean - np.mean(grid_clean)
    s0 = 0.0
    numerator = 0.0
    
    # Vectorized 4-neighbor spatial weights computation
    num_lat, num_lon = grid.shape
    # Horizontal pairs
    diff_h = z[:, :-1] * z[:, 1:]
    # Vertical pairs
    diff_v = z[:-1, :] * z[1:, :]
    
    numerator = np.sum(diff_h) + np.sum(diff_v)
    s0 = diff_h.size + diff_v.size
    denominator = np.sum(z**2)
    
    if denominator == 0:
        return 0.0
    return float((grid.size / s0) * (numerator / denominator))


def compute_gearys_c(grid: np.ndarray) -> float:
    """Computes global Geary's C for a 2D spatial grid."""
    grid_clean = np.nan_to_num(grid)
    z = grid_clean - np.mean(grid_clean)
    
    diff_h = (grid_clean[:, :-1] - grid_clean[:, 1:]) ** 2
    diff_v = (grid_clean[:-1, :] - grid_clean[1:, :]) ** 2
    
    numerator = np.sum(diff_h) + np.sum(diff_v)
    s0 = diff_h.size + diff_v.size
    denominator = np.sum(z**2)
    
    if denominator == 0:
        return 1.0
    return float(((grid.size - 1) / (2 * s0)) * (numerator / denominator))


def compute_gradient_energy(grid: np.ndarray) -> float:
    """Computes mean spatial gradient magnitude squared using Sobel operators."""
    grid_clean = np.nan_to_num(grid)
    gx = sobelfilter(grid_clean, axis=0) if hasattr(sobelfilter, '__call__') else np.gradient(grid_clean, axis=0)
    gy = sobelfilter(grid_clean, axis=1) if hasattr(sobelfilter, '__call__') else np.gradient(grid_clean, axis=1)
    grad_mag_sq = gx**2 + gy**2
    return float(np.mean(grad_mag_sq))


def compute_texture_entropy(grid: np.ndarray, bins: int = 32) -> float:
    """Computes 1D histogram Shannon entropy across the frame."""
    grid_clean = np.nan_to_num(grid)
    counts, _ = np.histogram(grid_clean, bins=bins)
    probabilities = counts / np.sum(counts)
    probabilities = probabilities[probabilities > 0]
    return float(entropy(probabilities, base=2))


def process_era5_metrics(data_array: np.ndarray, dates: list = None) -> pd.DataFrame:
    """
    Processes a 3D array [time, lat, lon] and computes spatial metrics for each time step.
    Returns a tidy pandas DataFrame.
    """
    if data_array.ndim == 2:
        data_array = np.expand_dims(data_array, axis=0)

    n_times = data_array.shape[0]
    results = []

    print(f"Extracting metrics across {n_times} time steps...")

    for t in range(n_times):
        frame = data_array[t]
        t_label = dates[t] if dates and t < len(dates) else t

        # Calculate metrics frame-by-frame
        tb = compute_tracebind(frame)
        moran = compute_morans_i(frame)
        geary = compute_gearys_c(frame)
        var = float(np.var(frame))
        ent = compute_texture_entropy(frame)
        grad_energy = compute_gradient_energy(frame)

        results.append({
            "Time": t_label,
            "TRACEBIND": tb,
            "Moran_I": moran,
            "Geary_C": geary,
            "Variance": var,
            "Texture_Entropy": ent,
            "Gradient_Energy": grad_energy
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    print("`compute_metrics.py` module ready.")