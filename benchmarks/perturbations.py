"""
Perturbation Operators and Field Generators for Spatial Benchmarking
"""

import numpy as np
from scipy import ndimage


def generate_base_field(size: int = 128, seed: int = None) -> np.ndarray:
    """Generates a reproducible 2D Gaussian random field with continuous spatial structure."""
    if seed is not None:
        np.random.seed(seed)
    x = np.linspace(-3, 3, size)
    xx, yy = np.meshgrid(x, x)
    base = np.sin(xx) * np.cos(yy) + np.random.normal(0, 0.1, (size, size))
    return base


def apply_perturbation(grid: np.ndarray, p_type: str, severity: float = 1.0) -> np.ndarray:
    """Applies a specific controlled structural perturbation to an input field."""
    out = grid.copy()
    h, w = out.shape

    if p_type == "Gaussian Noise":
        out += np.random.normal(0, 0.3 * severity, out.shape)

    elif p_type == "Salt-and-Pepper":
        mask = np.random.rand(*out.shape) < (0.05 * severity)
        out[mask] = np.max(out) * 2.0

    elif p_type == "Phase Randomization":
        fft_grid = np.fft.fft2(out)
        amp = np.abs(fft_grid)
        orig_phase = np.angle(fft_grid)
        rand_phase = np.random.uniform(-np.pi, np.pi, out.shape)
        mixed_phase = (1.0 - severity) * orig_phase + severity * rand_phase
        out = np.real(np.fft.ifft2(amp * np.exp(1j * mixed_phase)))

    elif p_type == "Spatial Warp":
        out = ndimage.gaussian_filter(out, sigma=2.0 * severity)

    elif p_type == "Local Masking":
        mask_size = int(h * 0.25 * severity)
        y0, x0 = h // 4, w // 4
        out[y0 : y0 + mask_size, x0 : x0 + mask_size] = 0.0

    elif p_type == "Edge Masking":
        margin = int(min(h, w) * 0.1 * severity)
        out[:margin, :] = 0.0
        out[-margin:, :] = 0.0
        out[:, :margin] = 0.0
        out[:, -margin:] = 0.0

    elif p_type == "Block Removal":
        out[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3] = 0.0

    elif p_type == "Rotation":
        angle = 45.0 * severity
        out = ndimage.rotate(out, angle, reshape=False, mode="nearest")

    elif p_type == "Translation":
        shift = int(10 * severity)
        out = np.roll(out, shift=(shift, shift), axis=(0, 1))

    elif p_type == "Histogram Equalization":
        flat = out.flatten()
        ranks = np.argsort(np.argsort(flat))
        out = (ranks / len(flat)).reshape(out.shape)

    elif p_type == "Contrast Scaling":
        out = out * (1.0 + severity)

    return out