import numpy as np

def generate_white_noise(grid_size: int = 200, mean: float = 0.0, std_dev: float = 10.0, seed: int = 42):
    """
    Generates deterministic Gaussian noise using an explicit RNG seed.
    """
    rng = np.random.default_rng(seed)
    return rng.normal(mean, std_dev, (grid_size, grid_size))