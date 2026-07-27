# scripts/generate_reference_fixtures.py
import sys
import platform
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from tests.test_domain_validation import run_pipeline

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "tests" / "reference"
FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

# Metadata constants
FIXTURE_VERSION = "1.0.0"
CONTRACT_VERSION = "1.1.0"
DEFAULT_K = 4
DEFAULT_PERMUTATIONS = 50
DEFAULT_SEED = 42


def save_fixture(filename: str, data: np.ndarray, result):
    """Saves array data along with complete provenance metadata."""
    filepath = FIXTURE_DIR / filename
    
    np.savez(
        filepath,
        # Core data and results
        data=data,
        r_observed=np.array(result.r_observed, dtype=np.float64),
        z_score=np.array(result.z_score, dtype=np.float64),
        p_value=np.array(result.p_value, dtype=np.float64),
        fingerprint=str(result.fingerprint),
        
        # Pipeline Configuration (Prevents parameter mismatch confusion)
        config_k=np.array(DEFAULT_K),
        config_permutations=np.array(DEFAULT_PERMUTATIONS),
        config_seed=np.array(DEFAULT_SEED),
        config_null_strategy="GlobalPermutationNull",
        
        # Environment & Contract Provenance
        fixture_version=FIXTURE_VERSION,
        contract_version=CONTRACT_VERSION,
        numpy_version=np.__version__,
        python_version=platform.python_version(),
        platform_system=platform.system()
    )
    print(f"  ✓ Created {filename:<22} | R_obs: {result.r_observed:.6f} | FP: {result.fingerprint[:12]}...")


def generate_all_fixtures():
    print(f"\n=========================================================================")
    print(f"Generating Reference Fixtures with Provenance Data...")
    print(f"Target Directory: {FIXTURE_DIR}")
    print(f"NumPy Version: {np.__version__} | OS: {platform.system()}")
    print(f"=========================================================================")

    # 1. Smooth Field
    x = np.linspace(0, 10, 16)
    y = np.linspace(0, 10, 16)
    xx, yy = np.meshgrid(x, y)
    smooth_data = np.sin(xx) + np.cos(yy)
    res_smooth, _, _, _ = run_pipeline(smooth_data, k=DEFAULT_K, n_permutations=DEFAULT_PERMUTATIONS, seed=DEFAULT_SEED)
    save_fixture("smooth_16x16.npz", smooth_data, res_smooth)

    # 2. White Noise Field
    rng = np.random.default_rng(DEFAULT_SEED)
    noise_data = rng.normal(loc=0.0, scale=1.0, size=(16, 16))
    res_noise, _, _, _ = run_pipeline(noise_data, k=DEFAULT_K, n_permutations=DEFAULT_PERMUTATIONS, seed=DEFAULT_SEED)
    save_fixture("noise_16x16.npz", noise_data, res_noise)

    # 3. Checkerboard Field
    row = np.array([1.0, -1.0] * 8)
    checker_data = np.array([row if i % 2 == 0 else -row for i in range(16)])
    res_checker, _, _, _ = run_pipeline(checker_data, k=DEFAULT_K, n_permutations=DEFAULT_PERMUTATIONS, seed=DEFAULT_SEED)
    save_fixture("checkerboard_16x16.npz", checker_data, res_checker)

    # 4. Gradient Field
    grad_data = np.tile(np.linspace(0, 10, 16), (16, 1))
    res_grad, _, _, _ = run_pipeline(grad_data, k=DEFAULT_K, n_permutations=DEFAULT_PERMUTATIONS, seed=DEFAULT_SEED)
    save_fixture("gradient_16x16.npz", grad_data, res_grad)

    print(f"=========================================================================\n")


if __name__ == "__main__":
    generate_all_fixtures()