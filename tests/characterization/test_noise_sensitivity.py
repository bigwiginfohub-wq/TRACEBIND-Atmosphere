import sys
from pathlib import Path
import numpy as np

# Force project root onto sys.path (C:\TRACEBIND-Atmosphere)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.test_domain_validation import run_pipeline
from tests.test_synthetic_autocorrelation import generate_exponential_grf

def run_noise_sensitivity_study(num_realizations=20, grid_size=128):
    """
    Evaluates metric degradation under increasing additive zero-mean Gaussian noise.
    Noise levels range from 0% to 50% of the underlying signal standard deviation.
    """
    print("=" * 100)
    print(f" PHASE 3.3: NOISE SENSITIVITY & PERTURBATION ROBUSTNESS")
    print(f" Grid: {grid_size}x{grid_size} | Realizations per Noise Level: {num_realizations}")
    print("=" * 100)

    noise_fractions = [0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    results = {'noise_frac': [], 'r_mean': [], 'r_sd': [], 'r_drop_pct': []}

    # Reference signal with zero noise (fixed 10% physical correlation length)
    l_px = grid_size * 0.10  

    print(f"{'Noise Fraction (σ_n / σ_s)':<28} | {'TRACEBIND R (Mean ± SD)':<28} | {'Relative Signal Drop (%)':<22}")
    print("-" * 100)

    r_clean_baseline = None

    for n_frac in noise_fractions:
        r_list = []
        
        for seed in range(200, 200 + num_realizations):
            # 1. Clean continuous Gaussian Random Field
            clean_field = generate_exponential_grf(
                shape=(grid_size, grid_size),
                correlation_length=l_px,
                seed=seed
            )
            
            # 2. Add zero-mean Gaussian noise scaled to signal standard deviation
            sigma_signal = np.std(clean_field)
            rng = np.random.default_rng(seed)
            noise = rng.normal(0, n_frac * sigma_signal, size=clean_field.shape)
            noisy_field = clean_field + noise

            # 3. Evaluate through pipeline
            res, _, _, _ = run_pipeline(noisy_field, k=4, n_permutations=20, seed=seed)
            r_list.append(res.r_observed)

        r_arr = np.array(r_list)
        mean_r, sd_r = np.mean(r_arr), np.std(r_arr)

        if n_frac == 0.00:
            r_clean_baseline = mean_r
            drop_pct = 0.0
        else:
            drop_pct = ((r_clean_baseline - mean_r) / r_clean_baseline) * 100.0

        results['noise_frac'].append(n_frac)
        results['r_mean'].append(mean_r)
        results['r_sd'].append(sd_r)
        results['r_drop_pct'].append(drop_pct)

        print(f"{n_frac * 100:>5.1f}% {'(' + str(n_frac) + ')':<21} | {mean_r:.4f} ± {sd_r:.4f}{'':<14} | {drop_pct:>6.2f}%")

    print("-" * 100)
    return results

if __name__ == "__main__":
    run_noise_sensitivity_study()