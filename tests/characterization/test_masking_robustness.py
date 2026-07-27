import sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

# Force project root onto sys.path (C:\TRACEBIND-Atmosphere)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.test_domain_validation import run_pipeline
from tests.test_synthetic_autocorrelation import generate_exponential_grf


def run_masking_robustness_study(num_realizations=20, grid_size=128):
    """
    Evaluates metric behavior under increasing fractions of random missing data / spatial masking.
    Missing observations are excluded from graph construction (drop_nan=True).
    """
    print("=" * 115)
    print(" PHASE 3.4: SPATIAL MASKING & MISSING DATA ROBUSTNESS")
    print(f" Grid: {grid_size}x{grid_size} | Realizations per Mask Level: {num_realizations}")
    print("=" * 115)

    requested_mask_fractions = [0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    results = {
        'req_mask_frac': [], 
        'actual_retained_pct': [], 
        'r_mean': [], 
        'r_sd': [], 
        'r_drop_pct': []
    }

    l_px = grid_size * 0.10  # Fixed 10% physical correlation length

    print(f"{'Requested Mask (%)':<20} | {'Actual Retained (%)':<22} | {'TRACEBIND R (Mean ± SD)':<28} | {'Relative Drop (%)':<20}")
    print("-" * 115)

    r_clean_baseline = None
    all_mask_levels = []
    all_r_scores = []

    for m_frac in requested_mask_fractions:
        r_list = []
        retained_ratios = []
        
        for seed in range(300, 300 + num_realizations):
            # 1. Clean continuous Gaussian Random Field
            clean_field = generate_exponential_grf(
                shape=(grid_size, grid_size),
                correlation_length=l_px,
                seed=seed
            )
            
            # 2. Missing observations marked as NaN
            masked_field = clean_field.copy()
            if m_frac > 0.0:
                rng = np.random.default_rng(seed)
                mask = rng.random(size=clean_field.shape) < m_frac
                masked_field[mask] = np.nan

            actual_retained = np.mean(np.isfinite(masked_field)) * 100.0
            retained_ratios.append(actual_retained)

            # 3. Evaluate through pipeline (explicit drop_nan=True excludes missing observations)
            res, _, _, _ = run_pipeline(masked_field, k=4, n_permutations=20, seed=seed, drop_nan=True)
            r_list.append(res.r_observed)
            
            all_mask_levels.append(m_frac)
            all_r_scores.append(res.r_observed)

        r_arr = np.array(r_list)
        mean_r, sd_r = np.mean(r_arr), np.std(r_arr)
        avg_retained_pct = np.mean(retained_ratios)

        if m_frac == 0.00:
            r_clean_baseline = mean_r
            drop_pct = 0.0
        else:
            drop_pct = ((r_clean_baseline - mean_r) / r_clean_baseline) * 100.0

        results['req_mask_frac'].append(m_frac)
        results['actual_retained_pct'].append(avg_retained_pct)
        results['r_mean'].append(mean_r)
        results['r_sd'].append(sd_r)
        results['r_drop_pct'].append(drop_pct)

        print(f"{m_frac * 100:>5.1f}% {'(' + str(m_frac) + ')':<13} | {avg_retained_pct:>6.2f}%{'':<14} | {mean_r:.4f} ± {sd_r:.4f}{'':<14} | {drop_pct:>6.2f}%")

    # Statistical summary: Spearman rank correlation over all N realizations
    total_obs = len(all_mask_levels)
    rho, p_val = spearmanr(all_mask_levels, all_r_scores)
    print("-" * 115)
    print(f"Spearman Rank Correlation (Mask Fraction vs. R): ρ = {rho:.4f} (p = {p_val:.4e}, N = {total_obs})")
    print("-" * 115)

    return results


if __name__ == "__main__":
    run_masking_robustness_study()