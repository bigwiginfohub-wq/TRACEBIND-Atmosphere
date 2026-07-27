import sys
from pathlib import Path

# Force project root onto sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tests.test_synthetic_autocorrelation import generate_exponential_grf


def compute_2d_autocovariance(field):
    """Computes normalized 2D auto-covariance surface using 2D FFT."""
    h, w = field.shape
    f_centered = field - np.mean(field)
    
    # Zero-padded FFT2 for linear covariance
    fft_f = np.fft.fft2(f_centered, s=(2 * h, 2 * w))
    power_spec = np.abs(fft_f) ** 2
    autocov = np.fft.ifft2(power_spec).real
    
    # Shift center to (0,0) and normalize by sample variance
    autocov = np.fft.fftshift(autocov) / (h * w * np.var(field))
    
    # Crop to centered box [-h//2, h//2] x [-w//2, w//2]
    cy, cx = h, w
    return autocov[cy - h // 2 : cy + h // 2, cx - w // 2 : cx + w // 2]


# --- ESTIMATOR 1: Model-Free 1/e Crossing ---
def estimate_length_1over_e(profile):
    """Linearly interpolates r where C(r) = 1/e ~ 0.3679."""
    n = len(profile)
    mid = n // 2
    y = profile[mid:]
    r = np.arange(len(y))
    
    target = 1.0 / np.e
    below_idx = np.where(y < target)[0]
    if len(below_idx) == 0:
        return np.nan
        
    idx2 = below_idx[0]
    if idx2 == 0:
        return 0.1
        
    idx1 = idx2 - 1
    r1, r2 = r[idx1], r[idx2]
    y1, y2 = y[idx1], y[idx2]
    
    return float(r1 + (target - y1) * (r2 - r1) / (y2 - y1))


# --- ESTIMATOR 2: Log-Linear OLS ---
def estimate_length_ols(profile, min_threshold=0.10):
    """Fits log(C(r)) = ln(A) - r/l via OLS over domain C(r) >= min_threshold."""
    n = len(profile)
    mid = n // 2
    y = profile[mid:]
    r = np.arange(len(y))
    
    mask = y >= min_threshold
    if np.sum(mask) < 3:
        mask = np.zeros_like(y, dtype=bool)
        mask[:3] = True
        
    r_fit = r[mask]
    y_fit = np.maximum(y[mask], 1e-6)
    
    try:
        poly = np.polyfit(r_fit, np.log(y_fit), deg=1)
        slope = poly[0]
        return -1.0 / slope if slope < 0 else np.nan
    except Exception:
        return np.nan


# --- ESTIMATOR 3: 2-Parameter Non-Linear Fit ---
def estimate_length_nonlinear(profile, min_threshold=0.10):
    """Fits A * exp(-r/l) with floating amplitude A."""
    n = len(profile)
    mid = n // 2
    y = profile[mid:]
    r = np.arange(len(y))
    
    mask = y >= min_threshold
    if np.sum(mask) < 3:
        return np.nan
        
    r_fit = r[mask]
    y_fit = y[mask]
    
    def exp_model(x, A, l):
        return A * np.exp(-x / l)
        
    try:
        popt, _ = curve_fit(exp_model, r_fit, y_fit, p0=[1.0, 12.0], bounds=([0.1, 0.1], [2.0, 64.0]))
        return popt[1]
    except Exception:
        return np.nan


# --- ESTIMATOR 4: Orientation Recovery via Structure Tensor ---
def estimate_orientation_angle(field):
    """Estimates dominant principal axis orientation angle in degrees [-90, 90]."""
    gy, gx = np.gradient(field)
    j11 = np.sum(gx * gx)
    j22 = np.sum(gy * gy)
    j12 = np.sum(gx * gy)
    
    # Dominant direction perpendicular to maximum gradient
    theta_rad = 0.5 * np.arctan2(2 * j12, j11 - j22)
    theta_deg = np.degrees(theta_rad)
    return theta_deg


def compute_bootstrap_ci(data_x, data_y, num_bootstraps=1000, ci=95):
    """Computes non-parametric bootstrap confidence interval for ratio of means."""
    boot_ratios = []
    n_x, n_y = len(data_x), len(data_y)
    
    if n_x < 3 or n_y < 3:
        return np.nan, np.nan
        
    rng = np.random.default_rng(42)
    for _ in range(num_bootstraps):
        sample_x = rng.choice(data_x, size=n_x, replace=True)
        sample_y = rng.choice(data_y, size=n_y, replace=True)
        boot_ratios.append(np.mean(sample_x) / np.mean(sample_y))
        
    lower_p = (100 - ci) / 2.0
    upper_p = 100 - lower_p
    return np.percentile(boot_ratios, lower_p), np.percentile(boot_ratios, upper_p)


def run_comprehensive_benchmark(num_realizations=30, grid_size=128):
    print("=" * 125)
    print(f" COMPREHENSIVE CORRELATION & ORIENTATION BENCHMARK (GRID: {grid_size}x{grid_size}, N = {num_realizations})")
    print("=" * 125)
    
    target_ratios = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
    base_lx = 12.0
    
    # Store comparative results
    benchmark_data = {
        '1e': {'ratios': [], 'bias': [], 'rel_bias': [], 'rmse': [], 'ci_low': [], 'ci_high': []},
        'ols': {'ratios': [], 'bias': [], 'rel_bias': [], 'rmse': [], 'ci_low': [], 'ci_high': []},
        'nl': {'ratios': [], 'bias': [], 'rel_bias': [], 'rmse': [], 'ci_low': [], 'ci_high': []}
    }
    
    print(f"\n{'Target R':<8} | {'Estimator':<12} | {'Ratio of Means':<15} | {'Bias':<10} | {'Rel Bias (%)':<12} | {'RMSE':<10} | {'95% Bootstrap CI':<20}")
    print("-" * 125)
    
    for r in target_ratios:
        e_lx, e_ly, ols_lx, ols_ly, nl_lx, nl_ly = [], [], [], [], [], []
        
        for seed in range(100, 100 + num_realizations):
            field = generate_exponential_grf(
                shape=(grid_size, grid_size), 
                correlation_length=base_lx, 
                aspect_ratio=r, 
                theta_deg=0.0, 
                seed=seed
            )
            
            autocov = compute_2d_autocovariance(field)
            h, w = autocov.shape
            mid_y, mid_x = h // 2, w // 2
            
            # Spatial Band Averaging (+/- 3 rows/cols = 7-pixel band)
            row_p = np.mean(autocov[mid_y - 3 : mid_y + 4, :], axis=0)
            col_p = np.mean(autocov[:, mid_x - 3 : mid_x + 4], axis=1)
            
            # Extract estimates
            e_lx.append(estimate_length_1over_e(row_p))
            e_ly.append(estimate_length_1over_e(col_p))
            ols_lx.append(estimate_length_ols(row_p))
            ols_ly.append(estimate_length_ols(col_p))
            nl_lx.append(estimate_length_nonlinear(row_p))
            nl_ly.append(estimate_length_nonlinear(col_p))
            
        # NaN filtering
        methods = {
            '1e': (np.array(e_lx)[np.isfinite(e_lx)], np.array(e_ly)[np.isfinite(e_ly)]),
            'ols': (np.array(ols_lx)[np.isfinite(ols_lx)], np.array(ols_ly)[np.isfinite(ols_ly)]),
            'nl': (np.array(nl_lx)[np.isfinite(nl_lx)], np.array(nl_ly)[np.isfinite(nl_ly)])
        }
        
        for key, (lx_arr, ly_arr) in methods.items():
            ratio_m = np.mean(lx_arr) / np.mean(ly_arr)
            individual_ratios = lx_arr / ly_arr
            bias = ratio_m - r
            rel_bias = (bias / r) * 100.0
            rmse = np.sqrt(np.mean((individual_ratios - r) ** 2))
            ci_l, ci_h = compute_bootstrap_ci(lx_arr, ly_arr)
            
            benchmark_data[key]['ratios'].append(ratio_m)
            benchmark_data[key]['bias'].append(bias)
            benchmark_data[key]['rel_bias'].append(rel_bias)
            benchmark_data[key]['rmse'].append(rmse)
            benchmark_data[key]['ci_low'].append(ci_l)
            benchmark_data[key]['ci_high'].append(ci_h)
            
            est_name = {'1e': '1/e Crossing', 'ols': 'Log-OLS', 'nl': 'Nonlinear'}[key]
            print(f"{r:<8.1f} | {est_name:<12} | {ratio_m:<15.2f} | {bias:<+10.2f} | {rel_bias:<+12.1f} | {rmse:<10.2f} | [{ci_l:.2f}, {ci_h:.2f}]")
        print("-" * 125)

    # --- ORIENTATION RECOVERY SWEEP ---
    print("\n" + "=" * 125)
    print(" ORIENTATION RECOVERY BENCHMARK (Aspect Ratio = 4:1)")
    print("=" * 125)
    test_angles = [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0]
    print(f"\n{'Target Theta (deg)':<20} | {'Estimated Mean Theta (deg)':<28} | {'Absolute Angular Error (deg)':<30}")
    print("-" * 125)
    
    for theta_target in test_angles:
        angle_estimates = []
        for seed in range(200, 200 + num_realizations):
            field = generate_exponential_grf(
                shape=(grid_size, grid_size),
                correlation_length=base_lx,
                aspect_ratio=4.0,
                theta_deg=theta_target,
                seed=seed
            )
            angle_estimates.append(estimate_orientation_angle(field))
            
        angle_arr = np.array(angle_estimates)
        mean_angle = np.abs(np.mean(angle_arr))
        ang_err = abs(mean_angle - theta_target)
        print(f"{theta_target:<20.1f} | {mean_angle:<28.2f} | {ang_err:<30.2f}")
    print("=" * 125)

    # --- PLOTTING ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)
    r_arr = np.array(target_ratios)
    
    # Plot 1: Recovery + 95% Bootstrap Confidence Bands
    colors = {'1e': '#1f77b4', 'ols': '#2ca02c', 'nl': '#ff7f0e'}
    labels = {'1e': '1/e Crossing', 'ols': 'Log-Linear OLS', 'nl': 'Nonlinear Fit'}
    
    for key in ['1e', 'ols', 'nl']:
        meas = np.array(benchmark_data[key]['ratios'])
        ci_l = np.array(benchmark_data[key]['ci_low'])
        ci_h = np.array(benchmark_data[key]['ci_high'])
        
        ax1.plot(r_arr, meas, 'o-', color=colors[key], linewidth=2, label=labels[key])
        ax1.fill_between(r_arr, ci_l, ci_h, color=colors[key], alpha=0.15)
        
    ax1.plot(r_arr, r_arr, 'k--', label='Ideal Recovery (y=x)')
    ax1.set_title("A. Estimator Anisotropy Recovery (with 95% Bootstrap Bands)", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Target Aspect Ratio")
    ax1.set_ylabel("Measured Aspect Ratio")
    ax1.set_xticks(target_ratios)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()
    
    # Plot 2: Relative Bias Comparison
    for key in ['1e', 'ols', 'nl']:
        rel_b = np.array(benchmark_data[key]['rel_bias'])
        ax2.plot(r_arr, rel_b, 's-', color=colors[key], linewidth=2, label=labels[key])
        
    ax2.axhline(0, color='black', linestyle=':', linewidth=1)
    ax2.set_title("B. Relative Bias Across Aspect Ratios (%)", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Target Aspect Ratio")
    ax2.set_ylabel("Relative Bias (%)")
    ax2.set_xticks(target_ratios)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend()
    
    save_path = Path(__file__).parent / "benchmark_anisotropy_characterization.png"
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\n[✓] Publication-grade benchmark plot saved to: {save_path}\n")


if __name__ == "__main__":
    run_comprehensive_benchmark(num_realizations=30, grid_size=128)