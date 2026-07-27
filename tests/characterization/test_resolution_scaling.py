import sys
import time
import tracemalloc
from pathlib import Path

# Force project root onto sys.path (C:\TRACEBIND-Atmosphere)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt

# Import run_pipeline directly from the validated domain test suite
from tests.test_domain_validation import run_pipeline
from tests.test_synthetic_autocorrelation import generate_exponential_grf


def compute_classic_spatial_stats(field):
    """
    Computes Moran's I and Geary's C using a standard 4-neighbor spatial weight structure.
    """
    h, w = field.shape
    n = h * w
    z = field - np.mean(field)
    s2 = np.sum(z**2)
    
    if s2 == 0:
        return 0.0, 1.0

    # Neighbor shift differences (horizontal + vertical)
    diff_x = np.diff(field, axis=1)
    diff_y = np.diff(field, axis=0)
    geary_num = np.sum(diff_x**2) + np.sum(diff_y**2)
    
    # Total edge weights W
    w_sum = 2 * (w * (h - 1) + h * (w - 1))
    
    geary_c = ((n - 1) * geary_num) / (2 * w_sum * s2)
    
    # Moran's I
    prod_x = z[:, :-1] * z[:, 1:]
    prod_y = z[:-1, :] * z[1:, :]
    moran_num = np.sum(prod_x) + np.sum(prod_y)
    
    moran_i = (n * 2 * moran_num) / (w_sum * s2)
    
    return moran_i, geary_c


def run_resolution_scaling_study(num_realizations=20, n_permutations=20):
    print("=" * 125)
    print(f" PHASE 3.2: RESOLUTION SCALING & DISCRETIZATION STUDY")
    print(f" Engine: Validated TRACEBIND run_pipeline() | Permutations = {n_permutations} | Realizations = {num_realizations}")
    print("=" * 125)
    
    # Grid sizes (512x512 serves as the reference baseline R_512)
    resolutions = [16, 32, 64, 128, 256, 512]
    
    # Explicit domain fraction (10% of domain width)
    physical_corr_fraction = 0.10  
    
    results = {
        'res': [], 'l_px': [], 'grid_h': [],
        'tb_r_mean': [], 'tb_r_sd': [], 'tb_cv': [],
        'moran_mean': [], 'moran_sd': [],
        'geary_mean': [], 'geary_sd': [],
        'runtime_ms': [], 'peak_mem_mb': []
    }
    
    print(f"\n{'Grid Size':<10} | {'l_px (px)':<10} | {'TRACEBIND R (Mean ± SD)':<28} | {'% CV':<8} | {'Moran I':<10} | {'Time (ms)':<10} | {'Peak Mem (MB)':<12}")
    print("-" * 125)
    
    tracemalloc.start()
    
    for N in resolutions:
        l_px = N * physical_corr_fraction
        grid_h = 1.0 / N
        
        tb_r_list = []
        moran_list = []
        geary_list = []
        time_list = []
        mem_list = []
        
        for seed in range(100, 100 + num_realizations):
            # 1. Generate field with invariant physical correlation fraction
            field = generate_exponential_grf(
                shape=(N, N),
                correlation_length=l_px,
                aspect_ratio=1.0,
                theta_deg=0.0,
                seed=seed
            )
            
            # 2. Benchmark full TRACEBIND pipeline execution & memory footprint
            tracemalloc.reset_peak()
            t0 = time.perf_counter()
            
            # run_pipeline returns (result, collection, graph, roi)
            result_obj, _, _, _ = run_pipeline(
                field,
                k=4,
                n_permutations=n_permutations,
                seed=seed
            )
            
            t1 = time.perf_counter()
            _, peak_bytes = tracemalloc.get_traced_memory()
            
            r_val = result_obj.r_observed
            
            # 3. Compute reference classical stats
            m_i, g_c = compute_classic_spatial_stats(field)
            
            tb_r_list.append(float(r_val))
            moran_list.append(m_i)
            geary_list.append(g_c)
            time_list.append((t1 - t0) * 1000.0)  # ms
            mem_list.append(peak_bytes / (1024 * 1024))  # MB
            
        r_arr = np.array(tb_r_list)
        m_arr = np.array(moran_list)
        g_arr = np.array(geary_list)
        t_arr = np.array(time_list)
        mem_arr = np.array(mem_list)
        
        r_mean, r_sd = np.mean(r_arr), np.std(r_arr)
        cv_pct = (r_sd / r_mean) * 100.0 if r_mean != 0 else 0.0
        
        results['res'].append(N)
        results['l_px'].append(l_px)
        results['grid_h'].append(grid_h)
        results['tb_r_mean'].append(r_mean)
        results['tb_r_sd'].append(r_sd)
        results['tb_cv'].append(cv_pct)
        results['moran_mean'].append(np.mean(m_arr))
        results['moran_sd'].append(np.std(m_arr))
        results['geary_mean'].append(np.mean(g_arr))
        results['geary_sd'].append(np.std(g_arr))
        results['runtime_ms'].append(np.mean(t_arr))
        results['peak_mem_mb'].append(np.mean(mem_arr))
        
        tb_str = f"{r_mean:.4f} ± {r_sd:.4f}"
        print(f"{f'{N}x{N}':<10} | {l_px:<10.1f} | {tb_str:<28} | {cv_pct:<8.2f} | {np.mean(m_arr):<10.4f} | {np.mean(t_arr):<10.2f} | {np.mean(mem_arr):<12.2f}")
        
    tracemalloc.stop()
    print("-" * 125)
    
    # 4. Explicit Convergence Analysis relative to Reference Baseline (512x512)
    r_512 = results['tb_r_mean'][-1]
    rel_errors = [abs(r - r_512) / abs(r_512) for r in results['tb_r_mean']]
    results['rel_error'] = rel_errors
    
    print("\n[Grid Convergence Summary Relative to Reference Resolution (512x512)]")
    for N, r_m, err in zip(results['res'], results['tb_r_mean'], rel_errors):
        print(f"  Grid {N:>3}x{N:<3} | Mean R = {r_m:.4f} | Relative Error vs R_512 = {err * 100:.2f}%")
        
    # 5. Calculate Empirical Convergence Order p: E(h) ~ C * h^p
    h_vals = np.array(results['grid_h'][:-1])
    err_vals = np.array(rel_errors[:-1])
    
    valid_idx = err_vals > 0
    if np.sum(valid_idx) >= 2:
        log_h = np.log(h_vals[valid_idx])
        log_e = np.log(err_vals[valid_idx])
        p_order, log_C = np.polyfit(log_h, log_e, 1)
        print(f"\n  [✓] Empirical Convergence Order: p = {p_order:.2f} (Error E(h) ∝ h^{p_order:.2f})")
    else:
        p_order = None
        
    print("-" * 125)

    # --- PLOTTING ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    res_arr = np.array(results['res'])
    r_means = np.array(results['tb_r_mean'])
    r_sds = np.array(results['tb_r_sd'])
    
    # Plot A: TRACEBIND R Grid Convergence & Asymptote
    ax1 = axes[0, 0]
    ax1.errorbar(res_arr, r_means, yerr=r_sds, fmt='o-', color='#1f77b4', ecolor='#1f77b4', elinewidth=1.5, capsize=4, label='TRACEBIND R (Mean ± 1 SD)')
    ax1.axhline(r_512, color='red', linestyle='--', alpha=0.7, label=f'Reference Baseline (512x512: {r_512:.4f})')
    ax1.set_xscale('log', base=2)
    ax1.set_xticks(resolutions)
    ax1.set_xticklabels([f"{n}x{n}" for n in resolutions])
    ax1.set_title("A. TRACEBIND R Metric Resolution Convergence", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Grid Resolution (Fixed Physical Domain)")
    ax1.set_ylabel("TRACEBIND Metric R")
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()
    
    # Plot B: Discretization Relative Error
    ax2 = axes[0, 1]
    ax2.plot(res_arr[:-1], np.array(rel_errors[:-1]) * 100, 's-', color='#d62728', linewidth=2, label='Measured Error')
    if p_order is not None:
        ax2.plot(res_arr[:-1], (np.exp(log_C) * (1.0 / res_arr[:-1])**p_order) * 100, 'k--', alpha=0.7, label=f'Fit: $O(h^{{{p_order:.2f}}})$')
    ax2.set_xscale('log', base=2)
    ax2.set_yscale('log')
    ax2.set_xticks(resolutions[:-1])
    ax2.set_xticklabels([f"{n}x{n}" for n in resolutions[:-1]])
    ax2.set_title("B. Discretization Relative Error |R_N - R_512| / R_512 (%)", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Grid Resolution")
    ax2.set_ylabel("Relative Error (%)")
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend()
    
    # Plot C: Statistical Stability (% CV)
    ax3 = axes[1, 0]
    ax3.plot(res_arr, results['tb_cv'], 'd-', color='#2ca02c', linewidth=2)
    ax3.set_xscale('log', base=2)
    ax3.set_xticks(resolutions)
    ax3.set_xticklabels([f"{n}x{n}" for n in resolutions])
    ax3.set_title("C. Statistical Stability (% CV = SD / Mean)", fontsize=11, fontweight='bold')
    ax3.set_xlabel("Grid Resolution")
    ax3.set_ylabel("Coefficient of Variation (%)")
    ax3.grid(True, linestyle='--', alpha=0.5)
    
    # Plot D: Execution Runtime & Memory Scaling
    ax4 = axes[1, 1]
    ax4_mem = ax4.twinx()
    
    l1 = ax4.plot(res_arr, results['runtime_ms'], 'o-', color='#9467bd', linewidth=2, label='Runtime (ms)')
    l2 = ax4_mem.plot(res_arr, results['peak_mem_mb'], 'd--', color='#8c564b', linewidth=2, label='Peak Memory (MB)')
    
    ax4.set_xscale('log', base=2)
    ax4.set_yscale('log')
    ax4_mem.set_yscale('log')
    ax4.set_xticks(resolutions)
    ax4.set_xticklabels([f"{n}x{n}" for n in resolutions])
    
    ax4.set_title("D. Empirical Resource Footprint (Tracemalloc)", fontsize=11, fontweight='bold')
    ax4.set_xlabel("Grid Resolution")
    ax4.set_ylabel("Execution Time (ms)", color='#9467bd')
    ax4_mem.set_ylabel("Peak Memory Allocated (MB)", color='#8c564b')
    ax4.grid(True, linestyle='--', alpha=0.5)
    
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='upper left')
    
    save_path = Path(__file__).parent / "resolution_scaling_characterization.png"
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\n[✓] Phase 3.2 Resolution Scaling plot saved to: {save_path}\n")

def plot_log_log_grid_convergence(results, save_path="grid_convergence_loglog.png"):
    """
    Plots log(Observed Relative Error) vs log(Grid Spacing h) using standard
    numerical analysis conventions and terminology.
    """
    # Exclude the finest reference grid (512x512) where error is 0 by definition
    h_vals = np.array(results['grid_h'][:-1])
    rel_errors = np.array(results['rel_error'][:-1])
    
    # Log10 transformation for linear regression
    log_h = np.log10(h_vals)
    log_e = np.log10(rel_errors)
    p_order, log10_C = np.polyfit(log_h, log_e, 1)
    
    # Generate line fit
    h_fit = np.linspace(min(h_vals), max(h_vals), 100)
    fit_err = (10**log10_C) * (h_fit**p_order)

    plt.figure(figsize=(7, 5.5), dpi=300)
    
    # Observed points
    plt.loglog(h_vals, rel_errors, 'so', markersize=7, color='#d62728', 
               label='Observed Relative Error $E(h)$')
    
    # Least-Squares Fit
    plt.loglog(h_fit, fit_err, 'k--', linewidth=1.5, 
               label=f'Least-Squares Fit ($p = {p_order:.2f}$)')
    
    plt.title("Grid-Refinement Log-Log Convergence", fontsize=11, fontweight='bold')
    plt.xlabel("Grid Spacing $h = 1/N$ (Log Scale)", fontsize=10)
    plt.ylabel("Observed Relative Error $|R_N - R_{512}| / R_{512}$ (Log Scale)", fontsize=10)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=10, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[✓] Numerical convergence plot saved: {save_path}")    


if __name__ == "__main__":
    run_resolution_scaling_study(num_realizations=20, n_permutations=20)