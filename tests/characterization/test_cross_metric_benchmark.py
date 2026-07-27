import sys
import argparse
from pathlib import Path

# Force project root onto sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from esda.moran import Moran
from esda.geary import Geary
from libpysal.weights import W, WSP

from tests.test_domain_validation import run_pipeline
from tests.test_synthetic_autocorrelation import generate_exponential_grf

# Standardized Permutation and Realization Constants
DEFAULT_PERMUTATIONS = 50
PUBLICATION_PERMUTATIONS = 999
DEFAULT_REALIZATIONS = 10
PUBLICATION_REALIZATIONS = 50


def parse_args():
    parser = argparse.ArgumentParser(description="TRACEBIND Cross-Metric Scientific Benchmark")
    parser.add_argument(
        "--publication", 
        action="store_true", 
        help=f"Run publication-grade benchmark ({PUBLICATION_REALIZATIONS} realizations, {PUBLICATION_PERMUTATIONS} permutations)"
    )
    return parser.parse_args()


def get_graph_node_neighbors(graph, node_id: int) -> list[int]:
    """Unified accessor for neighborhood lookup across varying graph interface versions."""
    if hasattr(graph, 'get_neighbors'):
        raw_neighs = graph.get_neighbors(node_id)
    elif hasattr(graph, 'neighbours'):
        raw_neighs = graph.neighbours(node_id) if callable(graph.neighbours) else graph.neighbours[node_id]
    elif hasattr(graph, 'neighbors'):
        raw_neighs = graph.neighbors(node_id) if callable(graph.neighbors) else graph.neighbors[node_id]
    else:
        raise AttributeError("Graph object does not expose a recognized neighbors accessor.")
    
    flat_indices = np.asarray(raw_neighs, dtype=int).ravel().tolist()
    return [int(n) for n in flat_indices if int(n) != int(node_id)]


def get_graph_node_count(graph) -> int:
    """Unified accessor for graph node count."""
    if hasattr(graph, 'n_nodes'):
        return graph.n_nodes
    if hasattr(graph, 'num_nodes'):
        return graph.num_nodes
    raise AttributeError("Graph object does not expose n_nodes or num_nodes.")


def build_pysal_weights_from_graph(graph, transform: str = 'B', symmetrize: bool = True) -> W:
    """Converts TRACEBIND NeighborhoodGraph into a PySAL sparse W object."""
    n_nodes = get_graph_node_count(graph)
    neighbors_dict = {}
    
    for i in range(n_nodes):
        node_id = int(i)
        neighbors_dict[node_id] = get_graph_node_neighbors(graph, node_id)

    weights_dict = {
        int(i): [1.0] * len(neighbors_dict[int(i)]) for i in range(n_nodes)
    }
    
    w = W(neighbors_dict, weights_dict, silence_warnings=True)
    
    if symmetrize:
        sparse_sym = w.sparse + w.sparse.T
        sparse_sym.data = np.ones_like(sparse_sym.data, dtype=float)
        w = WSP(sparse_sym).to_W(silence_warnings=True)

    w.transform = transform
    return w


def get_graph_mean_degree(graph) -> float:
    """Safely extracts average external degree from neighborhood graph."""
    if hasattr(graph, 'mean_degree'):
        return float(graph.mean_degree)
    
    n_nodes = get_graph_node_count(graph)
    degrees = []
    for i in range(n_nodes):
        degrees.append(len(get_graph_node_neighbors(graph, i)))
    return float(np.mean(degrees))


def run_phase_2a_pysal_cross_metric_benchmark(n_permutations: int):
    """Phase 2A: Standardized Cross-Metric Benchmark comparing TRACEBIND against PySAL ESDA."""
    print("\n" + "="*95)
    print(" PHASE 2A: STANDARDIZED PYSAL CROSS-METRIC BENCHMARK (TRACEBIND vs. ESDA Moran/Geary)")
    print("="*95)
    
    datasets = []
    rng = np.random.default_rng(42)
    datasets.append(("White Noise (l=0.5)", rng.normal(size=(16, 16))))
    
    for l in [1.5, 3.0, 6.0, 12.0]:
        field = generate_exponential_grf(shape=(16, 16), correlation_length=l, seed=42)
        datasets.append((f"GRF (l={l:4.1f})", field))
        
    cb = np.indices((16, 16)).sum(axis=0) % 2
    datasets.append(("Checkerboard", cb.astype(np.float64)))
    
    x, y = np.meshgrid(np.linspace(0, 1, 16), np.linspace(0, 1, 16))
    datasets.append(("Gradient", x + y))
    
    results_summary = []
    for transform in ['B', 'R']:
        t_label = "Binary Weighting ('B')" if transform == 'B' else "Row-Standardized ('R')"
        print(f"\n--- Weight Transform Scheme: {t_label} ---")
        print(f"{'Field / Pattern':<20} | {'TRACEBIND R':<10} {'Z(R)':<8} | {'Moran I':<10} {'Z(I)':<8} | {'Geary C':<10} {'Z(C)':<8}")
        print("-" * 95)
        
        for name, data in datasets:
            res, collection, graph, _ = run_pipeline(data, k=4, n_permutations=n_permutations, seed=42)
            values = collection.point_cloud.values
            
            pysal_w = build_pysal_weights_from_graph(graph, transform=transform, symmetrize=True)
            mi = Moran(values, pysal_w, permutations=n_permutations)
            gc = Geary(values, pysal_w, permutations=n_permutations)
            
            print(
                f"{name:<20} | "
                f"{res.r_observed:10.4f} {res.z_score:8.2f} | "
                f"{mi.I:10.4f} {mi.z_sim:8.2f} | "
                f"{gc.C:10.4f} {gc.z_sim:8.2f}"
            )
            results_summary.append((res.r_observed, mi.I, gc.C))
        print("-" * 95)
    return results_summary


def run_grf_correlation_tracking_analysis(n_realizations: int, n_permutations: int):
    """Quantifies metric tracking against spatial correlation length ℓ across realizations."""
    print("\n" + "="*95)
    print(" QUANTITATIVE METRIC TRACKING VS. SPATIAL CORRELATION LENGTH (ℓ)")
    print("="*95)
    
    correlation_lengths = [0.5, 1.0, 1.5, 2.5, 4.0, 6.0, 8.0, 12.0]
    stats_r, stats_mi, stats_gc = [], [], []
    
    print(f"\nEvaluating GRFs across {len(correlation_lengths)} scale lengths ({n_realizations} realizations each):\n")
    print(f"{'Scale (ℓ)':<10} | {'TRACEBIND R (Mean ± Std)':<25} | {'Moran I (Mean ± Std)':<22} | {'Geary C (Mean ± Std)':<20}")
    print("-" * 95)
    
    for l in correlation_lengths:
        r_ens, mi_ens, gc_ens = [], [], []
        for seed in range(100, 100 + n_realizations):
            field = generate_exponential_grf(shape=(16, 16), correlation_length=l, seed=seed)
            res, collection, graph, _ = run_pipeline(field, k=4, n_permutations=n_permutations, seed=seed)
            values = collection.point_cloud.values
            
            pysal_w = build_pysal_weights_from_graph(graph, transform='R', symmetrize=True)
            mi = Moran(values, pysal_w, permutations=0)
            gc = Geary(values, pysal_w, permutations=0)
            
            r_ens.append(res.r_observed)
            mi_ens.append(mi.I)
            gc_ens.append(gc.C)
            
        m_r, std_r = np.mean(r_ens), np.std(r_ens)
        m_mi, std_mi = np.mean(mi_ens), np.std(mi_ens)
        m_gc, std_gc = np.mean(gc_ens), np.std(gc_ens)
        
        stats_r.append((m_r, std_r))
        stats_mi.append((m_mi, std_mi))
        stats_gc.append((m_gc, std_gc))
        
        print(f"ℓ = {l:<6.1f} | {m_r:8.4f} ± {std_r:<12.4f} | {m_mi:8.4f} ± {std_mi:<10.4f} | {m_gc:8.4f} ± {std_gc:<8.4f}")
        
    print("-" * 95)
    
    r_means = [s[0] for s in stats_r]
    moran_means = [s[0] for s in stats_mi]
    geary_means = [s[0] for s in stats_gc]
    
    p_r, s_r = pearsonr(correlation_lengths, r_means)[0], spearmanr(correlation_lengths, r_means)[0]
    p_mi, s_mi = pearsonr(correlation_lengths, moran_means)[0], spearmanr(correlation_lengths, moran_means)[0]
    p_gc, s_gc = pearsonr(correlation_lengths, geary_means)[0], spearmanr(correlation_lengths, geary_means)[0]
    
    aligned_geary = [1.0 - c for c in geary_means]
    p_gc_align, s_gc_align = pearsonr(correlation_lengths, aligned_geary)[0], spearmanr(correlation_lengths, aligned_geary)[0]
    
    print("\nMetric Correlation Tracking Summary:")
    print(f"  • TRACEBIND R   : Spearman Rank = {s_r:.4f} | Pearson r = {p_r:.4f}")
    print(f"  • Moran's I     : Spearman Rank = {s_mi:.4f} | Pearson r = {p_mi:.4f}")
    print(f"  • Geary's C     : Spearman Rank = {s_gc:.4f} | Pearson r = {p_gc:.4f} (Raw inverse)")
    print(f"  • Geary (1 - C) : Spearman Rank = {s_gc_align:.4f} | Pearson r = {p_gc_align:.4f} (Aligned)")
    print("="*95 + "\n")

    return correlation_lengths, stats_r, stats_mi, stats_gc


def run_phase_2b_snr_degradation_benchmark(length_scale: float, n_realizations: int, n_permutations: int):
    """Phase 2B: Evaluates TRACEBIND R, Moran's I, and Geary's C decay under additive noise."""
    print("="*95)
    print(f" PHASE 2B: ADDITIVE NOISE (SNR) DEGRADATION BENCHMARK (Base GRF l={length_scale})")
    print("="*95)
    
    noise_sigmas = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    stats_r, stats_mi, stats_gc = [], [], []
    
    print(f"\nEvaluating Noise Degradation across σ_N ∈ {noise_sigmas} ({n_realizations} realizations each):\n")
    print(f"{'Noise (σ_N)':<12} | {'TRACEBIND R (Mean ± Std)':<25} | {'Moran I (Mean ± Std)':<22} | {'Geary C (Mean ± Std)':<20}")
    print("-" * 95)
    
    for sigma in noise_sigmas:
        r_ens, mi_ens, gc_ens = [], [], []
        for seed in range(200, 200 + n_realizations):
            rng = np.random.default_rng(seed)
            base_field = generate_exponential_grf(shape=(16, 16), correlation_length=length_scale, seed=seed)
            noise = rng.normal(0, sigma, size=base_field.shape) if sigma > 0 else 0
            noisy_field = base_field + noise
            
            res, collection, graph, _ = run_pipeline(noisy_field, k=4, n_permutations=n_permutations, seed=seed)
            values = collection.point_cloud.values
            
            pysal_w = build_pysal_weights_from_graph(graph, transform='R', symmetrize=True)
            mi = Moran(values, pysal_w, permutations=0)
            gc = Geary(values, pysal_w, permutations=0)
            
            r_ens.append(res.r_observed)
            mi_ens.append(mi.I)
            gc_ens.append(gc.C)
            
        m_r, std_r = np.mean(r_ens), np.std(r_ens)
        m_mi, std_mi = np.mean(mi_ens), np.std(mi_ens)
        m_gc, std_gc = np.mean(gc_ens), np.std(gc_ens)
        
        stats_r.append((m_r, std_r))
        stats_mi.append((m_mi, std_mi))
        stats_gc.append((m_gc, std_gc))
        
        print(f"σ_N = {sigma:<6.2f} | {m_r:8.4f} ± {std_r:<12.4f} | {m_mi:8.4f} ± {std_mi:<10.4f} | {m_gc:8.4f} ± {std_gc:<8.4f}")
        
    print("-" * 95 + "\n")
    return noise_sigmas, stats_r, stats_mi, stats_gc


def run_k_neighborhood_sensitivity_sweep(n_permutations: int):
    """Neighborhood Size (k) Parameter Stability Study featuring average external graph degree."""
    print("="*95)
    print(" NEIGHBORHOOD SIZE (k) PARAMETER STABILITY SWEEP")
    print("="*95)
    
    k_values = [2, 4, 6, 8, 12]
    field = generate_exponential_grf(shape=(16, 16), correlation_length=3.0, seed=42)
    
    print(f"\nTesting GRF (l=3.0) across k ∈ {k_values}:\n")
    print(f"{'Req k':<8} | {'Mean Ext Deg':<12} | {'TRACEBIND R':<10} {'Z(R)':<8} | {'Moran I':<10} {'Z(I)':<8} | {'Geary C':<10} {'Z(C)':<8}")
    print("-" * 95)
    
    for k in k_values:
        res, collection, graph, _ = run_pipeline(field, k=k, n_permutations=n_permutations, seed=42)
        values = collection.point_cloud.values
        
        avg_deg = get_graph_mean_degree(graph)
        pysal_w = build_pysal_weights_from_graph(graph, transform='R', symmetrize=True)
        mi = Moran(values, pysal_w, permutations=n_permutations)
        gc = Geary(values, pysal_w, permutations=n_permutations)
        
        print(
            f"k = {k:<4} | {avg_deg:<12.2f} | "
            f"{res.r_observed:10.4f} {res.z_score:8.2f} | "
            f"{mi.I:10.4f} {mi.z_sim:8.2f} | "
            f"{gc.C:10.4f} {gc.z_sim:8.2f}"
        )
        
    print("-" * 95 + "\n")


def plot_benchmark_results(l_data, noise_data, save_path: str = "benchmark_summary.png"):
    """Generates publication-ready figures for metric tracking and noise degradation."""
    lengths, r_scale, mi_scale, gc_scale = l_data
    sigmas, r_noise, mi_noise, gc_noise = noise_data
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    # --- Plot 1: Correlation Length Sweep ---
    ax1 = axes[0]
    ax1.errorbar(lengths, [s[0] for s in r_scale], yerr=[s[1] for s in r_scale], fmt='-o', capsize=3, label=r'TRACEBIND $R$', color='#1f77b4')
    ax1.errorbar(lengths, [s[0] for s in mi_scale], yerr=[s[1] for s in mi_scale], fmt='-s', capsize=3, label=r"Moran's $I$", color='#ff7f0e')
    ax1.errorbar(lengths, [1.0 - s[0] for s in gc_scale], yerr=[s[1] for s in gc_scale], fmt='-^', capsize=3, label=r"$1 - \text{Geary } C \text{ (Aligned)}$", color='#2ca02c')
    
    ax1.set_xlabel(r"Spatial Correlation Length $(\ell)$", fontsize=11)
    ax1.set_ylabel("Metric Value", fontsize=11)
    ax1.set_title(r"A. Metric Response vs. Spatial Correlation Length", fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(frameon=True)
    
    # --- Plot 2: Noise Degradation Sweep ---
    ax2 = axes[1]
    ax2.errorbar(sigmas, [s[0] for s in r_noise], yerr=[s[1] for s in r_noise], fmt='-o', capsize=3, label=r'TRACEBIND $R$', color='#1f77b4')
    ax2.errorbar(sigmas, [s[0] for s in mi_noise], yerr=[s[1] for s in mi_noise], fmt='-s', capsize=3, label=r"Moran's $I$", color='#ff7f0e')
    ax2.errorbar(sigmas, [1.0 - s[0] for s in gc_noise], yerr=[s[1] for s in gc_noise], fmt='-^', capsize=3, label=r"$1 - \text{Geary } C \text{ (Aligned)}$", color='#2ca02c')
    
    ax2.set_xlabel(r"Additive Noise Std Dev $(\sigma_N)$", fontsize=11)
    ax2.set_ylabel("Metric Value", fontsize=11)
    ax2.set_title(r"B. Metric Decay Under Additive Noise ($\ell=6.0$)", fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(frameon=True)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"[✓] Benchmark figure saved successfully to: {save_path}")


# --- PyTest Integration Block ---
def test_cross_metric_benchmark():
    """PyTest entry point to validate cross-metric benchmark pipeline execution."""
    results = run_phase_2a_pysal_cross_metric_benchmark(n_permutations=10)
    assert len(results) > 0, "Cross-metric benchmark failed to produce execution metrics."


if __name__ == "__main__":
    args = parse_args()
    
    if args.publication:
        n_permutations = PUBLICATION_PERMUTATIONS
        n_realizations = PUBLICATION_REALIZATIONS
        print(f"Running in PUBLICATION mode ({n_realizations} realizations, {n_permutations} permutations)...")
    else:
        n_permutations = DEFAULT_PERMUTATIONS
        n_realizations = DEFAULT_REALIZATIONS
        print(f"Running in FAST development mode ({n_realizations} realizations, {n_permutations} permutations). Use --publication for full runs.")

    run_phase_2a_pysal_cross_metric_benchmark(n_permutations=n_permutations)
    l_data = run_grf_correlation_tracking_analysis(n_realizations=n_realizations, n_permutations=n_permutations)
    noise_data = run_phase_2b_snr_degradation_benchmark(length_scale=6.0, n_realizations=n_realizations, n_permutations=n_permutations)
    run_k_neighborhood_sensitivity_sweep(n_permutations=n_permutations)
    
    plot_benchmark_results(l_data, noise_data)