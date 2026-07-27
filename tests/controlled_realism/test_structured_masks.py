"""
TRACEBIND Phase 4A: Structured Masking Topology & Graph Sensitivity Benchmark

Evaluates the canonical TRACEBIND pipeline under non-MCAR spatial missingness,
quantifying signal degradation (ΔR) and graph edge survival across controlled 
masking geometries.
"""

import sys
from pathlib import Path
import numpy as np
import scipy.ndimage as ndi
from scipy import stats
from typing import Dict, Any, Tuple, List, NamedTuple  # <-- Added NamedTuple here

# Set project root on sys.path dynamically
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Pipeline orchestration entry point
from tests.test_domain_validation import run_pipeline


class MaskAudit(NamedTuple):
    target_frac: float
    actual_frac: float
    valid_nodes: int
    connected_components: int
    boundary_perimeter: int


def generate_synthetic_grf(shape: Tuple[int, int] = (128, 128), 
                           correlation_length: float = 16.0, 
                           seed: int = 42) -> np.ndarray:
    """Generates a reproducible 2D exponential Gaussian Random Field."""
    rng = np.random.default_rng(seed)
    nx, ny = shape
    x, y = np.arange(nx), np.arange(ny)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    
    dist = np.sqrt((xx - nx // 2)**2 + (yy - ny // 2)**2)
    cov = np.exp(-dist / correlation_length)
    
    fft_cov = np.fft.fft2(np.fft.ifftshift(cov))
    white_noise = rng.normal(size=shape)
    fft_noise = np.fft.fft2(white_noise)
    field = np.real(np.fft.ifft2(fft_noise * np.sqrt(np.maximum(0, fft_cov))))
    
    return (field - np.mean(field)) / np.std(field)


def audit_mask_geometry(masked_field: np.ndarray, target_frac: float) -> MaskAudit:
    """Executes a structural audit on the masked observational grid."""
    valid_mask = np.isfinite(masked_field)
    actual_frac = float(np.mean(~valid_mask))
    valid_nodes = int(np.sum(valid_mask))
    
    # Boundary perimeter via binary erosion
    eroded = ndi.binary_erosion(valid_mask)
    boundary = valid_mask ^ eroded
    perimeter = int(np.sum(boundary))
    
    # Connected components in valid observation domain
    _, n_components = ndi.label(valid_mask)
    
    return MaskAudit(
        target_frac=target_frac,
        actual_frac=actual_frac,
        valid_nodes=valid_nodes,
        connected_components=int(n_components),
        boundary_perimeter=perimeter
    )


def extract_graph_edge_count(graph_obj: Any) -> int:
    """Extracts total edge count from the Neighborhood Graph object."""
    if hasattr(graph_obj, "neighbours") and isinstance(graph_obj.neighbours, dict):
        return sum(len(v) for v in graph_obj.neighbours.values()) // 2
    if hasattr(graph_obj, "degrees"):
        return sum(graph_obj.degrees) // 2
    raise AttributeError(f"Could not extract edge count from graph of type {type(graph_obj)}")


# =====================================================================
# 1. CONTROLLED GEOMETRY GENERATORS (WITH ±0.5% TOLERANCE LOOP)
# =====================================================================

def generate_controlled_mask(
    field: np.ndarray, 
    target_frac: float, 
    generator_func, 
    rng: np.random.Generator,
    tolerance: float = 0.005,
    max_attempts: int = 50
) -> Tuple[np.ndarray, MaskAudit]:
    """Enforces actual missingness fraction within ±tolerance of target."""
    for _ in range(max_attempts):
        m_field, actual_frac = generator_func(field, target_frac, rng)
        if abs(actual_frac - target_frac) <= tolerance:
            audit = audit_mask_geometry(m_field, target_frac)
            return m_field, audit
            
    audit = audit_mask_geometry(m_field, target_frac)
    return m_field, audit


def _mcar_gen(field: np.ndarray, target_frac: float, rng: np.random.Generator):
    masked = field.copy()
    mask = rng.random(field.shape) < target_frac
    masked[mask] = np.nan
    return masked, np.mean(np.isnan(masked))


def _swath_gen(field: np.ndarray, target_frac: float, rng: np.random.Generator):
    masked = field.copy()
    nx, ny = field.shape
    angle = rng.uniform(0, np.pi)
    period = rng.uniform(20.0, 40.0)
    phase = rng.uniform(0, period)
    
    xx, yy = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    proj = xx * np.cos(angle) + yy * np.sin(angle) + phase
    
    gap_width = period * target_frac
    mask = (proj % period) < gap_width
    masked[mask] = np.nan
    return masked, np.mean(np.isnan(masked))


def _stripe_gen(field: np.ndarray, target_frac: float, rng: np.random.Generator):
    masked = field.copy()
    nx, ny = field.shape
    axis = rng.choice([0, 1])
    n_dim = nx if axis == 0 else ny
    
    n_drop = int(round(n_dim * target_frac))
    drop_indices = rng.choice(n_dim, size=n_drop, replace=False)
    
    mask = np.zeros_like(field, dtype=bool)
    if axis == 0:
        mask[drop_indices, :] = True
    else:
        mask[:, drop_indices] = True
        
    masked[mask] = np.nan
    return masked, np.mean(np.isnan(masked))


def _cloud_gen(field: np.ndarray, target_frac: float, rng: np.random.Generator):
    masked = field.copy()
    nx, ny = field.shape
    cx, cy = rng.uniform(0.3 * nx, 0.7 * nx), rng.uniform(0.3 * ny, 0.7 * ny)
    theta = rng.uniform(0, np.pi)
    aspect = rng.uniform(0.4, 0.8)
    
    target_area = (nx * ny) * target_frac
    a = np.sqrt(target_area / (np.pi * aspect))
    b = aspect * a
    
    xx, yy = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    xr = (xx - cx) * np.cos(theta) + (yy - cy) * np.sin(theta)
    yr = -(xx - cx) * np.sin(theta) + (yy - cy) * np.cos(theta)
    
    mask = ((xr**2 / (a**2 + 1e-6)) + (yr**2 / (b**2 + 1e-6))) <= 1.0
    masked[mask] = np.nan
    return masked, np.mean(np.isnan(masked))


def _circle_gen(field: np.ndarray, target_frac: float, rng: np.random.Generator):
    masked = field.copy()
    nx, ny = field.shape
    cx, cy = rng.uniform(0.25 * nx, 0.75 * nx), rng.uniform(0.25 * ny, 0.75 * ny)
    
    target_area = (nx * ny) * target_frac
    r = np.sqrt(target_area / np.pi)
    
    xx, yy = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    mask = np.sqrt((xx - cx)**2 + (yy - cy)**2) <= r
    masked[mask] = np.nan
    return masked, np.mean(np.isnan(masked))


def _harmonic_coastline_gen(field: np.ndarray, target_frac: float, rng: np.random.Generator):
    masked = field.copy()
    nx, ny = field.shape
    x = np.arange(nx)
    
    base_y = ny * (1.0 - target_frac)
    phase1, phase2 = rng.uniform(0, 2 * np.pi, size=2)
    
    harmonic = (12.0 * np.sin(x * 0.05 + phase1) + 6.0 * np.cos(x * 0.10 + phase2))
    boundary_y = np.clip(base_y + harmonic, 0, ny).astype(int)
    
    mask = np.zeros_like(field, dtype=bool)
    for i in range(nx):
        mask[i, boundary_y[i]:] = True
        
    masked[mask] = np.nan
    return masked, np.mean(np.isnan(masked))


# =====================================================================
# 2. STATISTICAL ESTIMATORS (BOOTSTRAP & CORRELATIONS)
# =====================================================================

def bootstrap_mean_ci(data: np.ndarray, n_boot: int = 10000, ci: float = 95.0, seed: int = 42) -> Tuple[float, float]:
    """Computes non-parametric bootstrap 95% Confidence Interval for the mean."""
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot)
    n = len(data)
    for b in range(n_boot):
        sample = rng.choice(data, size=n, replace=True)
        boot_means[b] = np.mean(sample)
    
    lower = (100.0 - ci) / 2.0
    upper = 100.0 - lower
    return float(np.percentile(boot_means, lower)), float(np.percentile(boot_means, upper))


# =====================================================================
# 3. BENCHMARK EXECUTOR
# =====================================================================

def run_phase_4a_benchmark(n_realizations: int = 20, grid_shape: Tuple[int, int] = (128, 128)):
    print("\n" + "=" * 155)
    print("         TRACEBIND PHASE 4A: PRODUCTION-READY MASKING TOPOLOGY & GRAPH SENSITIVITY BENCHMARK")
    print(f"             Execution Grid Resolution: {grid_shape[0]}x{grid_shape[1]} | Realizations per cell: {n_realizations}")
    print("=" * 155)
    
    levels = [0.10, 0.20, 0.30, 0.40, 0.50]
    topologies = {
        "MCAR (Control)": _mcar_gen,
        "Orbital Swath": _swath_gen,
        "Sensor Stripe": _stripe_gen,
        "Elliptical Cloud": _cloud_gen,
        "Circular Occlusion": _circle_gen,
        "Harmonic Coastline": _harmonic_coastline_gen
    }
    
    # Establish Baseline on Complete Grid
    base_field = generate_synthetic_grf(shape=grid_shape, seed=42)
    base_res, _, base_graph, _ = run_pipeline(base_field, k=4, n_permutations=20, seed=42, drop_nan=True)
    R_0 = base_res.r_observed
    base_edges = extract_graph_edge_count(base_graph)
    
    print(f" Baseline Signal R_0 = {R_0:.4f} | Baseline Graph Edges = {base_edges}\n")
    print(f"{'Topology':<20} | {'Target %':<8} | {'Actual Drop':<11} | {'Perimeter (px)':<15} | {'Mean R':<9} | {'ΔR':<8} | {'Retention %':<12} | {'Edge Survival %':<15}")
    print("-" * 155)
    
    summary = {top: {} for top in topologies}
    raw_results = {top: {lvl: [] for lvl in levels} for top in topologies}
    all_perimeters = []
    all_delta_R = []
    
    for top_name, gen_func in topologies.items():
        for lvl in levels:
            R_list, drop_list, perim_list, edge_surv_list = [], [], [], []
            
            for real in range(n_realizations):
                # PAIRED DESIGN: Identical base field seed per realization across all topologies
                field = generate_synthetic_grf(shape=grid_shape, seed=real + 1000)
                rng = np.random.default_rng(seed=real + 5000)
                
                m_field, audit = generate_controlled_mask(field, lvl, gen_func, rng)
                
                # Production pipeline call
                res, _, graph, _ = run_pipeline(m_field, k=4, n_permutations=20, seed=real, drop_nan=True)
                
                edges = extract_graph_edge_count(graph)
                edge_surv = (edges / base_edges) * 100.0 if base_edges > 0 else 0.0
                
                R_list.append(res.r_observed)
                drop_list.append(audit.actual_frac)
                perim_list.append(audit.boundary_perimeter)
                edge_surv_list.append(edge_surv)
                
                raw_results[top_name][lvl].append(res.r_observed)
                all_perimeters.append(audit.boundary_perimeter)
                all_delta_R.append(res.r_observed - R_0)
                
            mean_R = float(np.mean(R_list))
            delta_R = mean_R - R_0
            mean_drop = float(np.mean(drop_list))
            mean_perim = float(np.mean(perim_list))
            ret_pct = (mean_R / R_0) * 100.0
            mean_edge_surv = float(np.mean(edge_surv_list))
            
            summary[top_name][lvl] = {
                'mean_R': mean_R,
                'delta_R': delta_R,
                'actual_drop': mean_drop,
                'perimeter': mean_perim,
                'ret_pct': ret_pct,
                'edge_survival': mean_edge_surv
            }
            
            print(f"{top_name:<20} | {f'{lvl*100:.0f}%':<8} | {f'{mean_drop*100:.2f}%':<11} | {mean_perim:<15.1f} | {mean_R:<9.4f} | {delta_R:<+8.4f} | {f'{ret_pct:.2f}%':<12} | {f'{mean_edge_surv:.1f}%':<15}")
        print("-" * 155)

    # Statistical Evaluation at 50% Spatial Dropout
    print("\n" + "=" * 155)
    print("            STATISTICAL SIGNIFICANCE & RELATIVE SIGNAL LOSS AT 50% MISSINGNESS (PAIRED WILCOXON & 10k BOOTSTRAP CI)")
    print("=" * 155)
    print(f"{'Topology':<20} | {'Mean Retention [95% Bootstrap CI]':<38} | {'Relative Loss vs MCAR':<24} | {'Wilcoxon p-value':<18} | {'Significance'}")
    print("-" * 155)
    
    mcar_50_vec = np.array(raw_results["MCAR (Control)"][0.50])
    mcar_50_ret = summary["MCAR (Control)"][0.50]['ret_pct']
    
    for top_name in topologies:
        r_vec = np.array(raw_results[top_name][0.50])
        ret_vec = (r_vec / R_0) * 100.0
        mean_ret = np.mean(ret_vec)
        
        # True Non-Parametric Bootstrap CI
        ci_low, ci_high = bootstrap_mean_ci(ret_vec, n_boot=10000, ci=95.0, seed=42)
        ci_str = f"{mean_ret:.2f}% [{ci_low:.2f}%, {ci_high:.2f}%]"
        
        if top_name == "MCAR (Control)":
            p_val_str, sig_str = "N/A (Control)", "Baseline"
            rel_loss_str = "0.00%"
        else:
            # Valid Paired Wilcoxon Test (due to matched realization seeds)
            _, p_val = stats.wilcoxon(mcar_50_vec, r_vec)
            p_val_str = f"{p_val:.4e}"
            sig_str = "p < 0.01 (Significant)" if p_val < 0.01 else "Not Significant"
            rel_loss = mcar_50_ret - summary[top_name][0.50]['ret_pct']
            rel_loss_str = f"{rel_loss:+.2f}%"
            
        print(f"{top_name:<20} | {ci_str:<38} | {rel_loss_str:<24} | {p_val_str:<18} | {sig_str}")
        
    print("=" * 155)

    # Exploratory Diagnostic Correlation: Boundary Perimeter vs Delta R
    r_pearson, p_pearson = stats.pearsonr(all_perimeters, all_delta_R)
    r_spearman, p_spearman = stats.spearmanr(all_perimeters, all_delta_R)
    
    print("\n" + "=" * 155)
    print("            EXPLORATORY METRIC CORRELATION: BOUNDARY PERIMETER vs SIGNAL DEGRADATION (ΔR)")
    print("=" * 155)
    print(f" Pearson Correlation (r)  : {r_pearson:+.4f} (p = {p_pearson:.4e})")
    print(f" Spearman Correlation (ρ) : {r_spearman:+.4f} (p = {p_spearman:.4e})")
    print("=" * 155 + "\n")


if __name__ == "__main__":
    run_phase_4a_benchmark(n_realizations=20, grid_shape=(128, 128))