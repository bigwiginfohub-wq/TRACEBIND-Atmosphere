"""
TRACEBIND Falsification Framework: Isolating Masking & Normalization Mechanisms

Executes paired diagnostic comparisons across 5 experimental stages:
  1. Original Field (GRF baseline, l=16.0)
  2. Masked Field (50% Occlusion with NaNs)
  3. Ground-Truth Restored Field (Restores original values to NaNs)
  4. Nearest-Neighbor Interpolation
  5. Linear Interpolation

Tracks both signal coherence (R) and graph topology invariants (|V|, |E|, mean degree,
connected components, largest component fraction) to isolate estimator artifacts.
"""

import sys
from pathlib import Path
import numpy as np
import scipy.ndimage as ndi
from scipy.interpolate import griddata
from typing import Dict, Any, Tuple, NamedTuple, List

# Set project root on sys.path dynamically
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Pipeline orchestration entry point
from tests.test_domain_validation import run_pipeline


class GraphMetrics(NamedTuple):
    nodes: int
    edges: int
    mean_degree: float
    components: int
    largest_component_frac: float


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


def extract_graph_topology(graph_obj: Any) -> GraphMetrics:
    """Extracts structural graph invariants directly from the Neighborhood Graph object."""
    # Active nodes
    if hasattr(graph_obj, "n_nodes"):
        nodes = int(graph_obj.n_nodes)
    elif hasattr(graph_obj, "neighbours"):
        nodes = len(graph_obj.neighbours)
    else:
        nodes = 0

    # Total Edges
    if hasattr(graph_obj, "neighbours") and isinstance(graph_obj.neighbours, dict):
        edges = sum(len(v) for v in graph_obj.neighbours.values()) // 2
    elif hasattr(graph_obj, "degrees"):
        edges = sum(graph_obj.degrees) // 2
    else:
        edges = 0

    # Mean Degree
    if hasattr(graph_obj, "mean_degree"):
        mean_deg = float(graph_obj.mean_degree)
    elif nodes > 0:
        mean_deg = (2.0 * edges) / nodes
    else:
        mean_deg = 0.0

    # Connected Components & Largest Component via Neighbor Adjacency
    if hasattr(graph_obj, "neighbours") and isinstance(graph_obj.neighbours, dict):
        adj = graph_obj.neighbours
        visited = set()
        comp_sizes = []
        for node in adj:
            if node not in visited:
                comp_size = 0
                stack = [node]
                visited.add(node)
                while stack:
                    curr = stack.pop()
                    comp_size += 1
                    for nbr in adj.get(curr, []):
                        if nbr not in visited:
                            visited.add(nbr)
                            stack.append(nbr)
                comp_sizes.append(comp_size)
        components = len(comp_sizes)
        largest_frac = (max(comp_sizes) / nodes) if nodes > 0 else 0.0
    else:
        components = 1
        largest_frac = 1.0

    return GraphMetrics(
        nodes=nodes,
        edges=edges,
        mean_degree=mean_deg,
        components=components,
        largest_component_frac=largest_frac
    )


def apply_circle_mask(field: np.ndarray, target_frac: float = 0.50, seed: int = 42) -> np.ndarray:
    """Applies a centered circular occlusion mask matching requested missingness."""
    masked = field.copy()
    nx, ny = field.shape
    cx, cy = nx / 2.0, ny / 2.0
    
    target_area = (nx * ny) * target_frac
    r = np.sqrt(target_area / np.pi)
    
    xx, yy = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    mask = np.sqrt((xx - cx)**2 + (yy - cy)**2) <= r
    masked[mask] = np.nan
    return masked


def reconstruct_grid(masked_field: np.ndarray, method: str = "nearest") -> np.ndarray:
    """Interpolates NaN values in a 2D observational grid."""
    nx, ny = masked_field.shape
    grid_x, grid_y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    
    valid_mask = np.isfinite(masked_field)
    points = np.column_stack((grid_x[valid_mask], grid_y[valid_mask]))
    values = masked_field[valid_mask]
    
    reconstructed = griddata(points, values, (grid_x, grid_y), method=method)
    
    # Handle border NaNs if convex hull excludes grid edges
    if np.isnan(reconstructed).any():
        fallback_nn = griddata(points, values, (grid_x, grid_y), method="nearest")
        reconstructed[np.isnan(reconstructed)] = fallback_nn[np.isnan(reconstructed)]
        
    return reconstructed


def run_falsification_benchmark(n_realizations: int = 15, grid_shape: Tuple[int, int] = (128, 128)):
    print("\n" + "=" * 125)
    print("      TRACEBIND FALSIFICATION BENCHMARK: MECHANISTIC ANALYSIS OF MASKING & RECONSTRUCTION")
    print(f"          Grid Resolution: {grid_shape[0]}x{grid_shape[1]} | Paired Realizations: {n_realizations} | Base: GRF (l=16.0)")
    print("=" * 125)
    
    stages = ["Original", "Masked (50%)", "Ground-Truth Restored", "Interp (Nearest)", "Interp (Linear)"]
    metrics_store = {stage: {"R": [], "nodes": [], "edges": [], "mean_degree": [], "components": [], "largest_frac": []} for stage in stages}

    for i in range(n_realizations):
        seed = 1000 + i
        
        # 1. Base Realization (GRF)
        orig_field = generate_synthetic_grf(shape=grid_shape, correlation_length=16.0, seed=seed)
        res_orig, _, g_orig, _ = run_pipeline(orig_field, k=4, n_permutations=20, seed=seed, drop_nan=True)
        top_orig = extract_graph_topology(g_orig)
        
        # 2. Masked Realization (Circular Occlusion 50%)
        masked_field = apply_circle_mask(orig_field, target_frac=0.50, seed=seed)
        res_mask, _, g_mask, _ = run_pipeline(masked_field, k=4, n_permutations=20, seed=seed, drop_nan=True)
        top_mask = extract_graph_topology(g_mask)
        
        # 3. Ground-Truth Restored Control (Exact Pixel Restoration)
        restored_field = masked_field.copy()
        nan_mask = np.isnan(restored_field)
        restored_field[nan_mask] = orig_field[nan_mask]
        res_rest, _, g_rest, _ = run_pipeline(restored_field, k=4, n_permutations=20, seed=seed, drop_nan=True)
        top_rest = extract_graph_topology(g_rest)
        
        # 4. Nearest-Neighbor Interpolation
        nn_field = reconstruct_grid(masked_field, method="nearest")
        res_nn, _, g_nn, _ = run_pipeline(nn_field, k=4, n_permutations=20, seed=seed, drop_nan=True)
        top_nn = extract_graph_topology(g_nn)
        
        # 5. Linear Interpolation
        lin_field = reconstruct_grid(masked_field, method="linear")
        res_lin, _, g_lin, _ = run_pipeline(lin_field, k=4, n_permutations=20, seed=seed, drop_nan=True)
        top_lin = extract_graph_topology(g_lin)
        
        # Store results
        stage_data = [
            ("Original", res_orig.r_observed, top_orig),
            ("Masked (50%)", res_mask.r_observed, top_mask),
            ("Ground-Truth Restored", res_rest.r_observed, top_rest),
            ("Interp (Nearest)", res_nn.r_observed, top_nn),
            ("Interp (Linear)", res_lin.r_observed, top_lin)
        ]
        
        for name, r_val, top in stage_data:
            metrics_store[name]["R"].append(r_val)
            metrics_store[name]["nodes"].append(top.nodes)
            metrics_store[name]["edges"].append(top.edges)
            metrics_store[name]["mean_degree"].append(top.mean_degree)
            metrics_store[name]["components"].append(top.components)
            metrics_store[name]["largest_frac"].append(top.largest_component_frac)

    # Print Summary Table
    print(f"{'Experimental Stage':<25} | {'Mean R':<8} | {'ΔR vs Orig':<11} | {'Active Nodes':<12} | {'Edges':<8} | {'Mean Degree':<11} | {'Components':<10} | {'Largest Comp %':<14}")
    print("-" * 125)
    
    R_0_mean = np.mean(metrics_store["Original"]["R"])
    
    for stage in stages:
        m = metrics_store[stage]
        r_mean = np.mean(m["R"])
        delta_r = r_mean - R_0_mean
        nodes_mean = np.mean(m["nodes"])
        edges_mean = np.mean(m["edges"])
        deg_mean = np.mean(m["mean_degree"])
        comp_mean = np.mean(m["components"])
        l_frac_mean = np.mean(m["largest_frac"]) * 100.0
        
        print(f"{stage:<25} | {r_mean:<8.4f} | {delta_r:<+11.4f} | {nodes_mean:<12.1f} | {edges_mean:<8.1f} | {deg_mean:<11.2f} | {comp_mean:<10.1f} | {f'{l_frac_mean:.1f}%':<14}")
        
    print("=" * 125)
    
    # Mechanistic Diagnostics
    r_mask_mean = np.mean(metrics_store["Masked (50%)"]["R"])
    r_rest_mean = np.mean(metrics_store["Ground-Truth Restored"]["R"])
    r_lin_mean = np.mean(metrics_store["Interp (Linear)"]["R"])
    
    print("\n--- MECHANISTIC DIAGNOSTIC SUMMARY ---")
    print(f"1. Ground-Truth Restoration Identity Check:")
    print(f"   Original R_0 = {R_0_mean:.6f} | Restored R = {r_rest_mean:.6f} | Abs Diff = {abs(R_0_mean - r_rest_mean):.2e}")
    if abs(R_0_mean - r_rest_mean) < 1e-5:
        print("   -> CONFIRMED: Pipeline is deterministic and restores exact metric when NaNs are filled with ground truth.")
    else:
        print("   -> WARNING: Pipeline state drift or non-determinism detected upon pixel re-insertion.")
        
    print(f"\n2. Masking Shift Analysis:")
    print(f"   R Shift under Masking (NaNs): {r_mask_mean - R_0_mean:+.4f} ({((r_mask_mean - R_0_mean)/R_0_mean)*100:.2f}%)")
    print(f"   R Shift under Linear Interp : {r_lin_mean - R_0_mean:+.4f} ({((r_lin_mean - R_0_mean)/R_0_mean)*100:.2f}%)")
    print("=" * 125 + "\n")


if __name__ == "__main__":
    run_falsification_benchmark(n_realizations=15, grid_shape=(128, 128))