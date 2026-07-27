# scripts/benchmark_scaling.py
import sys
import time
import tracemalloc
from pathlib import Path

# Force project root onto sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from tests.test_domain_validation import run_pipeline


def run_benchmark():
    grid_sizes = [16, 32, 64]  # Grid dimensions (N = size^2)
    n_permutations = 50
    k_neighbors = 4
    
    print("\n=========================================================================================")
    print("                      TRACEBIND PERFORMANCE & SCALING BENCHMARK                          ")
    print("=========================================================================================")
    print(f"{'Grid Size':<10} | {'Nodes (N)':<10} | {'Edges (E)':<10} | {'Perms':<8} | {'Runtime (s)':<12} | {'Peak Mem (MB)':<12} | {'R_obs':<8}")
    print("-" * 89)
    
    for size in grid_sizes:
        # Generate smooth test grid
        x = np.linspace(0, 10, size)
        y = np.linspace(0, 10, size)
        xx, yy = np.meshgrid(x, y)
        data = np.sin(xx) + np.cos(yy)
        
        n_nodes = size * size
        n_edges = n_nodes * k_neighbors  # Directed k-NN graph
        
        tracemalloc.start()
        t0 = time.perf_counter()
        
        result, _, graph, _ = run_pipeline(
            data, k=k_neighbors, n_permutations=n_permutations, seed=42
        )
        
        elapsed = time.perf_counter() - t0
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        peak_mb = peak_bytes / (1024 * 1024)
        
        print(f"{f'{size}x{size}':<10} | {n_nodes:<10} | {n_edges:<10} | {n_permutations:<8} | {elapsed:<12.4f} | {peak_mb:<12.2f} | {result.r_observed:<8.4f}")

    print("=========================================================================================\n")


if __name__ == "__main__":
    run_benchmark()