import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import (
    compute_tracebind_v1,
    compute_tracebind_v2,
    compute_morans_i,
    compute_gearys_c,
    compute_gradient_energy,
    compute_texture_entropy
)

def block_shuffle(grid: np.ndarray, block_size: int = 8) -> np.ndarray:
    """Scale-preserving block shuffle control."""
    h, w = grid.shape
    bh, bw = h // block_size, w // block_size
    blocks = [grid[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size].copy() 
              for i in range(bh) for j in range(bw)]
    np.random.shuffle(blocks)
    shuffled_grid = np.zeros_like(grid)
    idx = 0
    for i in range(bh):
        for j in range(bw):
            shuffled_grid[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size] = blocks[idx]
            idx += 1
    return shuffled_grid

def real_phase_scramble(grid: np.ndarray) -> np.ndarray:
    """Fourier phase scrambling preserving Hermitian symmetry."""
    fft_f = np.fft.fft2(grid)
    amplitude = np.abs(fft_f)
    random_phases = np.random.uniform(-np.pi, np.pi, grid.shape)
    sym_phases = random_phases.copy()
    h, w = grid.shape
    for i in range(h):
        for j in range(w):
            if i == 0 and j == 0:
                sym_phases[i, j] = 0.0
            else:
                sym_phases[i, j] = -sym_phases[-i % h, -j % w]
    scrambled_fft = amplitude * np.exp(1j * sym_phases)
    return np.real(np.fft.ifft2(scrambled_fft))

def generate_topologies(shape=(64, 64)):
    x = np.linspace(-3, 3, shape[1])
    y = np.linspace(-3, 3, shape[0])
    X, Y = np.meshgrid(x, y)
    r = np.sqrt(X**2 + Y**2)
    
    return {
        "White Noise": np.random.normal(0, 1, shape),
        "Single Blob": np.exp(-r**2),
        "Multi-Blob": np.exp(-((X-1.5)**2 + (Y-1.5)**2)) + np.exp(-((X+1.5)**2 + (Y+1.5)**2)),
        "Cyclonic Spiral": np.exp(-r**2) * np.cos(3 * np.arctan2(Y, X)),
        "Filamentary Front": np.sin(2 * np.pi * X / 2.0) * np.exp(-Y**2 / 2.0)
    }

def run_diagnostics():
    np.random.seed(42)
    
    # 1. CONTROLS COMPARISON TABLE
    print("=========================================================================")
    print("   1. DUAL METRIC BENCHMARK: TRACEBIND v1 vs v2 UNDER CONTROLS          ")
    print("=========================================================================\n")
    
    storm_field = generate_topologies()["Cyclonic Spiral"]
    
    controls = {
        "Original": storm_field,
        "Pixel Shuffle": storm_field.copy(),
        "8x8 Block Shuffle": block_shuffle(storm_field, block_size=8),
        "Phase Scramble": real_phase_scramble(storm_field)
    }
    np.random.shuffle(controls["Pixel Shuffle"].ravel())
    
    summary_data = []
    for name, f in controls.items():
        summary_data.append({
            "Condition": name,
            "TRACEBIND v1": compute_tracebind_v1(f, mode="coherence"),
            "TRACEBIND v2": compute_tracebind_v2(f, mode="coherence"),
            "Moran's I": compute_morans_i(f),
            "Geary's C": compute_gearys_c(f),
            "Grad Energy": compute_gradient_energy(f),
            "Entropy": compute_texture_entropy(f)
        })
    
    df_controls = pd.DataFrame(summary_data)
    print(df_controls.to_string(index=False))
    print("\n=========================================================================\n")
    
    # 2. TOPOLOGICAL CALIBRATION CURVE
    print("=========================================================================")
    print("   2. DUAL METRIC BENCHMARK: TOPOLOGICAL CALIBRATION CURVE               ")
    print("=========================================================================\n")
    
    calibration_data = []
    for topo_name, f in generate_topologies().items():
        calibration_data.append({
            "Topology Field": topo_name,
            "TRACEBIND v1": compute_tracebind_v1(f, mode="coherence"),
            "TRACEBIND v2": compute_tracebind_v2(f, mode="coherence"),
            "Moran's I": compute_morans_i(f),
            "Geary's C": compute_gearys_c(f),
            "Grad Energy": compute_gradient_energy(f)
        })
        
    df_calib = pd.DataFrame(calibration_data)
    print(df_calib.to_string(index=False))
    print("\n=========================================================================")

if __name__ == "__main__":
    run_diagnostics()