import sys
from pathlib import Path

# Force project root onto sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from tests.test_domain_validation import run_pipeline

def generate_exponential_grf(shape=(64, 64), correlation_length=4.0, aspect_ratio=1.0, theta_deg=0.0, seed=42):
    """
    Generates an anisotropic 2D Gaussian Random Field using Spectral (FFT) synthesis.
    O(N log N) time and O(N) memory complexity.
    """
    rng = np.random.default_rng(seed)
    h, w = shape
    
    # Frequency grid
    ky = np.fft.fftfreq(h)[:, None]
    kx = np.fft.fftfreq(w)[None, :]
    
    # Scale frequencies to physical space distances (2*pi*k)
    kx_phys = 2 * np.pi * kx
    ky_phys = 2 * np.pi * ky
    
    # Rotate frequencies
    theta_rad = np.radians(theta_deg)
    cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)
    kx_rot =  kx_phys * cos_t + ky_phys * sin_t
    ky_rot = -kx_phys * sin_t + ky_phys * cos_t
    
    # Effective directional scales
    l_x = correlation_length
    l_y = correlation_length / aspect_ratio
    
    # Anisotropic spectral power density for exponential covariance
    # P(k) is proportional to (1 + (l_x*k_x)^2 + (l_y*k_y)^2)^(-1.5) in 2D
    k_scaled = np.sqrt((l_x * kx_rot)**2 + (l_y * ky_rot)**2)
    psd = (1.0 + k_scaled**2)**(-1.5)
    
    # Complex white noise in frequency domain
    white_noise = rng.normal(0, 1, size=shape) + 1j * rng.normal(0, 1, size=shape)
    
    # Filter white noise by spectral density amplitude
    field_fft = white_noise * np.sqrt(psd)
    
    # Transform back to spatial domain and extract real component
    field = np.fft.ifft2(field_fft).real
    
    # Normalize field to zero mean and unit variance
    field = (field - np.mean(field)) / np.std(field)
    return field

def test_ensemble_correlation_length_sweep():
    """Verifies that ENSEMBLE MEAN spatial correlation R increases monotonically with GRF scale length."""
    print("\n[Characterization] Testing Ensemble-Averaged Metric Sensitivity Across GRF Scales...")
    
    correlation_lengths = [0.5, 1.5, 3.0, 6.0, 12.0]
    num_realizations = 15  # Ensemble size per scale
    mean_results = []
    
    for l in correlation_lengths:
        r_ensemble = []
        for seed in range(100, 100 + num_realizations):
            field = generate_exponential_grf(shape=(16, 16), correlation_length=l, seed=seed)
            res, _, _, _ = run_pipeline(field, k=4, n_permutations=30, seed=seed)
            r_ensemble.append(res.r_observed)
        
        mean_r = np.mean(r_ensemble)
        std_r = np.std(r_ensemble)
        mean_results.append(mean_r)
        print(f"  -> Length Scale l={l:4.1f} | Mean R: {mean_r:7.4f} (±{std_r:.4f}) over {num_realizations} realizations")
    
    # Assert monotonic increase of ENSEMBLE MEAN R with spatial correlation length
    is_monotonic = all(x <= y for x, y in zip(mean_results[:-1], mean_results[1:]))
    assert is_monotonic, f"Expected monotonic increase in ensemble mean R, got: {mean_results}"
    print("  ✓ Ensemble Monotonicity Confirmed: Spatial scale predictably increases observed structural coherence.\n")

if __name__ == "__main__":
    test_ensemble_correlation_length_sweep()