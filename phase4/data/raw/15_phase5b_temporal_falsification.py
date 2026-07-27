"""
15_phase5b_temporal_falsification.py
------------------------------------
TRACEBIND Phase 5B: Full Temporal Time-Series Falsification Suite
Evaluates full temporal trajectory GE(t) and dGE/dt against phase-randomized 
surrogate time series to test precursor timing relationships.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

RAW_DATA_DIR = Path(r"C:\TRACEBIND-Atmosphere\phase4\data\raw")
OUTPUT_DIR = Path(r"C:\TRACEBIND-Atmosphere\phase5\results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STORMS = ["Amphan", "Fani", "Mocha", "Yaas", "Sidr", "Nargis"]
TARGET_FIELD = "msl"
N_SURROGATES = 500

def resolve_nc_path(storm_name: str) -> Path | None:
    name_low = storm_name.lower()
    candidates = [
        RAW_DATA_DIR / f"era5_{name_low}_72h.nc",
        RAW_DATA_DIR / f"era5_{name_low}.nc",
        RAW_DATA_DIR / f"{name_low}_era5.nc",
        RAW_DATA_DIR / f"{storm_name}_era5.nc",
    ]
    for p in candidates:
        if p.exists(): return p
    matches = list(RAW_DATA_DIR.glob(f"*{name_low}*.nc"))
    return matches[0] if matches else None

def compute_spatial_ge(array_2d: np.ndarray) -> float:
    gy, gx = np.gradient(array_2d)
    return float(np.sum(gx**2 + gy**2))

def generate_fourier_surrogate_ts(ts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Generates Fourier phase-randomized surrogate time series (preserves power spectrum)."""
    n = len(ts)
    fft_vals = np.fft.rfft(ts)
    phases = rng.uniform(0, 2 * np.pi, size=len(fft_vals))
    # Keep DC component and Nyquist phase zero if even
    phases[0] = 0
    if n % 2 == 0:
        phases[-1] = 0
    fft_surrogate = np.abs(fft_vals) * np.exp(1j * phases)
    return np.fft.irfft(fft_surrogate, n=n)

def run_phase5b_temporal_falsification():
    print("=========================================================================================")
    print(f"       TRACEBIND PHASE 5B: TEMPORAL EVOLUTION FALSIFICATION ({TARGET_FIELD.upper()})    ")
    print("=========================================================================================\n")

    summary_records = []
    rng = np.random.default_rng(seed=42)

    for storm in STORMS:
        nc_path = resolve_nc_path(storm)
        if not nc_path or not nc_path.exists(): continue

        ds = xr.open_dataset(nc_path)
        if TARGET_FIELD not in ds:
            ds.close()
            continue

        da = ds[TARGET_FIELD]
        time_dim = next((d for d in ["valid_time", "time", "date"] if d in da.dims), None)
        
        # Extract full GE(t) trajectory
        n_times = da.sizes[time_dim]
        ge_trajectory = np.zeros(n_times, dtype=np.float64)

        for t_idx in range(n_times):
            frame = np.squeeze(da.isel({time_dim: t_idx}).values).astype(np.float64)
            ge_trajectory[t_idx] = compute_spatial_ge(frame)

        # Compute observed trajectory metrics: Peak GE, Onset Derivative (dGE/dt max)
        obs_peak_ge = float(np.max(ge_trajectory))
        dge_dt = np.gradient(ge_trajectory)
        obs_max_dge = float(np.max(dge_dt))

        # Generate Phase-Randomized Time-Series Surrogates
        surr_peaks = []
        surr_max_dge = []

        for _ in range(N_SURROGATES):
            surr_ge_ts = generate_fourier_surrogate_ts(ge_trajectory, rng)
            surr_peaks.append(np.max(surr_ge_ts))
            surr_max_dge.append(np.max(np.gradient(surr_ge_ts)))

        surr_peaks = np.array(surr_peaks)
        surr_max_dge = np.array(surr_max_dge)

        # Statistical Metrics
        peak_cohen_d = (obs_peak_ge - np.mean(surr_peaks)) / np.std(surr_peaks)
        dge_cohen_d  = (obs_max_dge - np.mean(surr_max_dge)) / np.std(surr_max_dge)
        
        p_val_dge = float(np.sum(surr_max_dge >= obs_max_dge) + 1) / (N_SURROGATES + 1)

        summary_records.append({
            "Storm": storm,
            "Timesteps": n_times,
            "Obs_Peak_GE": f"{obs_peak_ge:.4e}",
            "Obs_Max_dGE_dt": f"{obs_max_dge:.4e}",
            "Surr_Max_dGE_Mean": f"{np.mean(surr_max_dge):.4e}",
            "dGE_dt_Cohens_d": round(dge_cohen_d, 2),
            "p_val_temporal_onset": round(p_val_dge, 4),
            "Falsification_Passed": p_val_dge < 0.05
        })

        print(
            f"[✓] {storm:8s} | T={n_times} | Obs Max dGE/dt: {obs_max_dge:.3e} | "
            f"Surr Mean: {np.mean(surr_max_dge):.3e} | dGE d: {dge_cohen_d:5.2f} SD | p-val: {p_val_dge:.4f}"
        )
        ds.close()

    df_out = pd.DataFrame(summary_records)
    out_file = OUTPUT_DIR / "phase5b_temporal_falsification_summary.csv"
    df_out.to_csv(out_file, index=False)

    print("\n=========================================================================================")
    print(f"               PHASE 5B TEMPORAL FALSIFICATION SAVED TO: {out_file.name}")
    print("=========================================================================================\n")
    print(df_out.to_string(index=False))

if __name__ == "__main__":
    run_phase5b_temporal_falsification()