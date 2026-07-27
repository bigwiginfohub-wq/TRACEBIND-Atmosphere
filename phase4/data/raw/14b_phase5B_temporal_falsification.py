"""
14b_phase5B_temporal_falsification.py
------------------------------------
Phase 5B: Dynamic Precursor Trajectory Falsification
Computes full time-series GE(t) and dGE/dt across 72h windows.
Evaluates surrogate null distributions using temporal phase randomization.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

RAW_DATA_DIR = Path(r"C:\TRACEBIND-Atmosphere\phase4\data\raw")
OUTPUT_DIR = Path(r"C:\TRACEBIND-Atmosphere\phase5\results")

STORMS = ["Amphan", "Fani", "Mocha", "Yaas", "Sidr", "Nargis"]
TARGET_FIELD = "msl"
N_TEMPORAL_SURROGATES = 500

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

def compute_spatial_ge(frame_2d: np.ndarray) -> float:
    gy, gx = np.gradient(frame_2d)
    return float(np.sum(gx**2 + gy**2))

def generate_ft_surrogate_series(ts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Fourier Phase Randomization preserving time series autocorrelation spectrum."""
    n = len(ts)
    ft = np.fft.rfft(ts)
    phases = rng.uniform(0, 2 * np.pi, len(ft))
    phases[0] = 0 # Preserve DC component
    if n % 2 == 0: phases[-1] = 0
    surr_ft = np.abs(ft) * np.exp(1j * phases)
    return np.fft.irfft(surr_ft, n=n)

def run_phase5b_temporal_falsification():
    print("=========================================================================================")
    print(f"      TRACEBIND PHASE 5B: DYNAMIC TEMPORAL TRAJECTORY FALSIFICATION ({TARGET_FIELD.upper()})")
    print("=========================================================================================\n")

    summary_records = []
    rng = np.random.default_rng(42)

    for storm in STORMS:
        nc_path = resolve_nc_path(storm)
        if not nc_path or not nc_path.exists(): continue

        ds = xr.open_dataset(nc_path)
        if TARGET_FIELD not in ds:
            ds.close()
            continue

        da = ds[TARGET_FIELD]
        time_dim = next((d for d in ["valid_time", "time", "date"] if d in da.dims), None)
        if not time_dim:
            ds.close()
            continue

        time_steps = da.sizes[time_dim]
        obs_ge_ts = np.empty(time_steps, dtype=np.float64)

        # 1. Compute GE(t) series across all time steps
        for t in range(time_steps):
            frame = np.squeeze(da.isel({time_dim: t}).values).astype(np.float64)
            obs_ge_ts[t] = compute_spatial_ge(frame)

        # 2. Compute Derivative Trajectory dGE/dt
        obs_dge_dt = np.gradient(obs_ge_ts)
        obs_max_rate = float(np.max(np.abs(obs_dge_dt)))
        obs_peak_t = int(np.argmax(obs_ge_ts))

        # 3. Generate Fourier Phase-Randomized Time-Series Surrogates
        surr_max_rates = np.empty(N_TEMPORAL_SURROGATES, dtype=np.float64)
        for i in range(N_TEMPORAL_SURROGATES):
            surr_ts = generate_ft_surrogate_series(obs_ge_ts, rng)
            surr_dge_dt = np.gradient(surr_ts)
            surr_max_rates[i] = np.max(np.abs(surr_dge_dt))

        null_rate_mean = float(np.mean(surr_max_rates))
        null_rate_std  = float(np.std(surr_max_rates))
        cohen_d_rate   = (obs_max_rate - null_rate_mean) / null_rate_std if null_rate_std > 0 else np.nan

        # P-value for trajectory rate significance
        p_val_trajectory = float(np.sum(surr_max_rates >= obs_max_rate) + 1) / (N_TEMPORAL_SURROGATES + 1)

        summary_records.append({
            "Storm": storm,
            "Time_Steps": time_steps,
            "Peak_GE_Hour": obs_peak_t,
            "Obs_Max_|dGE/dt|": f"{obs_max_rate:.4e}",
            "Surr_Null_|dGE/dt|_Mean": f"{null_rate_mean:.4e}",
            "Rate_Cohens_d": round(cohen_d_rate, 2),
            "Trajectory_p_value": round(p_val_trajectory, 4),
            "Precursor_Falsification_Passed": p_val_trajectory < 0.05
        })

        print(f"[✓] {storm:8s} | Steps: {time_steps:2d} | Peak t: {obs_peak_t:2d}h | Max |dGE/dt|: {obs_max_rate:.3e} | Cohen's d: {cohen_d_rate:5.2f} SD | p-val: {p_val_trajectory:.4f}")
        ds.close()

    results_df = pd.DataFrame(summary_records)
    output_path = OUTPUT_DIR / "phase5b_temporal_falsification_summary.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\n[+] Saved Phase 5B Temporal Falsification results to: {output_path}\n")
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    run_phase5b_temporal_falsification()