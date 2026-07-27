import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def find_mslp_column(df: pd.DataFrame) -> str:
    """Finds the central pressure column regardless of exact naming convention."""
    candidates = ['min_mslp', 'mslp', 'mslp_hpa', 'center_mslp', 'p_min', 'min_pressure', 'pressure']
    for col in candidates:
        if col in df.columns:
            return col
    # Case-insensitive search fallback
    for col in df.columns:
        if 'mslp' in col.lower() or 'press' in col.lower():
            return col
    return None

def calculate_lagged_correlations(series_a, series_b, max_lag_steps=8):
    """
    Computes cross-correlation between series_a and series_b for lags in [-max_lag_steps, +max_lag_steps].
    For 3-hour ERA5 steps, max_lag_steps=8 corresponds to +/-24 hours.
    A positive lag means series_b leads series_a.
    """
    lags = np.arange(-max_lag_steps, max_lag_steps + 1)
    corrs = []
    
    # Standardize inputs
    a = (series_a - series_a.mean()) / (series_a.std() + 1e-12)
    b = (series_b - series_b.mean()) / (series_b.std() + 1e-12)

    for lag in lags:
        if lag < 0:
            c = np.corrcoef(a[-lag:], b[:lag])[0, 1]
        elif lag > 0:
            c = np.corrcoef(a[:-lag], b[lag:])[0, 1]
        else:
            c = np.corrcoef(a, b)[0, 1]
        corrs.append(c)
        
    return lags * 3, np.array(corrs)  # Return lag in hours alongside correlation values

def analyze_and_plot_metrics(csv_path: Path, output_dir: Path):
    df = pd.read_csv(csv_path)
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)

    mslp_col = find_mslp_column(df)
    if mslp_col:
        print(f"[i] Detected MSLP column: '{mslp_col}'")
    else:
        print(f"[!] Warning: No MSLP column found in CSV. Available columns: {list(df.columns)}")

    # Core metrics to evaluate against storm pressure evolution
    target_metrics = [
        'tb_v1', 'tb_v2', 'morans_i', 'gearys_c', 
        'gradient_energy', 'spectral_slope', 'spatial_entropy'
    ]
    
    # Filter to metrics present in the generated metrics.csv file
    available_metrics = [m for m in target_metrics if m in df.columns]

    # --- 1. Compute & Print Diagnostic Statistics ---
    print("\n" + "="*80)
    print("                      METRIC STATISTICAL SUMMARY & CORRELATIONS              ")
    print("="*80)
    
    stats_list = []
    lag_summary = {}

    for m in available_metrics:
        # Summary Stats
        min_val = df[m].min()
        max_val = df[m].max()
        mean_val = df[m].mean()
        std_val = df[m].std()
        
        # Zero-lag Correlation with MSLP
        if mslp_col:
            corr_mslp = df[m].corr(df[mslp_col])
        else:
            corr_mslp = np.nan
        
        stats_list.append({
            'Metric': m,
            'Min': min_val,
            'Max': max_val,
            'Mean': mean_val,
            'Std': std_val,
            'Corr (MSLP)': corr_mslp
        })
        
        # Lagged Correlation (+/- 24 hours in 3h increments)
        if mslp_col:
            lag_hrs, corrs = calculate_lagged_correlations(df[mslp_col], df[m], max_lag_steps=8)
            peak_idx = np.argmax(np.abs(corrs))
            lag_summary[m] = {
                'peak_lag_hr': lag_hrs[peak_idx],
                'peak_corr': corrs[peak_idx],
                'lag_series': dict(zip(lag_hrs, corrs))
            }

    stats_df = pd.DataFrame(stats_list)
    print(stats_df.to_string(index=False, float_format=lambda x: f"{x:11.6f}"))
    
    if mslp_col:
        print("\n" + "-"*80)
        print("               LAGGED CROSS-CORRELATION SUMMARY (vs. Minimum MSLP)           ")
        print("               (Negative Lag = Metric Leads MSLP / Positive Lag = Metric Lags MSLP)")
        print("-"*80)
        for m in available_metrics:
            peak_lag = lag_summary[m]['peak_lag_hr']
            peak_c = lag_summary[m]['peak_corr']
            print(f" • {m:<16}: Peak Correlation = {peak_c:+6.3f} at Lag = {peak_lag:+3d} hours")

    # --- 2. Multi-panel Visualization ---
    num_plots = len(available_metrics) + (1 if mslp_col else 0)
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 2.5 * num_plots), sharex=True)
    if num_plots == 1:
        axes = [axes]

    ax_idx = 0
    if mslp_col:
        # Subplot 1: Minimum MSLP
        axes[0].plot(df['time'], df[mslp_col], color='black', linewidth=2, label=f'Min MSLP ({mslp_col})')
        axes[0].set_ylabel('MSLP (hPa)', fontweight='bold')
        axes[0].grid(True, linestyle='--', alpha=0.5)
        axes[0].legend(loc='upper right')
        axes[0].set_title('Cyclone Amphan (2020) — Trajectory Metrics vs. Storm Intensity', fontsize=14, fontweight='bold', pad=15)

        # Highlight Maximum Intensity (Minimum MSLP)
        min_mslp_idx = df[mslp_col].idxmin()
        peak_time = df.loc[min_mslp_idx, 'time']
        
        for ax in axes:
            ax.axvline(peak_time, color='red', linestyle=':', alpha=0.7, label='Peak Intensity (Min MSLP)' if ax == axes[0] else "")
        ax_idx = 1

    # Metrics Subplots
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    for idx, metric in enumerate(available_metrics):
        ax = axes[ax_idx + idx]
        c = colors[idx % len(colors)]
        ax.plot(df['time'], df[metric], color=c, linewidth=1.8, label=metric)
        ax.set_ylabel(metric, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='upper right')

    axes[-1].set_xlabel('UTC Time', fontweight='bold', fontsize=11)
    plt.xticks(rotation=30)
    plt.tight_layout()

    out_fig = output_dir / "metrics_timeseries_analysis.png"
    plt.savefig(out_fig, dpi=300)
    print(f"\n[✓] Figure generated and saved to: {out_fig}")

if __name__ == "__main__":
    csv_file = Path("./output_case_study/metrics.csv")
    out_directory = Path("./output_case_study")
    analyze_and_plot_metrics(csv_file, out_directory)