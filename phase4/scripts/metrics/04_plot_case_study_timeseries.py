"""
04_plot_case_study_timeseries.py
--------------------------------
Generates a synchronized publication-quality multi-panel time-series figure
comparing storm pressure dynamics directly against spatial organization metrics.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

def plot_case_study_timeseries(csv_path: Path, output_fig: Path):
    df = pd.read_csv(csv_path)
    df['time'] = pd.to_datetime(df['time'])
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    
    # --- Panel 1: Meteorological Anchor ---
    ax1 = axes[0]
    ax1.plot(df['time'], df['min_mslp_hpa'], color='black', linewidth=2.5, label='Min MSLP (hPa)')
    ax1.set_ylabel('MSLP (hPa)', fontweight='bold')
    ax1.invert_yaxis()  # Invert pressure so intensification goes UP visually
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper left')
    ax1.set_title('Cyclone Amphan 72-Hour Intensification: Spatial Metric Dynamics', fontsize=14, fontweight='bold', pad=12)
    
    # --- Panel 2: Local Gradient Alignment vs Autocorrelation ---
    ax2 = axes[1]
    ax2.plot(df['time'], df['tb_v2'], color='crimson', linewidth=2.0, label='TB-v2 (Local Gradient Coherence)')
    ax2.plot(df['time'], df['moran_i'], color='royalblue', linewidth=2.0, linestyle='--', label="Moran's I (Spatial Autocorrelation)")
    ax2.set_ylabel('Metric Score', fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper left')
    
    # --- Panel 3: Macro Distribution vs Spatial Entropy ---
    ax3 = axes[2]
    ax3.plot(df['time'], df['tb_v1'], color='darkgreen', linewidth=2.0, label='TB-v1 (Global Variance Index)')
    ax3_twin = ax3.twinx()
    ax3_twin.plot(df['time'], df['spatial_entropy'], color='purple', linewidth=1.8, linestyle=':', label='Spatial Entropy')
    ax3.set_ylabel('TB-v1 Score', fontweight='bold', color='darkgreen')
    ax3_twin.set_ylabel('Entropy (nats)', fontweight='bold', color='purple')
    ax3.grid(True, linestyle='--', alpha=0.5)
    
    # Combine legends for twin axis
    lines_3, labels_3 = ax3.get_legend_handles_labels()
    lines_3t, labels_3t = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines_3 + lines_3t, labels_3 + labels_3t, loc='upper left')
    
    # --- Panel 4: Physical Gradient Energy ---
    ax4 = axes[3]
    ax4.plot(df['time'], df['gradient_energy'], color='darkorange', linewidth=2.0, label='Gradient Energy ($m^2 / m^2$)')
    ax4.set_ylabel('Gradient Energy', fontweight='bold')
    ax4.grid(True, linestyle='--', alpha=0.5)
    ax4.legend(loc='upper left')
    
    # Formatting X-Axis
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M UTC'))
    ax4.set_xlabel('Time (UTC)', fontweight='bold', labelpad=8)
    
    plt.tight_layout()
    plt.savefig(output_fig, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[COMPLETE] Timeseries figure successfully generated at: {output_fig}")

if __name__ == "__main__":
    metrics_file = Path("./output/amphan_case_study/metrics.csv")
    figure_file = Path("./output/amphan_case_study/timeseries_comparison.png")
    plot_case_study_timeseries(metrics_file, figure_file)