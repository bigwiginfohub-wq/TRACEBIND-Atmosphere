"""
04_plot_case_study_timeseries.py
--------------------------------
Generates Phase B-1 multi-panel timeseries plot comparing Minimum MSLP against
full-domain TRACEBIND v1/v2, Moran's I, Spatial Entropy, and Gradient Energy.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

def plot_full_domain_timeseries(csv_path: Path, output_fig: Path):
    df = pd.read_csv(csv_path)
    df['time'] = pd.to_datetime(df['time'])
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    
    # --- Panel 1: Minimum MSLP (Storm Core Anchor) ---
    ax1 = axes[0]
    ax1.plot(df['time'], df['min_mslp_hpa'], color='black', linewidth=2.5, label='Min MSLP (hPa)')
    ax1.set_ylabel('MSLP (hPa)', fontweight='bold')
    ax1.invert_yaxis()  # Invert pressure so deepening pressure goes UP
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper left')
    ax1.set_title('Phase B-1: Full-Domain ERA5 Metric Dynamics (Cyclone Amphan, May 17–20, 2020)', fontsize=13, fontweight='bold', pad=10)
    
    # --- Panel 2: Local Gradient Alignment vs Autocorrelation ---
    ax2 = axes[1]
    ax2.plot(df['time'], df['tb_v2'], color='crimson', linewidth=2.0, label='TB-v2 (Local Gradient Coherence)')
    ax2.plot(df['time'], df['moran_i'], color='royalblue', linewidth=2.0, linestyle='--', label="Moran's I (Spatial Autocorrelation)")
    ax2.set_ylabel('Metric Score', fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper left')
    
    # --- Panel 3: Global Variance vs Entropy ---
    ax3 = axes[2]
    ax3.plot(df['time'], df['tb_v1'], color='darkgreen', linewidth=2.0, label='TB-v1 (Global Summary Index)')
    ax3_twin = ax3.twinx()
    ax3_twin.plot(df['time'], df['spatial_entropy'], color='purple', linewidth=1.8, linestyle=':', label='Spatial Entropy')
    ax3.set_ylabel('TB-v1 Score', fontweight='bold', color='darkgreen')
    ax3_twin.set_ylabel('Entropy (nats)', fontweight='bold', color='purple')
    ax3.grid(True, linestyle='--', alpha=0.5)
    
    # Merge legends
    lines_3, labels_3 = ax3.get_legend_handles_labels()
    lines_3t, labels_3t = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines_3 + lines_3t, labels_3 + labels_3t, loc='upper left')
    
    # --- Panel 4: Physical Gradient Energy ---
    ax4 = axes[3]
    ax4.plot(df['time'], df['gradient_energy'], color='darkorange', linewidth=2.0, label='Gradient Energy ($m^2 / m^2$)')
    ax4.set_ylabel('Gradient Energy', fontweight='bold')
    ax4.grid(True, linestyle='--', alpha=0.5)
    ax4.legend(loc='upper left')
    
    # Format X-Axis
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M UTC'))
    ax4.set_xlabel('Time (UTC)', fontweight='bold', labelpad=8)
    
    plt.tight_layout()
    plt.savefig(output_fig, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Phase B-1 Timeseries plot generated at: {output_fig}")

if __name__ == "__main__":
    metrics_csv = Path("./output_phase_b1/metrics.csv")
    out_png = Path("./output_phase_b1/phase_b1_timeseries.png")
    plot_full_domain_timeseries(metrics_csv, out_png)