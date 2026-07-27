"""
05_run_multi_storm_cohort.py
----------------------------
Batch validator and multi-storm diagnostic analyzer for TRACEBIND Stage B.
Executes dual metric variants (intensity vs. cosine) and computes both:
 1. Direct Event Peak Difference (t_metric_peak - t_mslp_min)
 2. Cross-Correlation Lag (Shape similarity shift)
"""

import os
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Dynamic import helper for numbered script files
def import_numbered_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

base_path = Path(__file__).parent if "__file__" in locals() else Path(".")
compute_metrics_mod = import_numbered_module("compute_metrics", base_path / "03_compute_metrics.py")
plot_timeseries_mod = import_numbered_module("plot_timeseries", base_path / "04_plot_timeseries.py")

process_era5_case_study = compute_metrics_mod.process_era5_case_study
find_mslp_column = plot_timeseries_mod.find_mslp_column
calculate_lagged_correlations = plot_timeseries_mod.calculate_lagged_correlations

STORM_COHORT = [
    {"name": "Amphan", "file": "era5_amphan_72h.nc"},
    {"name": "Fani",   "file": "era5_fani_72h.nc"},
    {"name": "Mocha",  "file": "era5_mocha_72h.nc"},
    {"name": "Yaas",   "file": "era5_yaas_72h.nc"},
    {"name": "Sidr",   "file": "era5_sidr_72h.nc"},
    {"name": "Nargis", "file": "era5_nargis_72h.nc"},
]

def analyze_single_storm(storm_info: dict, base_dir: Path) -> dict:
    storm_name = storm_info["name"]
    nc_path = base_dir / storm_info["file"]
    out_dir = base_dir / "output_cohort" / storm_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if not nc_path.exists():
        print(f"[!] Skipping {storm_name}: '{storm_info['file']}' not found in directory.")
        return None

    print(f"\n{'='*80}\nPROCESSING STORM: {storm_name.upper()}\n{'='*80}")
    
    # 1. Process ERA5 and extract metrics table
    df = process_era5_case_study(nc_path, out_dir, window_deg=20.0)
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    
    mslp_col = find_mslp_column(df)
    
    # 2. Determine MSLP Minimum Reference Event
    min_mslp_idx = df[mslp_col].idxmin()
    mslp_min_val = df.loc[min_mslp_idx, mslp_col]
    mslp_min_time = df.loc[min_mslp_idx, 'time']

    metrics = ['gradient_energy', 'tb_v2_intensity', 'tb_v2_cosine', 'tb_v1', 'morans_i', 'spatial_entropy']
    available_metrics = [m for m in metrics if m in df.columns]

    storm_summary = {
        "Storm": storm_name,
        "Min_MSLP_hPa": mslp_min_val,
        "Min_MSLP_Time": mslp_min_time.strftime("%Y-%m-%d %H:%M UTC"),
    }

    timing_records = []

    # 3. Analyze each metric using Dual Timing Methods
    for m in available_metrics:
        if m in ['morans_i', 'spatial_entropy']:
            peak_idx = df[m].idxmin()
        else:
            peak_idx = df[m].idxmax()
            
        peak_val = df.loc[peak_idx, m]
        peak_time = df.loc[peak_idx, 'time']
        
        direct_dt_hrs = (peak_time - mslp_min_time).total_seconds() / 3600.0

        lag_hrs, corrs = calculate_lagged_correlations(df[mslp_col], df[m], max_lag_steps=8)
        xcorr_peak_idx = np.argmax(np.abs(corrs))
        xcorr_lag_hr = lag_hrs[xcorr_peak_idx]
        xcorr_peak_val = corrs[xcorr_peak_idx]

        storm_summary[f"{m}_PeakVal"] = peak_val
        storm_summary[f"{m}_PeakTime"] = peak_time.strftime("%Y-%m-%d %H:%M UTC")
        storm_summary[f"{m}_DirectLeadLag_h"] = direct_dt_hrs
        storm_summary[f"{m}_XCorrLag_h"] = xcorr_lag_hr
        storm_summary[f"{m}_XCorr"] = xcorr_peak_val

        timing_records.append({
            "Metric": m,
            "Peak_Value": peak_val,
            "Peak_Time": peak_time,
            "Direct_Event_Delta_Hours": direct_dt_hrs,
            "CrossCorr_Lag_Hours": xcorr_lag_hr,
            "Max_CrossCorr": xcorr_peak_val
        })

    pd.DataFrame(timing_records).to_csv(out_dir / "lag_table.csv", index=False)
    plot_storm_timeseries(df, mslp_col, available_metrics, mslp_min_time, storm_name, out_dir / "timeseries.png")

    return storm_summary

def plot_storm_timeseries(df, mslp_col, metrics, peak_time, storm_name, out_path):
    num_plots = len(metrics) + 1
    fig, axes = plt.subplots(num_plots, 1, figsize=(11, 2.2 * num_plots), sharex=True)
    
    axes[0].plot(df['time'], df[mslp_col], color='black', lw=2, label=f"MSLP ({storm_name})")
    axes[0].set_ylabel("MSLP (hPa)", fontweight='bold')
    axes[0].grid(True, ls='--', alpha=0.5)
    axes[0].legend(loc='upper right')
    axes[0].set_title(f"Cyclone {storm_name} — Kinematic Dynamics & Timing", fontweight='bold', fontsize=13)

    for ax in axes:
        ax.axvline(peak_time, color='red', ls=':', alpha=0.8, label="Min MSLP Peak" if ax == axes[0] else "")

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for idx, m in enumerate(metrics):
        ax = axes[idx + 1]
        ax.plot(df['time'], df[m], color=colors[idx % len(colors)], lw=1.8, label=m)
        ax.set_ylabel(m, fontweight='bold', fontsize=9)
        ax.grid(True, ls='--', alpha=0.5)
        ax.legend(loc='upper right')

    axes[-1].set_xlabel("UTC Time", fontweight='bold')
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def generate_cohort_summary_visuals(summary_df: pd.DataFrame, summary_dir: Path):
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    direct_lag_cols = [c for c in summary_df.columns if c.endswith('_DirectLeadLag_h')]
    
    if direct_lag_cols:
        plt.figure(figsize=(10, 5))
        plot_data = []
        labels = []
        
        for col in direct_lag_cols:
            metric_label = col.replace('_DirectLeadLag_h', '')
            vals = summary_df[col].dropna().values
            if len(vals) > 0:
                plot_data.append(vals)
                labels.append(metric_label)

        if plot_data:
            plt.axhline(0, color='red', linestyle='--', alpha=0.7, label='Minimum MSLP Time (0h)')
            
            # Robust matplotlib boxplot tick_labels/labels handling
            try:
                plt.boxplot(plot_data, tick_labels=labels, patch_artist=True, boxprops=dict(facecolor='#d9e5f5'))
            except TypeError:
                plt.boxplot(plot_data, labels=labels, patch_artist=True, boxprops=dict(facecolor='#d9e5f5'))
                
            plt.ylabel("Lead (-) / Lag (+) Hours Relative to MSLP Peak", fontweight='bold')
            plt.title("Cross-Storm Lead/Lag Distribution (Direct Event Timing)", fontweight='bold', fontsize=12)
            plt.grid(True, ls='--', alpha=0.5)
            plt.tight_layout()
            plt.savefig(summary_dir / "lead_lag_distribution.png", dpi=300)
            plt.close()

def run_cohort_pipeline():
    base_dir = Path(".")
    summary_dir = base_dir / "output_cohort" / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    summary_records = []

    for storm in STORM_COHORT:
        res = analyze_single_storm(storm, base_dir)
        if res:
            summary_records.append(res)

    if summary_records:
        summary_df = pd.DataFrame(summary_records)
        summary_csv = summary_dir / "cross_storm_synthesis.csv"
        summary_df.to_csv(summary_csv, index=False)
        
        generate_cohort_summary_visuals(summary_df, summary_dir)

        print("\n" + "="*80)
        print("                  CROSS-STORM COHORT SYNTHESIS TABLE                  ")
        print("="*80)
        print(summary_df.to_string(index=False))
        print(f"\n[✓] Cohort evaluation completed. Synthesis saved to: {summary_csv}")
    else:
        print("\n[!] No storm netCDF files were found. Place era5_*.nc files in the working directory.")

if __name__ == "__main__":
    run_cohort_pipeline()