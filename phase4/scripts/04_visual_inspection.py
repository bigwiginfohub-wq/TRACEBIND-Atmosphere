import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_metrics(
    df: pd.DataFrame,
    output_dir: str = "phase4/outputs/figures",
    report_dir: str = "phase4/outputs/reports",
    normalize_type: str = "zscore", # "zscore", "relative", or None
    rolling_window: int = 5,
    anomaly_threshold: float = 3.0
):
    """
    Generates publication-quality diagnostic plots (RAW, Normalized, Derivatives, Correlation)
    and automated quality/summary reports from a metrics DataFrame.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    df = df.copy()

    # 1. Timestamp Parsing
    if "Time" in df.columns:
        try:
            df["Time"] = pd.to_datetime(df["Time"])
            df = df.sort_values("Time").reset_index(drop=True)
            time_axis = df["Time"]
        except Exception:
            time_axis = df.index
    else:
        time_axis = df.index

    numeric_cols = [c for c in df.columns if c != "Time" and pd.api.types.is_numeric_dtype(df[c])]

    # 2. Summary Statistics Report
    summary_df = pd.DataFrame({
        "mean": df[numeric_cols].mean(),
        "std": df[numeric_cols].std(),
        "CV": df[numeric_cols].std() / (df[numeric_cols].mean().abs() + 1e-8),
        "min": df[numeric_cols].min(),
        "max": df[numeric_cols].max(),
        "skew": df[numeric_cols].skew(),
        "kurtosis": df[numeric_cols].kurtosis()
    })
    summary_csv_path = os.path.join(report_dir, "timeseries_summary.csv")
    summary_df.to_csv(summary_csv_path)

    # 3. Figure 1: Raw Metrics Time Series (with Rolling Averages & Anomaly Markers)
    fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(12, 2.5 * len(numeric_cols)), sharex=True)
    if len(numeric_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, numeric_cols):
        series = df[col]
        rolling = series.rolling(window=rolling_window, min_periods=1, center=True).mean()

        # Anomaly Detection (|Δ| > anomaly_threshold * σ)
        diff = np.abs(np.diff(series, prepend=series.iloc[0]))
        diff_std = np.std(diff)
        anomalies = diff > (anomaly_threshold * diff_std)

        ax.plot(time_axis, series, color="gray", alpha=0.5, label="Raw")
        ax.plot(time_axis, rolling, color="crimson", linewidth=2, label=f"Rolling (w={rolling_window})")

        if np.any(anomalies):
            ax.scatter(time_axis[anomalies], series[anomalies], color="red", s=40, zorder=5, label=f"Anomaly (> {anomaly_threshold}σ Δ)")

        ax.set_ylabel(col, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", frameon=True)

    plt.xlabel("Time", fontweight="bold")
    plt.suptitle("Figure B7A: Raw Metric Time Series with Anomaly Markers", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    fig.savefig(os.path.join(output_dir, "Figure_B7A_Raw_Metrics.png"), dpi=300)
    fig.savefig(os.path.join(output_dir, "Figure_B7A_Raw_Metrics.pdf"))
    plt.close()

    # 4. Figure 2: Normalized Overlay Comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    for col in numeric_cols:
        series = df[col]
        if normalize_type == "zscore":
            norm_series = (series - series.mean()) / (series.std() + 1e-8)
            ylabel = "Z-Score"
        elif normalize_type == "relative":
            m0 = series.iloc[0]
            norm_series = (series - m0) / (abs(m0) + 1e-8)
            ylabel = "Relative Change (M - M0)/|M0|"
        else:
            norm_series = series
            ylabel = "Raw Value"

        ax.plot(time_axis, norm_series, label=col, linewidth=1.5)

    ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_xlabel("Time", fontweight="bold")
    ax.set_title(f"Figure B7B: Normalized Metric Dynamics ({normalize_type.upper()})", fontsize=14, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.0))
    plt.tight_layout()

    fig.savefig(os.path.join(output_dir, "Figure_B7B_Normalized_Metrics.png"), dpi=300)
    fig.savefig(os.path.join(output_dir, "Figure_B7B_Normalized_Metrics.pdf"))
    plt.close()

    # 5. Figure 3: Metric Derivatives (ΔM / Δt)
    fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(12, 2.2 * len(numeric_cols)), sharex=True)
    if len(numeric_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, numeric_cols):
        deriv = np.gradient(df[col].values)
        ax.plot(time_axis, deriv, color="teal", linewidth=1.5)
        ax.axhline(0, color="black", linestyle=":", alpha=0.7)
        ax.set_ylabel(f"Δ({col})", fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.xlabel("Time", fontweight="bold")
    plt.suptitle("Figure B7C: Temporal Metric Derivatives (Response Speeds)", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    fig.savefig(os.path.join(output_dir, "Figure_B7C_Metric_Derivatives.png"), dpi=300)
    fig.savefig(os.path.join(output_dir, "Figure_B7C_Metric_Derivatives.pdf"))
    plt.close()

    # 6. Figure 4: Metric Correlation Heatmap
    corr_matrix = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax, square=True)
    ax.set_title("Figure B7D: Inter-Metric Pearson Correlation Matrix", fontsize=12, fontweight="bold")
    plt.tight_layout()

    fig.savefig(os.path.join(output_dir, "Figure_B7D_Correlation_Heatmap.png"), dpi=300)
    fig.savefig(os.path.join(output_dir, "Figure_B7D_Correlation_Heatmap.pdf"))
    plt.close()

def plot_metrics_from_csv(csv_path: str, **kwargs):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    plot_metrics(df, **kwargs)