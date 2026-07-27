import os
import sys
import xarray as xr
import numpy as np
import pandas as pd

def load_and_inspect_era5(
    file_path: str,
    var_name: str = "z",
    lat_bounds: tuple = None,  # e.g., (10, 50)
    lon_bounds: tuple = None,  # e.g., (60, 100)
    level: float = 500         # e.g., 500 hPa if 4D dataset
) -> tuple[np.ndarray, dict]:
    """
    Loads ERA5 NetCDF, handles slicing, logs a data quality report,
    and returns a clean 2D/3D numpy array [time, lat, lon].
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"ERA5 file not found: {file_path}")

    ds = xr.open_dataset(file_path)

    # Handle pressure levels if present
    if "level" in ds[var_name].dims or "plev" in ds[var_name].dims:
        p_dim = "level" if "level" in ds[var_name].dims else "plev"
        ds_var = ds[var_name].sel({p_dim: level}, method="nearest")
    else:
        ds_var = ds[var_name]

    # Handle Geographic Slicing
    if lat_bounds:
        # Standardize lat indexing depending on ascending/descending order in NetCDF
        lat_min, lat_max = min(lat_bounds), max(lat_bounds)
        if ds_var.latitude[0] > ds_var.latitude[-1]:
            ds_var = ds_var.sel(latitude=slice(lat_max, lat_min))
        else:
            ds_var = ds_var.sel(latitude=slice(lat_min, lat_max))

    if lon_bounds:
        lon_min, lon_max = min(lon_bounds), max(lon_bounds)
        ds_var = ds_var.sel(longitude=slice(lon_min, lon_max))

    # Extract raw values and handle missing values (fill with nan or interpolate)
    data_array = ds_var.values
    missing_count = int(np.isnan(data_array).sum())

    # Build Data Quality Report Dictionary
    report = {
        "Dataset Name": os.path.basename(file_path),
        "Variable": ds_var.name,
        "Long Name": ds_var.attrs.get("long_name", "N/A"),
        "Units": ds_var.attrs.get("units", "N/A"),
        "Time Steps": int(ds_var.sizes.get("time", 1)),
        "Time Range": f"{str(ds.time.values[0])[:19]} to {str(ds.time.values[-1])[:19]}" if "time" in ds else "N/A",
        "Spatial Grid Shape": f"{ds_var.sizes.get('latitude', 0)} x {ds_var.sizes.get('longitude', 0)}",
        "Lat Bounds": f"[{float(ds_var.latitude.min()):.2f}°, {float(ds_var.latitude.max()):.2f}°]",
        "Lon Bounds": f"[{float(ds_var.longitude.min()):.2f}°, {float(ds_var.longitude.max()):.2f}°]",
        "Missing Values": missing_count,
        "Min": float(np.nanmin(data_array)),
        "Max": float(np.nanmax(data_array)),
        "Mean": float(np.nanmean(data_array)),
        "Std Dev": float(np.nanstd(data_array)),
    }

    return data_array, report


def print_quality_report(report: dict):
    """Prints a formatted markdown table for the Data Quality Report."""
    print("\n" + "="*50)
    print("        ERA5 DATA QUALITY REPORT")
    print("="*50)
    df = pd.DataFrame(list(report.items()), columns=["Property", "Value"])
    print(df.to_markdown(index=False))
    print("="*50 + "\n")


if __name__ == "__main__":
    # Example usage / test harness
    print("`load_era5.py` module ready for data ingestion.")