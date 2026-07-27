import cdsapi
import xarray as xr
from pathlib import Path

out_dir = Path("./data")
out_dir.mkdir(parents=True, exist_ok=True)

f_press = out_dir / "temp_pressure.nc"
f_surf = out_dir / "temp_surface.nc"
out_final = out_dir / "era5_amphan_72h.nc"

c = cdsapi.Client()

print("1/3 Downloading ERA5 500 hPa Geopotential Height...")
c.retrieve(
    'reanalysis-era5-pressure-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': ['geopotential'],
        'pressure_level': ['500'],
        'year': '2020',
        'month': '05',
        'day': ['17', '18', '19', '20'],
        'time': [f"{h:02d}:00" for h in range(24)],
        'area': [28, 78, 5, 98],
    },
    str(f_press)
)

print("2/3 Downloading ERA5 Surface Mean Sea Level Pressure...")
c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': ['mean_sea_level_pressure'],
        'year': '2020',
        'month': '05',
        'day': ['17', '18', '19', '20'],
        'time': [f"{h:02d}:00" for h in range(24)],
        'area': [28, 78, 5, 98],
    },
    str(f_surf)
)

print("3/3 Merging datasets into era5_amphan_72h.nc...")
ds_p = xr.open_dataset(f_press)
ds_s = xr.open_dataset(f_surf)

ds_merged = xr.merge([ds_p, ds_s])
ds_merged.to_netcdf(out_final)

# Cleanup temporary files
ds_p.close()
ds_s.close()
f_press.unlink()
f_surf.unlink()

print(f"\nSUCCESS! File saved to: {out_final.resolve()}")