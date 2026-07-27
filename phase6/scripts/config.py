"""
TRACEBIND v1.0 Executable Feature Freeze & Global Constants
"""
import sys

TRACEBIND_VERSION = "1.0.0"

# FROZEN VECTOR SPECIFICATION
DESCRIPTOR_SET = (
    "GE",        # Gradient Energy Density
    "LE",        # Laplacian Energy Density
    "C_orient",  # Global Phase Coherence
    "A_radial",  # Radial Symmetry / Anisotropy
    "S_orient",  # Shear-Oriented Anisotropy
)

assert len(DESCRIPTOR_SET) == 5, "Feature freeze violation: Vector dimension must remain 5."

# PHYSICAL BOUNDS FOR ERA5 SANITY CHECKS
PHYSICAL_LIMITS = {
    "MSLP_HPA_MIN": 850.0,    # Extreme TC central pressure lower bound (hPa)
    "MSLP_HPA_MAX": 1080.0,   # High-pressure system upper bound (hPa)
    "WIND_MS_MAX": 120.0,     # Max plausible wind magnitude (m/s)
    "VORTICITY_MAX": 1e-2,    # Plausible relative vorticity upper bound (1/s)
}

# EXPECTED COORDINATE KEYS
LAT_KEYS = {"latitude", "lat", "NAV_LAT"}
LON_KEYS = {"longitude", "lon", "NAV_LON"}
TIME_KEYS = {"time", "valid_time"}
VAR_ALIASES = {
    "msl": "msl",
    "mean_sea_level_pressure": "msl",
    "u10": "u10",
    "v10": "v10",
    "vo": "vo"
}