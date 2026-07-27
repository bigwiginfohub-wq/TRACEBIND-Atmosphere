"""
Phase 6A: Enums and Metadata Schema
"""
from enum import Enum
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Tuple

class SystemClass(str, Enum):
    TC = "TROPICAL_CYCLONE"
    ETC = "EXTRATROPICAL_CYCLONE"
    MONSOON = "MONSOON_LOW"
    FRONT = "FRONTAL_SYSTEM"
    WEAK_DEP = "WEAK_DEPRESSION"
    RANDOM = "RANDOM_BACKGROUND"

class OceanBasin(str, Enum):
    NI = "NORTH_INDIAN"
    SI = "SOUTH_INDIAN"
    NA = "NORTH_ATLANTIC"
    EP = "EAST_PACIFIC"
    WP = "WEST_PACIFIC"
    SP = "SOUTH_PACIFIC"
    GLOBAL = "GLOBAL_MIDLAT"

@dataclass
class SystemMetadata:
    # Core Identification
    system_id: str
    system_name: str
    event_year: int
    system_class: SystemClass
    basin: OceanBasin
    
    # Coordinates & Dimensions
    bounding_box: Tuple[float, float, float, float]  # (lat_min, lat_max, lon_min, lon_max)
    analysis_time: str                                # ISO UTC String
    storm_center_lat: Optional[float] = None
    storm_center_lon: Optional[float] = None
    
    # Meteorological Context (For future Phase 7 & paper)
    min_pressure_hpa: Optional[float] = None
    max_wind_kt: Optional[float] = None
    
    # Provenance & File Tracking
    source: str = "ERA5_Reanalysis"
    era5_filename: str = ""
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["system_class"] = self.system_class.value
        d["basin"] = self.basin.value
        return d

    @classmethod
    from_dict(cls, d: Dict[str, Any]) -> "SystemMetadata":
        d_copy = d.copy()
        d_copy["system_class"] = SystemClass(d_copy["system_class"])
        d_copy["basin"] = OceanBasin(d_copy["basin"])
        d_copy["bounding_box"] = tuple(d_copy["bounding_box"])
        return cls(**d_copy)