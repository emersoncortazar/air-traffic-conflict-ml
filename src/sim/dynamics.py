# note that a minimal aircraft model will have:
# position (x, y) in nautical miles
# altitude
# speed in knots (nautical miles per hour)
# heading in degrees
# vertical speed (feet per minute)

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

# unit converters
kts_to_nm_per_s = 1 / 3600 #knots = nautical miles / hour
fpm_to_ft_per_s = 1 / 60 # feet / minute

@dataclass
class aircraft_state:
    x_nm: float # position in x direction in nautical miles
    y_nm: float # position in y direction in nautical miles
    alt_ft: float # altitude in feet
    speed_kts: float # ground speed in knots
    heading_deg: float # heading in degrees (0 = +x, 90 = +y)
    vz_fpm: float = 0.0 # vertical speed in feet per minute (+ means up)

    def position_vector(self) -> np.ndarray:
        return np.array([self.x_nm, self.y_nm], dtype=float)
    
    def full_vector(self) -> np.ndarray:
        return np.array([self.x_nm, self.y_nm, self.alt_ft], dtype=float)
    
    