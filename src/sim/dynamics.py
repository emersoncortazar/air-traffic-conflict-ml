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
    """
    Minimal aircraft state for a kinematic simulation
    """
    x_nm: float # position in x direction in nautical miles
    y_nm: float # position in y direction in nautical miles
    alt_ft: float # altitude in feet
    speed_kts: float # ground speed in knots
    heading_deg: float # heading in degrees (0 = +x, 90 = +y)
    vz_fpm: float = 0.0 # vertical speed in feet per minute (+ means up)

    def xy_position_vector(self) -> np.ndarray:
        return np.array([self.x_nm, self.y_nm], dtype=float)
    
    def full_vector(self) -> np.ndarray:
        return np.array([self.x_nm, self.y_nm, self.alt_ft], dtype=float)
    

def step_state(state: aircraft_state, dt_s: float) -> aircraft_state:
    """
    Docstring for step_state
    
    :param state: Description
    :type state: aircraft_state
    :param dt_s: Description
    :type dt_s: float
    :return: Description
    :rtype: aircraft_state
    """
    heading_rad = np.deg2rad(state.heading_deg)

    v_nm_per_s = state.speed_kts * kts_to_nm_per_s
    dx = v_nm_per_s * np.cos(heading_rad) * dt_s
    dy = v_nm_per_s * np.sin(heading_rad) * dt_s

    vz_ft_per_s = state.vz_fpm * fpm_to_ft_per_s
    dalt = vz_ft_per_s * dt_s

    return aircraft_state(
        x_nm=state.x_nm + dx,
        y_nm=state.y_nm + dy,
        alt_ft=state.alt_ft + dalt,
        speed_kts=state.speed_kts,
        heading_deg=state.heading_deg,
        vz_fpm=state.vz_fpm
    )
