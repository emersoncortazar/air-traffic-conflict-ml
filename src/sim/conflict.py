import numpy as np

# Separation thresholds
HORIZONTAL_THRESHOLD_NM = 5.0
VERTICAL_THRESHOLD_FT = 1000.0


def horizontal_separation_nm(state0, state1) -> float:
    """
    Horizontal distance between aircraft in nautical miles
    Uses x_nm and y_nm from each aircraft_state
    """
    dx = state0.x_nm - state1.x_nm
    dy = state0.y_nm - state1.y_nm
    return float(np.sqrt(dx * dx + dy * dy))


def vertical_separation_ft(state0, state1) -> float:
    """
    Vertical separation between aircraft in feet
    """
    return float(abs(state0.alt_ft - state1.alt_ft))


def is_loss_of_separation(state0, state1) -> bool:
    """
    Returns True if aircraft are too close horizontally AND vertically
    """
    h = horizontal_separation_nm(state0, state1)
    v = vertical_separation_ft(state0, state1)
    return (h < HORIZONTAL_THRESHOLD_NM) and (v < VERTICAL_THRESHOLD_FT)
