import numpy as np
from src.sim.dynamics import step_state

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


def conflict_within_horizon(state0, state1, dt_s: float, horizon_s: float):
    """
    Simulate state0 and state1 forward in time and return whether they ever
    violate separation within horizon_s seconds

    Returns:
        (conflict_bool, time_to_first_conflict_s)
        If no conflict: (False, -1.0)
    """
    steps = int(horizon_s / dt_s)

    s0 = state0
    s1 = state1

    for i in range(steps):
        # Step both aircraft forward by dt_s
        s0 = step_state(s0, dt_s)
        s1 = step_state(s1, dt_s)

        # Check if they are in loss-of-separation now
        if is_loss_of_separation(s0, s1):
            return True, (i + 1) * dt_s

    return False, -1.0
