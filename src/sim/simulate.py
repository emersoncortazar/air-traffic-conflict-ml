from src.sim.dynamics import step_state

def simulate_pair(state0, state1, dt_s: float, horizon_s: float):
    """
    Simulate two aircraft forward in time and record their trajectories.

    Returns:
        rows: a list of dicts, one per timestep, containing:
              t_s, a_x_nm, a_y_nm, a_alt_ft, b_x_nm, b_y_nm, b_alt_ft
    """
    steps = int(horizon_s / dt_s)

    s0 = state0
    s1 = state1

    rows = []

    for i in range(steps + 1):
        # time in seconds at this recorded frame
        t_s = i * dt_s

        # record current states
        rows.append({
            "t_s": float(t_s),
            "a_x_nm": float(s0.x_nm),
            "a_y_nm": float(s0.y_nm),
            "a_alt_ft": float(s0.alt_ft),
            "b_x_nm": float(s1.x_nm),
            "b_y_nm": float(s1.y_nm),
            "b_alt_ft": float(s1.alt_ft),
        })

        # advance states for next step (but don't step after the final record)
        if i < steps:
            s0 = step_state(s0, dt_s)
            s1 = step_state(s1, dt_s)

    return rows
