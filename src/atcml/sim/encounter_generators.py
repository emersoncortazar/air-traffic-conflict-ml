from atcml.sim.dynamics import aircraft_state

def make_head_on(separation_nm: float = 20.0, speed_kts: float = 360.0, alt_ft: float = 30000.0):
    """
    Returns two aircraft states flying directly toward each other along the x-axis
    Aircraft A starts at x=0 heading +x (0 deg)
    Aircraft B starts at x=separation_nm heading -x (180 deg)
    """
    a = aircraft_state(
        x_nm=0.0,
        y_nm=0.0,
        alt_ft=alt_ft,
        speed_kts=speed_kts,
        heading_deg=0.0,
        vz_fpm=0.0
    )
    b = aircraft_state(
        x_nm=separation_nm,
        y_nm=0.0,
        alt_ft=alt_ft,
        speed_kts=speed_kts,
        heading_deg=180.0,
        vz_fpm=0.0
    )
    return a, b

def make_crossing(
    start_offset_nm: float = 10.0,
    speed_kts: float = 360.0,
    alt_ft_a: float = 30000.0,
    alt_ft_b: float = 30000.0,
    b_extra_offset_nm: float = 0.0,
):
    """
    Two aircraft crossing at (0,0):
    - Aircraft A starts at (-start_offset_nm, 0) heading east (0 deg)
    - Aircraft B starts at (0, -(start_offset_nm + b_extra_offset_nm)) heading north (90 deg)

    b_extra_offset_nm > 0 makes B start farther away (arrives later).
    """
    a = aircraft_state(
        x_nm=-start_offset_nm,
        y_nm=0.0,
        alt_ft=alt_ft_a,
        speed_kts=speed_kts,
        heading_deg=0.0,
        vz_fpm=0.0
    )
    b = aircraft_state(
        x_nm=0.0,
        y_nm=-(start_offset_nm + b_extra_offset_nm),
        alt_ft=alt_ft_b,
        speed_kts=speed_kts,
        heading_deg=90.0,
        vz_fpm=0.0
    )
    return a, b
