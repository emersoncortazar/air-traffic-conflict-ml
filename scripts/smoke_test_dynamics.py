from src.sim.dynamics import aircraft_state, step_state

def main():
    # test of starting at the origin, altitude of of 30,000 ft
    # speed will be 360 knots
    # heading will be 0 degrees (east)

    s = aircraft_state(
        x_nm=0,
        y_nm=0,
        alt_ft=30000,
        speed_kts=360,
        heading_deg=0,
        vz_fpm=0
    )

    # move the object forward by 10 seconds
    dt_s = 10
    s2 = step_state(s, dt_s = dt_s)

    print("start state: ", s)
    print("after 10s: ", s2)

    print("\nexpected: x_nm should be ~1.0, y_nm ~ 0.0, alt_ft 30,000")

if __name__ == "__main__":
    main()