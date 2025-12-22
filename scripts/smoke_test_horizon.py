from atcml.sim.dynamics import aircraft_state
from atcml.sim.conflict import conflict_within_horizon

def main():
    # Aircraft A starts at x=0 moving east at 360 kts (0.1 NM/s)
    a = aircraft_state(
        x_nm=0.0, y_nm=0.0, alt_ft=30000.0,
        speed_kts=360.0, heading_deg=0.0, vz_fpm=0.0
    )

    # Aircraft B starts at x=20 moving west at 360 kts
    b = aircraft_state(
        x_nm=20.0, y_nm=0.0, alt_ft=30000.0,
        speed_kts=360.0, heading_deg=180.0, vz_fpm=0.0
    )

    # They are closing speed: 0.1 + 0.1 = 0.2 NM/s
    # Initial horizontal separation: 20 NM
    # They reach 5 NM separation after closing 15 NM
    # time = 15 / 0.2 = 75 seconds
    conflict, ttc = conflict_within_horizon(a, b, dt_s=1.0, horizon_s=200.0)

    print("Conflict within horizon?:", conflict)
    print("Time to first conflict (s):", ttc)
    print("\nExpected: conflict True, time around 75 seconds.")

if __name__ == "__main__":
    main()
