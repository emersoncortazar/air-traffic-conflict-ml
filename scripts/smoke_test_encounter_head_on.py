from src.sim.encounter_generators import make_head_on
from src.sim.conflict import conflict_within_horizon

def main():
    a, b = make_head_on(separation_nm=20.0, speed_kts=360.0, alt_ft=30000.0)
    conflict, ttc = conflict_within_horizon(a, b, dt_s=1.0, horizon_s=200.0)

    print("Conflict?:", conflict)
    print("TTC (s):", ttc)
    print("Expected: True, ~75.0")

if __name__ == "__main__":
    main()
