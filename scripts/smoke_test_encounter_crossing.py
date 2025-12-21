from src.sim.encounter_generators import make_crossing
from src.sim.conflict import conflict_within_horizon

def main():
    # case 1: same altitude then should conflict if timing lines up
    a, b = make_crossing(start_offset_nm=10.0, speed_kts=360.0, alt_ft_a=30000.0, alt_ft_b=30000.0)
    conflict, ttc = conflict_within_horizon(a, b, dt_s=1.0, horizon_s=200.0)

    print("CASE 1 (same altitude)")
    print("Conflict?:", conflict)
    print("TTC (s):", ttc)

    # case 2: different altitude then should NOT conflict even though they cross horizontally
    a2, b2 = make_crossing(start_offset_nm=10.0, speed_kts=360.0, alt_ft_a=30000.0, alt_ft_b=32000.0)
    conflict2, ttc2 = conflict_within_horizon(a2, b2, dt_s=1.0, horizon_s=200.0)

    print("\nCASE 2 (2000 ft vertical separation)")
    print("Conflict?:", conflict2)
    print("TTC (s):", ttc2)
    print("\nExpected:")
    print("- Case 1 likely True (they meet near the intersection).")
    print("- Case 2 should be False (vertical separation >= 1000 ft).")

if __name__ == "__main__":
    main()
