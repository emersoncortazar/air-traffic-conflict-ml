from src.sim.encounter_generators import make_crossing
from src.sim.conflict import conflict_within_horizon

def main():
    total = 0
    conflicts = 0

    # try different timing offsets for B
    for extra_offset in range(0, 21, 2):  # 0, 2, 4, ...
        a, b = make_crossing(
            start_offset_nm=10.0,
            speed_kts=360.0,
            alt_ft_a=30000.0,
            alt_ft_b=30000.0,
            b_extra_offset_nm=float(extra_offset)
        )

        conflict, ttc = conflict_within_horizon(
            a, b, dt_s=1.0, horizon_s=300.0
        )

        total += 1
        if conflict:
            conflicts += 1

        print(
            f"B extra offset: {extra_offset:>2} NM | "
            f"Conflict: {conflict} | TTC: {ttc}"
        )

    print("\nSummary:")
    print("Total scenarios:", total)
    print("Conflicts:", conflicts)
    print("Non-conflicts:", total - conflicts)

if __name__ == "__main__":
    main()
