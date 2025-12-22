from atcml.sim.encounter_generators import make_crossing
from atcml.sim.conflict import conflict_within_horizon
from pathlib import Path
import csv

def main():
    total = 0
    conflicts = 0

    rows = []  # store scenario results here

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

        # store a row
        rows.append({
            "b_extra_offset_nm": float(extra_offset),
            "conflict": bool(conflict),
            "ttc_s": float(ttc),
        })

        print(
            f"B extra offset: {extra_offset:>2} NM | "
            f"Conflict: {conflict} | TTC: {ttc}"
        )

    print("\nSummary:")
    print("Total scenarios:", total)
    print("Conflicts:", conflicts)
    print("Non-conflicts:", total - conflicts)

    print("\nStored rows:", len(rows))
    print("First row example:", rows[0])

    out_path = Path("data") / "crossing_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)  # creates ./data if missing

    fieldnames = ["b_extra_offset_nm", "conflict", "ttc_s"]

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\nWrote CSV to:", out_path)

if __name__ == "__main__":
    main()
