import csv
from pathlib import Path

from src.sim.encounter_generators import make_head_on
from src.sim.simulate import simulate_pair

def main():
    a, b = make_head_on(separation_nm=20.0, speed_kts=360.0, alt_ft=30000.0)
    rows = simulate_pair(a, b, dt_s=1.0, horizon_s=120.0)

    out_path = Path("data") / "trajectories_head_on.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Wrote:", out_path)
    print("Rows:", len(rows))
    print("First row:", rows[0])
    print("Last row:", rows[-1])

if __name__ == "__main__":
    main()
