import csv
from pathlib import Path
import pandas as pd

from src.sim.encounter_generators import make_head_on, make_crossing
from src.sim.simulate import simulate_pair
from src.sim.conflict import conflict_within_horizon

def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

def main():
    dt_s = 1.0
    horizon_s = 180.0

    traj_rows = []
    label_rows = []

    scenario_id = 0

    # group 1: head-on scenarios with different starting separations
    for sep_nm in [10.0, 15.0, 20.0, 25.0]:
        a, b = make_head_on(separation_nm=sep_nm, speed_kts=360.0, alt_ft=30000.0)

        conflict, ttc = conflict_within_horizon(a, b, dt_s=dt_s, horizon_s=horizon_s)
        rows = simulate_pair(a, b, dt_s=dt_s, horizon_s=horizon_s)

        # attach scenario_id and encounter_type to every trajectory row
        for r in rows:
            r["scenario_id"] = scenario_id
            r["encounter_type"] = "head_on"
        traj_rows.extend(rows)

        label_rows.append({
            "scenario_id": scenario_id,
            "encounter_type": "head_on",
            "conflict": bool(conflict),
            "ttc_s": float(ttc),
            "sep_nm": float(sep_nm),
        })

        scenario_id += 1

    # group 2: crossing scenarios with different timing offsets
    for extra_offset in [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]:
        a, b = make_crossing(
            start_offset_nm=10.0,
            speed_kts=360.0,
            alt_ft_a=30000.0,
            alt_ft_b=30000.0,
            b_extra_offset_nm=extra_offset
        )

        conflict, ttc = conflict_within_horizon(a, b, dt_s=dt_s, horizon_s=horizon_s)
        rows = simulate_pair(a, b, dt_s=dt_s, horizon_s=horizon_s)

        for r in rows:
            r["scenario_id"] = scenario_id
            r["encounter_type"] = "crossing"
        traj_rows.extend(rows)

        label_rows.append({
            "scenario_id": scenario_id,
            "encounter_type": "crossing",
            "conflict": bool(conflict),
            "ttc_s": float(ttc),
            "b_extra_offset_nm": float(extra_offset),
        })

        scenario_id += 1

    # write outputs as csv
    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)

    traj_df = pd.DataFrame(traj_rows)
    labels_df = pd.DataFrame(label_rows)

    traj_path = out_dir / "trajectories.csv"
    labels_path = out_dir / "labels.csv"

    traj_df.to_csv(traj_path, index=False)
    labels_df.to_csv(labels_path, index=False)

    print("Wrote:", traj_path, "rows:", len(traj_df))
    print("Wrote:", labels_path, "rows:", len(labels_df))
    print("Example label row:", labels_df.iloc[0].to_dict())


if __name__ == "__main__":
    main()
