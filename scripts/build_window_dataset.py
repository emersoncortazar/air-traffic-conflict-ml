from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


H_NM = 5.0
V_FT = 1000.0


def compute_los_flags(g: pd.DataFrame) -> pd.DataFrame:
    """Add h_sep_nm, v_sep_ft, los flag for a single scenario."""
    g = g.sort_values("t_s").copy()

    dx = g["a_x_nm"] - g["b_x_nm"]
    dy = g["a_y_nm"] - g["b_y_nm"]
    h = np.sqrt(dx * dx + dy * dy)

    v = (g["a_alt_ft"] - g["b_alt_ft"]).abs()

    g["h_sep_nm"] = h.astype(float)
    g["v_sep_ft"] = v.astype(float)
    g["los"] = ((g["h_sep_nm"] < H_NM) & (g["v_sep_ft"] < V_FT)).astype(int)
    return g


def first_los_time(g: pd.DataFrame) -> float | None:
    """Return first time where los==1, else None."""
    idx = np.where(g["los"].to_numpy() == 1)[0]
    if len(idx) == 0:
        return None
    return float(g.iloc[int(idx[0])]["t_s"])


def build_windows_for_scenario(
    g: pd.DataFrame,
    scenario_id: int,
    horizon_s: float,
    H_s: float,
) -> list[dict]:
    """
    For each timestep t_end in [0, horizon_s], define label:
      y_next_H = 1 if first_los_time in (t_end, t_end + H_s], else 0
    If no conflict, labels are always 0.
    """
    g = g.sort_values("t_s")
    times = g["t_s"].to_numpy(dtype=float)

    t_los = first_los_time(g)  # None if no conflict

    rows = []
    for t_end in times:
        if t_end > horizon_s:
            break

        if t_los is None:
            y = 0
            ttc_next = -1.0
        else:
            # "conflict within next H seconds"
            y = 1 if (t_los > t_end) and (t_los <= t_end + H_s) else 0
            # optional "time to conflict from t_end", only meaningful if y==1
            ttc_next = float(t_los - t_end) if y == 1 else -1.0

        rows.append(
            {
                "scenario_id": int(scenario_id),
                "t_end_s": float(t_end),
                "y_next_H": int(y),
                "ttc_next_s": float(ttc_next),
            }
        )

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", type=str, default="data/trajectories.csv")
    ap.add_argument("--labels", type=str, default="data/labels.csv")
    ap.add_argument("--out", type=str, default="data/windows.csv")
    ap.add_argument("--H", type=float, default=60.0, help="future horizon H seconds for y_next_H")
    ap.add_argument("--max_scenarios", type=int, default=None)
    args = ap.parse_args()

    traj = pd.read_csv(args.traj)
    labels = pd.read_csv(args.labels)

    traj["scenario_id"] = traj["scenario_id"].astype(int)
    labels["scenario_id"] = labels["scenario_id"].astype(int)

    # Join meta so we know horizon per scenario (and can sanity check)
    meta = labels[["scenario_id", "conflict", "ttc_s", "horizon_s"]].copy()

    rows_out: list[dict] = []

    scenario_ids = meta["scenario_id"].tolist()
    if args.max_scenarios is not None:
        scenario_ids = scenario_ids[: int(args.max_scenarios)]

    for j, sid in enumerate(scenario_ids, start=1):
        g = traj[traj["scenario_id"] == sid].copy()
        if len(g) == 0:
            continue

        g = compute_los_flags(g)

        horizon_s = float(meta.loc[meta["scenario_id"] == sid, "horizon_s"].iloc[0])

        rows_out.extend(
            build_windows_for_scenario(
                g=g,
                scenario_id=int(sid),
                horizon_s=horizon_s,
                H_s=float(args.H),
            )
        )

        if j % 250 == 0:
            print(f"[{j}/{len(scenario_ids)}] windows so far: {len(rows_out)}")

    out_df = pd.DataFrame(rows_out)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    pos_rate = float(out_df["y_next_H"].mean()) if len(out_df) else 0.0
    print("\n=== Built window dataset ===")
    print("Saved:", out_path, "| rows:", len(out_df), "| pos_rate:", f"{pos_rate:.4f}")
    print(out_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
