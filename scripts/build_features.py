from pathlib import Path
import pandas as pd
import numpy as np

def main():
    traj_path = Path("data") / "trajectories.csv"
    labels_path = Path("data") / "labels.csv"
    out_path = Path("data") / "features.csv"

    traj = pd.read_csv(traj_path)
    labels = pd.read_csv(labels_path)

    # compute separations at every timestep
    dx = traj["a_x_nm"] - traj["b_x_nm"]
    dy = traj["a_y_nm"] - traj["b_y_nm"]
    traj["h_sep_nm"] = np.sqrt(dx * dx + dy * dy)
    traj["v_sep_ft"] = (traj["a_alt_ft"] - traj["b_alt_ft"]).abs()

    # group by scenario_id and reduce the time-series into features
    feature_rows = []
    for sid, g in traj.groupby("scenario_id", sort=True):
        # index of the timestep where horizontal separation is minimum
        idx_min_h = g["h_sep_nm"].idxmin()

        min_h = float(g.loc[idx_min_h, "h_sep_nm"])
        t_at_min_h = float(g.loc[idx_min_h, "t_s"])

        min_v = float(g["v_sep_ft"].min())

        # initial separations (at t=0) for comparison
        g0 = g[g["t_s"] == 0.0].iloc[0]
        h0 = float(g0["h_sep_nm"])
        v0 = float(g0["v_sep_ft"])

        feature_rows.append({
            "scenario_id": int(sid),
            "h0_nm": h0,
            "v0_ft": v0,
            "min_h_nm": min_h,
            "t_at_min_h_s": t_at_min_h,
            "min_v_ft": min_v,
        })

    features = pd.DataFrame(feature_rows)

    # join labels so the features file also contains the target
    dataset = features.merge(labels[["scenario_id", "conflict", "ttc_s", "encounter_type"]],
                             on="scenario_id", how="left")

    dataset.to_csv(out_path, index=False)

    print("Wrote:", out_path, "rows:", len(dataset))
    print(dataset.head(10))

if __name__ == "__main__":
    main()
