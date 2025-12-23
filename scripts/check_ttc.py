import pandas as pd
import numpy as np

SID = 1224

traj = pd.read_csv("data/trajectories.csv")
g = traj[traj["scenario_id"] == SID].sort_values("t_s").copy()

dx = g["a_x_nm"] - g["b_x_nm"]
dy = g["a_y_nm"] - g["b_y_nm"]
h = np.sqrt(dx * dx + dy * dy)

v = (g["a_alt_ft"] - g["b_alt_ft"]).abs()

los = (h < 5.0) & (v < 1000.0)

idx = np.where(los.to_numpy())[0]

first_los_t = float(g.iloc[idx[0]]["t_s"]) if len(idx) > 0 else None
min_h = float(h.min())
t_at_min_h = float(g.iloc[int(np.argmin(h))]["t_s"])

print("scenario_id:", SID)
print("rows:", len(g))
print("los_count:", int(los.sum()))
print("first_los_t:", first_los_t)
print("min_h_nm:", min_h)
print("t_at_min_h_s:", t_at_min_h)
