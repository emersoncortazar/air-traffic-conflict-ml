from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd


DEFAULT_FEATURE_COLS = [
    "a_x_nm", "a_y_nm", "a_alt_ft",
    "b_x_nm", "b_y_nm", "b_alt_ft",
]


@dataclass(frozen=True)
class WindowDatasetConfig:
    traj_csv: Path = Path("data/trajectories.csv")
    windows_csv: Path = Path("data/windows.csv")

    # how long the input history is (seconds)
    history_s: float = 60.0

    # sampling period expected in trajectories (seconds); used to compute expected length
    dt_s: float = 1.0

    scenario_id_col: str = "scenario_id"
    time_col: str = "t_s"          # in trajectories
    t_end_col: str = "t_end_s"     # in windows
    label_col: str = "y_next_H"

    feature_cols: Tuple[str, ...] = tuple(DEFAULT_FEATURE_COLS)

    pad_value: float = 0.0


def load_window_sequences(cfg: WindowDatasetConfig):
    import numpy as np
    import pandas as pd

    traj = pd.read_csv(cfg.traj_csv)
    win = pd.read_csv(cfg.windows_csv)

    traj[cfg.scenario_id_col] = traj[cfg.scenario_id_col].astype(np.int64)
    win[cfg.scenario_id_col] = win[cfg.scenario_id_col].astype(np.int64)

    traj = traj.sort_values([cfg.scenario_id_col, cfg.time_col]).reset_index(drop=True)
    win = win.sort_values([cfg.scenario_id_col, cfg.t_end_col]).reset_index(drop=True)

    T = int(round(cfg.history_s / cfg.dt_s)) + 1
    F = len(cfg.feature_cols)

    # cache per-scenario arrays once
    cache = {}
    for sid, g in traj.groupby(cfg.scenario_id_col, sort=False):
        t = g[cfg.time_col].to_numpy(dtype=np.float32)
        x = g[list(cfg.feature_cols)].to_numpy(dtype=np.float32)
        cache[int(sid)] = (t, x)

    N = len(win)
    X = np.full((N, T, F), cfg.pad_value, dtype=np.float32)
    y = win[cfg.label_col].to_numpy(dtype=np.int64)
    meta = np.empty((N, 2), dtype=object)

    progress_every = 50_000

    for i, row in enumerate(win.itertuples(index=False), start=0):
        sid = int(getattr(row, cfg.scenario_id_col))
        t_end = float(getattr(row, cfg.t_end_col))
        meta[i, 0] = sid
        meta[i, 1] = t_end

        tup = cache.get(sid)
        if tup is None:
            continue

        t_arr, x_arr = tup
        t0 = t_end - float(cfg.history_s)

        left = np.searchsorted(t_arr, t0, side="right")
        right = np.searchsorted(t_arr, t_end, side="right")  # exclusive
        seg = x_arr[left:right]

        if seg.shape[0] == 0:
            continue

        if seg.shape[0] >= T:
            X[i] = seg[-T:]
        else:
            X[i, -seg.shape[0]:, :] = seg

        if (i + 1) % progress_every == 0:
            print(f"[{i+1}/{N}] windows processed...")

    return X, y, meta
