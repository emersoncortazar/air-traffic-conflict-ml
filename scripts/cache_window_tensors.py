# scripts/cache_window_tensors.py
from __future__ import annotations

from pathlib import Path

import numpy as np

from atcml.ml.window_dataset import WindowDatasetConfig, load_window_sequences


def main():
    out_dir = Path("data/cache")
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = WindowDatasetConfig(
        traj_csv=Path("data/trajectories.csv"),
        windows_csv=Path("data/windows.csv"),
        history_s=60.0,
        dt_s=1.0,
    )

    X, y, meta = load_window_sequences(cfg)

    # Save in a compact way: meta as .npz of two arrays (sid, t_end)
    sid = meta[:, 0].astype(np.int64)
    t_end = meta[:, 1].astype(np.float32)

    np.save(out_dir / "X_H60_hist60_dt1.npy", X)
    np.save(out_dir / "y_H60_hist60_dt1.npy", y)
    np.savez(out_dir / "meta_H60_hist60_dt1.npz", scenario_id=sid, t_end_s=t_end)

    print("Saved:")
    print(" ", out_dir / "X_H60_hist60_dt1.npy")
    print(" ", out_dir / "y_H60_hist60_dt1.npy")
    print(" ", out_dir / "meta_H60_hist60_dt1.npz")


if __name__ == "__main__":
    main()
