# scripts/make_splits.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from atcml.ml.split import split_by_scenario_id


def main():
    out_dir = Path("data/splits")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use windows.csv (scenario_id + label col exists)
    win = pd.read_csv("data/windows.csv")

    splits = split_by_scenario_id(
        win,
        scenario_id_col="scenario_id",
        test_frac=0.20,
        val_frac=0.10,
        seed=42,
        stratify_cols=("y_next_H",),   # scenario-level stratification proxy
    )

    np.savetxt(out_dir / "train_sids.txt", splits.train_sids, fmt="%d")
    np.savetxt(out_dir / "val_sids.txt", splits.val_sids, fmt="%d")
    np.savetxt(out_dir / "test_sids.txt", splits.test_sids, fmt="%d")

    print("Wrote:")
    print(" ", out_dir / "train_sids.txt", f"({len(splits.train_sids)})")
    print(" ", out_dir / "val_sids.txt", f"({len(splits.val_sids)})")
    print(" ", out_dir / "test_sids.txt", f"({len(splits.test_sids)})")


if __name__ == "__main__":
    main()
