# FILE: src/atcml/ml/dataset.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass
class WindowedDataset:
    """
    Loads windowed trajectory data from data/windows.csv and
    converts it into model-ready tensors.
    """

    feature_cols: List[str]
    label_col: str = "y_next_H"
    time_col: str = "t"
    window_id_col: str = "window_id"

    def load(self, csv_path: str | None = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns
        -------
        X : np.ndarray
            Shape (N_windows, T, F)
        y : np.ndarray
            Shape (N_windows,)
        """
        if csv_path is None:
            raise ValueError("csv_path must be provided (e.g., data/windows.csv)")

        df = pd.read_csv(csv_path)

        # Ensure deterministic ordering
        df = df.sort_values([self.window_id_col, self.time_col])

        X_windows = []
        y_windows = []

        for wid, g in df.groupby(self.window_id_col):
            X_windows.append(g[self.feature_cols].to_numpy())
            y_windows.append(int(g[self.label_col].iloc[0]))

        X = np.stack(X_windows)
        y = np.asarray(y_windows, dtype=np.int64)

        return X, y
