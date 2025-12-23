# FILE: src/atcml/ml/paths.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataPaths:
    repo_root: Path
    data_dir: Path
    trajectories_csv: Path
    labels_csv: Path
    windows_csv: Path
    splits_dir: Path
    models_dir: Path
    reports_dir: Path

    @staticmethod
    def from_repo_root(repo_root: Path) -> "DataPaths":
        repo_root = repo_root.resolve()
        data = repo_root / "data"
        return DataPaths(
            repo_root=repo_root,
            data_dir=data,
            trajectories_csv=data / "trajectories.csv",
            labels_csv=data / "labels.csv",
            windows_csv=data / "windows.csv",
            splits_dir=data / "splits",
            models_dir=repo_root / "models",
            reports_dir=repo_root / "reports",
        )


def repo_root_from_file(file: str | Path) -> Path:
    """
    Works when this file is at: <repo>/src/atcml/ml/paths.py
    and is imported by scripts under <repo>/scripts or <repo>/src/...
    """
    p = Path(file).resolve()
    # paths.py -> ml -> atcml -> src -> repo_root
    return p.parents[3]
