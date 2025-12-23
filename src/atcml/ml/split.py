# FILE: src/atcml/ml/split.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScenarioSplits:
    train_sids: np.ndarray
    val_sids: np.ndarray
    test_sids: np.ndarray


def _stable_shuffle(arr: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    out = arr.copy()
    rng.shuffle(out)
    return out


def _make_strata_key(df: pd.DataFrame, stratify_cols: Sequence[str]) -> Optional[np.ndarray]:
    if not stratify_cols:
        return None
    missing = [c for c in stratify_cols if c not in df.columns]
    if missing:
        return None
    key = df[stratify_cols[0]].astype(str)
    for c in stratify_cols[1:]:
        key = key + "|" + df[c].astype(str)
    return key.to_numpy()


def split_by_scenario_id(
    df: pd.DataFrame,
    *,
    scenario_id_col: str = "scenario_id",
    test_frac: float = 0.20,
    val_frac: float = 0.10,
    seed: int = 42,
    stratify_cols: Sequence[str] = (),
    min_per_stratum: int = 2,
) -> ScenarioSplits:
    """
    Leakage-safe split: scenario_id is the unit of split.
    All rows for a scenario_id fall in exactly one of train/val/test.

    If stratify_cols are provided and sufficiently populated, we do per-stratum splitting.
    Otherwise we fall back to simple shuffled slicing.
    """
    if scenario_id_col not in df.columns:
        raise ValueError(f"Missing {scenario_id_col} in df")

    if test_frac < 0 or val_frac < 0 or (test_frac + val_frac) >= 1.0:
        raise ValueError("Require test_frac>=0, val_frac>=0, and test_frac+val_frac < 1.")

    # scenario-level frame (1 row per scenario)
    base_cols = [scenario_id_col] + [c for c in stratify_cols if c in df.columns]
    by_sid = df[base_cols].drop_duplicates(subset=[scenario_id_col]).copy()
    by_sid[scenario_id_col] = by_sid[scenario_id_col].astype(np.int64)

    sids = by_sid[scenario_id_col].to_numpy(dtype=np.int64)
    sids = np.unique(sids)

    # Try stratified if possible
    strat_key = _make_strata_key(by_sid, stratify_cols)
    if strat_key is not None:
        vc = pd.Series(strat_key).value_counts()
        if (vc < int(min_per_stratum)).any():
            strat_key = None  # too sparse => fallback

    sids_shuf = _stable_shuffle(sids, seed)

    n = len(sids_shuf)
    n_test = int(round(n * float(test_frac)))
    n_val = int(round(n * float(val_frac)))
    n_test = min(max(n_test, 0), n)
    n_val = min(max(n_val, 0), n - n_test)

    if strat_key is None:
        test = sids_shuf[:n_test]
        val = sids_shuf[n_test : n_test + n_val]
        train = sids_shuf[n_test + n_val :]
        return ScenarioSplits(train_sids=train, val_sids=val, test_sids=test)

    # Stratified: split within each stratum
    sid_to_key = dict(zip(by_sid[scenario_id_col].to_numpy(dtype=np.int64), strat_key))
    keys = np.array([sid_to_key[int(s)] for s in sids_shuf], dtype=object)

    train, val, test = [], [], []
    for k in np.unique(keys):
        idx = np.where(keys == k)[0]
        sid_k = sids_shuf[idx]
        sid_k = _stable_shuffle(sid_k, seed ^ (hash(k) & 0xFFFFFFFF))

        nk = len(sid_k)
        tk = int(round(nk * float(test_frac)))
        vk = int(round(nk * float(val_frac)))
        tk = min(max(tk, 0), nk)
        vk = min(max(vk, 0), nk - tk)

        test.extend(sid_k[:tk])
        val.extend(sid_k[tk : tk + vk])
        train.extend(sid_k[tk + vk :])

    return ScenarioSplits(
        train_sids=_stable_shuffle(np.unique(np.array(train, dtype=np.int64)), seed + 1),
        val_sids=_stable_shuffle(np.unique(np.array(val, dtype=np.int64)), seed + 2),
        test_sids=_stable_shuffle(np.unique(np.array(test, dtype=np.int64)), seed + 3),
    )


def filter_df_to_sids(
    df: pd.DataFrame,
    sids: Iterable[int],
    *,
    scenario_id_col: str = "scenario_id",
) -> pd.DataFrame:
    sid_set = set(int(s) for s in sids)
    return df[df[scenario_id_col].astype(int).isin(sid_set)].copy()
