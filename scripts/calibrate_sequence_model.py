from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss


# ----------------------------
# Repro
# ----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------
# Sequence building (must match training)
# ----------------------------

def _add_kinematics_per_scenario(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("t_s").copy()

    dx = (g["a_x_nm"] - g["b_x_nm"]).to_numpy(dtype=np.float32)
    dy = (g["a_y_nm"] - g["b_y_nm"]).to_numpy(dtype=np.float32)
    dz = (g["a_alt_ft"] - g["b_alt_ft"]).to_numpy(dtype=np.float32)

    t = g["t_s"].to_numpy(dtype=np.float32)
    dt = np.diff(t, prepend=t[0]).astype(np.float32)

    vx = np.zeros_like(dx, dtype=np.float32)
    vy = np.zeros_like(dy, dtype=np.float32)
    vz = np.zeros_like(dz, dtype=np.float32)

    for i in range(1, len(g)):
        if not np.isfinite(dt[i]) or dt[i] <= 1e-6:
            vx[i] = vx[i - 1]
            vy[i] = vy[i - 1]
            vz[i] = vz[i - 1]
        else:
            vx[i] = (dx[i] - dx[i - 1]) / dt[i]
            vy[i] = (dy[i] - dy[i - 1]) / dt[i]
            vz[i] = (dz[i] - dz[i - 1]) / dt[i]

    r = np.sqrt(dx * dx + dy * dy).astype(np.float32)
    r = np.where(r < 1e-6, 1e-6, r).astype(np.float32)
    closing = (-(dx * vx + dy * vy) / r).astype(np.float32)

    g["dx_nm"] = dx
    g["dy_nm"] = dy
    g["dz_ft"] = dz
    g["vx_rel_nmps"] = vx
    g["vy_rel_nmps"] = vy
    g["vz_rel_ftps"] = vz
    g["closing_rate_nmps"] = closing

    return g


def build_sequences(
    traj_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    k_seconds: float,
    dt_s_nominal: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    traj_df = traj_df.copy()
    labels_df = labels_df.copy()

    traj_df["scenario_id"] = traj_df["scenario_id"].astype(int)
    labels_df["scenario_id"] = labels_df["scenario_id"].astype(int)
    labels_df["conflict"] = labels_df["conflict"].astype(int)

    traj_df = traj_df.sort_values(["scenario_id", "t_s"]).copy()
    traj_df = traj_df.groupby("scenario_id", group_keys=False, sort=False).apply(_add_kinematics_per_scenario)

    T = int(math.floor(k_seconds / dt_s_nominal)) + 1
    feature_cols = [
        "dx_nm", "dy_nm", "dz_ft",
        "vx_rel_nmps", "vy_rel_nmps", "vz_rel_ftps",
        "closing_rate_nmps",
    ]
    D = len(feature_cols)

    scenario_ids = labels_df["scenario_id"].to_numpy(dtype=np.int64)
    y = labels_df["conflict"].to_numpy(dtype=np.int64)

    X = np.zeros((len(scenario_ids), T, D), dtype=np.float32)
    M = np.zeros((len(scenario_ids), T), dtype=np.float32)

    grouped = traj_df.groupby("scenario_id", sort=False)

    for i, sid in enumerate(scenario_ids):
        if sid not in grouped.groups:
            continue

        g = grouped.get_group(sid).sort_values("t_s")
        g = g[g["t_s"] <= k_seconds].copy()
        if len(g) == 0:
            continue

        feats = g[feature_cols].to_numpy(dtype=np.float32)

        L = min(len(feats), T)
        X[i, :L, :] = feats[:L, :]
        M[i, :L] = 1.0

        if L < T:
            X[i, L:, :] = feats[L - 1, :]

    return X, M, y, scenario_ids


def apply_norm(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32)


# ----------------------------
# Dataset / Model (must match training)
# ----------------------------

class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, M: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.M = torch.from_numpy(M)
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.M[idx], self.y[idx]


class GRUClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)  # (B,T,H)

        lengths = mask.sum(dim=1).long()
        lengths = torch.clamp(lengths, min=1)

        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, out.size(-1))
        last = out.gather(dim=1, index=idx).squeeze(1)  # (B,H)

        logits = self.head(last).squeeze(-1)  # (B,)
        return logits


# ----------------------------
# Calibration utilities
# ----------------------------

@torch.no_grad()
def collect_logits_and_labels(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits = []
    all_y = []
    for Xb, Mb, yb in loader:
        Xb = Xb.to(device)
        Mb = Mb.to(device)
        logits = model(Xb, Mb)
        all_logits.append(logits.detach().cpu().numpy())
        all_y.append(yb.detach().cpu().numpy())
    logits = np.concatenate(all_logits).astype(np.float64)
    y = np.concatenate(all_y).astype(np.int64)
    return logits, y


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def ece_score(y_true: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """
    Expected Calibration Error with equal-width bins over [0,1].
    """
    y_true = y_true.astype(np.int64)
    p = np.clip(p, 0.0, 1.0)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(p)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)

        if not np.any(mask):
            continue

        bin_p = p[mask]
        bin_y = y_true[mask]

        acc = bin_y.mean()
        conf = bin_p.mean()
        ece += (len(bin_p) / n) * abs(acc - conf)

    return float(ece)


def brier_score(y_true: np.ndarray, p: np.ndarray) -> float:
    y_true = y_true.astype(np.float64)
    p = np.clip(p, 0.0, 1.0).astype(np.float64)
    return float(np.mean((p - y_true) ** 2))


def nll_from_probs(y_true: np.ndarray, p: np.ndarray) -> float:
    """
    Negative log-likelihood (log loss).
    """
    y_true = y_true.astype(np.int64)
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return float(log_loss(y_true, p))


def fit_temperature(logits_val: np.ndarray, y_val: np.ndarray, device: torch.device) -> float:
    """
    Fit temperature T > 0 by minimizing NLL on validation set:
      p = sigmoid(logits / T)
    We optimize log_T (unconstrained).
    """
    logits_t = torch.tensor(logits_val, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_val, dtype=torch.float32, device=device)

    log_T = torch.zeros((), dtype=torch.float32, device=device, requires_grad=True)
    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.LBFGS([log_T], lr=0.1, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad(set_to_none=True)
        T = torch.exp(log_T) + 1e-6
        scaled_logits = logits_t / T
        loss = loss_fn(scaled_logits, y_t)
        loss.backward()
        return loss

    optimizer.step(closure)

    T_final = float((torch.exp(log_T) + 1e-6).detach().cpu().item())
    return T_final


def report_calibration(name: str, y: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> None:
    ece = ece_score(y, probs, n_bins=n_bins)
    brier = brier_score(y, probs)
    nll = nll_from_probs(y, probs)
    print(f"{name}:  ECE={ece:.4f} | Brier={brier:.4f} | NLL={nll:.4f}")


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="models/sequence_gru.pt")
    ap.add_argument("--out", type=str, default="models/sequence_gru_calibrated.pt")
    ap.add_argument("--traj", type=str, default="data/trajectories.csv")
    ap.add_argument("--labels", type=str, default="data/labels.csv")
    ap.add_argument("--bins", type=int, default=15)
    ap.add_argument("--val_frac", type=float, default=0.2, help="fraction of TRAIN set used for calibration fitting")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    saved_args = ckpt.get("args", {}) or {}
    mean = ckpt["mean"]
    std = ckpt["std"]

    seed = int(saved_args.get("seed", 0))
    k_seconds = float(saved_args.get("k", 20.0))
    dt_nominal = float(saved_args.get("dt", 1.0))
    hidden = int(saved_args.get("hidden", 64))
    layers = int(saved_args.get("layers", 1))
    dropout = float(saved_args.get("dropout", 0.1))

    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Rebuild dataset exactly like training
    traj_df = pd.read_csv(args.traj)
    labels_df = pd.read_csv(args.labels)

    X, M, y, _ = build_sequences(traj_df, labels_df, k_seconds=k_seconds, dt_s_nominal=dt_nominal)
    X = apply_norm(X, mean, std)

    idx = np.arange(len(y))

    # SAME split as training: 75/25 train/test
    idx_tr, idx_te, y_tr, y_te = train_test_split(
        idx, y, test_size=0.25, random_state=seed, stratify=y
    )

    # Now split TRAIN into train_sub / val for temperature fitting
    idx_tr_sub, idx_val, y_tr_sub, y_val = train_test_split(
        idx_tr, y_tr, test_size=args.val_frac, random_state=seed, stratify=y_tr
    )

    # Build loaders
    def make_loader(indices: np.ndarray, shuffle: bool) -> DataLoader:
        ds = SeqDataset(X[indices], M[indices], y[indices])
        return DataLoader(ds, batch_size=args.batch, shuffle=shuffle, drop_last=False)

    loader_val = make_loader(idx_val, shuffle=False)
    loader_test = make_loader(idx_te, shuffle=False)

    # Recreate model + load weights
    model = GRUClassifier(input_dim=X.shape[-1], hidden_dim=hidden, num_layers=layers, dropout=dropout).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    # Collect logits
    logits_val, y_val2 = collect_logits_and_labels(model, loader_val, device)
    logits_test, y_test2 = collect_logits_and_labels(model, loader_test, device)

    assert np.array_equal(y_val2, y[idx_val].astype(int)), "Validation labels mismatch"
    assert np.array_equal(y_test2, y[idx_te].astype(int)), "Test labels mismatch"

    # Before calibration
    probs_val_raw = sigmoid(logits_val)
    probs_test_raw = sigmoid(logits_test)

    print("\n=== Temperature Scaling Calibration ===")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Device: {device}")
    print(f"Seed: {seed} | K={k_seconds}s | dt={dt_nominal}s | D={X.shape[-1]}")
    print(f"Val size: {len(idx_val)} | Test size: {len(idx_te)} | bins={args.bins}")

    report_calibration("VAL  (raw)", y_val2, probs_val_raw, n_bins=args.bins)
    report_calibration("TEST (raw)", y_test2, probs_test_raw, n_bins=args.bins)

    # Fit T on VAL
    T = fit_temperature(logits_val, y_val2, device=device)

    # After calibration
    probs_val_cal = sigmoid(logits_val / T)
    probs_test_cal = sigmoid(logits_test / T)

    print(f"\nFitted temperature: T = {T:.6f}  (T>1 softens, T<1 sharpens)")
    report_calibration("VAL  (cal)", y_val2, probs_val_cal, n_bins=args.bins)
    report_calibration("TEST (cal)", y_test2, probs_test_cal, n_bins=args.bins)

    # Save calibrated checkpoint
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    calibrated = dict(ckpt)
    calibrated["temperature"] = float(T)
    calibrated["calibration"] = {
        "bins": int(args.bins),
        "val_frac": float(args.val_frac),
        "val_raw": {
            "ece": ece_score(y_val2, probs_val_raw, n_bins=args.bins),
            "brier": brier_score(y_val2, probs_val_raw),
            "nll": nll_from_probs(y_val2, probs_val_raw),
        },
        "val_cal": {
            "ece": ece_score(y_val2, probs_val_cal, n_bins=args.bins),
            "brier": brier_score(y_val2, probs_val_cal),
            "nll": nll_from_probs(y_val2, probs_val_cal),
        },
        "test_raw": {
            "ece": ece_score(y_test2, probs_test_raw, n_bins=args.bins),
            "brier": brier_score(y_test2, probs_test_raw),
            "nll": nll_from_probs(y_test2, probs_test_raw),
        },
        "test_cal": {
            "ece": ece_score(y_test2, probs_test_cal, n_bins=args.bins),
            "brier": brier_score(y_test2, probs_test_cal),
            "nll": nll_from_probs(y_test2, probs_test_cal),
        },
    }

    torch.save(calibrated, out_path)
    print(f"\nSaved calibrated checkpoint to: {out_path}")


if __name__ == "__main__":
    main()
