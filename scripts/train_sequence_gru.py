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
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


# ----------------------------
# Repro
# ----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # harmless if no CUDA


# ----------------------------
# Sequence building
# ----------------------------

def _add_kinematics_per_scenario(g: pd.DataFrame) -> pd.DataFrame:
    """
    Given one scenario's trajectory rows sorted by time, compute:
      dx_nm, dy_nm, dz_ft
      vx_rel_nmps, vy_rel_nmps, vz_rel_ftps (finite diff w/ actual dt between rows)
      closing_rate_nmps = -(dx*vx + dy*vy) / sqrt(dx^2 + dy^2)  (positive means closing)
    """
    g = g.sort_values("t_s").copy()

    # Relative position
    dx = (g["a_x_nm"] - g["b_x_nm"]).to_numpy(dtype=np.float32)
    dy = (g["a_y_nm"] - g["b_y_nm"]).to_numpy(dtype=np.float32)
    dz = (g["a_alt_ft"] - g["b_alt_ft"]).to_numpy(dtype=np.float32)

    t = g["t_s"].to_numpy(dtype=np.float32)
    dt = np.diff(t, prepend=t[0]).astype(np.float32)

    # Relative velocity (finite difference)
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

    # Closing rate (NM/s): positive means distance decreasing
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
    """
    Build fixed-length sequences per scenario using the FIRST k_seconds.

    Outputs:
      X: (N, T, D) float32  - padded sequence features
      M: (N, T) float32     - mask (1 for real timestep present, 0 for padded)
      y: (N,) int64         - conflict label
      scenario_ids: (N,) int64
    """
    traj_df = traj_df.copy()
    labels_df = labels_df.copy()

    traj_df["scenario_id"] = traj_df["scenario_id"].astype(int)
    labels_df["scenario_id"] = labels_df["scenario_id"].astype(int)
    labels_df["conflict"] = labels_df["conflict"].astype(int)

    # Add kinematics per scenario
    traj_df = traj_df.sort_values(["scenario_id", "t_s"]).copy()
    traj_df = traj_df.groupby("scenario_id", group_keys=False).apply(_add_kinematics_per_scenario)

    # Windowing
    T = int(math.floor(k_seconds / dt_s_nominal)) + 1  # expected length if fully observed
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

        # Keep only first K seconds (by observed timestamps)
        g = g[g["t_s"] <= k_seconds].copy()
        if len(g) == 0:
            continue

        feats = g[feature_cols].to_numpy(dtype=np.float32)

        L = min(len(feats), T)
        X[i, :L, :] = feats[:L, :]
        M[i, :L] = 1.0

        # Pad by repeating last observed value (keeps sequence stable under missingness)
        if L < T:
            X[i, L:, :] = feats[L - 1, :]

    return X, M, y, scenario_ids


def compute_norm_stats(X: np.ndarray, M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute mean/std over valid (masked) timesteps for each feature dim.
    """
    N, T, D = X.shape
    mask = M.reshape(N, T, 1)  # broadcast over D

    count = mask.sum(axis=(0, 1))  # (D,)
    count = np.maximum(count, 1.0)

    mean = (X * mask).sum(axis=(0, 1)) / count
    var = ((X - mean) ** 2 * mask).sum(axis=(0, 1)) / count
    std = np.sqrt(var)

    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_norm(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32)


# ----------------------------
# PyTorch Dataset/Model
# ----------------------------

class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, M: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)           # float32
        self.M = torch.from_numpy(M)           # float32
        self.y = torch.from_numpy(y).float()   # float32

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
        """
        x: (B,T,D)
        mask: (B,T) with 1 for valid timesteps
        Uses last valid hidden state
        """
        out, _ = self.gru(x)  # (B,T,H)

        lengths = mask.sum(dim=1).long()
        lengths = torch.clamp(lengths, min=1)

        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, out.size(-1))
        last = out.gather(dim=1, index=idx).squeeze(1)  # (B,H)

        logits = self.head(last).squeeze(-1)  # (B,)
        return logits


# ----------------------------
# Train/Eval
# ----------------------------

@torch.no_grad()
def evaluate(model, loader, device: torch.device):
    model.eval()
    ys = []
    ps = []
    for Xb, Mb, yb in loader:
        Xb = Xb.to(device)
        Mb = Mb.to(device)
        yb = yb.to(device)

        logits = model(Xb, Mb)
        prob = torch.sigmoid(logits)

        ys.append(yb.detach().cpu().numpy())
        ps.append(prob.detach().cpu().numpy())

    y_true = np.concatenate(ys).astype(int)
    p = np.concatenate(ps)

    y_hat = (p >= 0.5).astype(int)

    auc = roc_auc_score(y_true, p)
    ap = average_precision_score(y_true, p)
    acc = accuracy_score(y_true, y_hat)
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_hat, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_hat)

    return {
        "roc_auc": auc,
        "pr_auc": ap,
        "acc": acc,
        "prec": pr,
        "recall": rc,
        "f1": f1,
        "cm": cm,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", type=str, default="data/trajectories.csv")
    ap.add_argument("--labels", type=str, default="data/labels.csv")
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--k", type=float, default=20.0, help="use first K seconds as input window")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="models/sequence_gru.pt")
    args = ap.parse_args()

    set_seed(args.seed)

    traj_df = pd.read_csv(args.traj)
    labels_df = pd.read_csv(args.labels)

    X, M, y, scenario_ids = build_sequences(
        traj_df=traj_df,
        labels_df=labels_df,
        k_seconds=args.k,
        dt_s_nominal=args.dt,
    )

    idx = np.arange(len(y))
    idx_tr, idx_te, y_tr, y_te = train_test_split(
        idx, y, test_size=0.25, random_state=args.seed, stratify=y
    )

    X_tr, M_tr = X[idx_tr], M[idx_tr]
    X_te, M_te = X[idx_te], M[idx_te]

    mean, std = compute_norm_stats(X_tr, M_tr)
    X_tr = apply_norm(X_tr, mean, std)
    X_te = apply_norm(X_te, mean, std)

    train_ds = SeqDataset(X_tr, M_tr, y_tr)
    test_ds = SeqDataset(X_te, M_te, y_te)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GRUClassifier(
        input_dim=X.shape[-1],
        hidden_dim=args.hidden,
        num_layers=args.layers,
        dropout=args.dropout
    ).to(device)

    # pos_weight for BCEWithLogitsLoss
    pos = float((y_tr == 1).sum())
    neg = float((y_tr == 0).sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print("\n=== GRU Sequence Classifier (with kinematics) ===")
    print(f"Device: {device}")
    print(f"Train N: {len(train_ds)} | Test N: {len(test_ds)} | Pos rate: {y.mean():.3f}")
    print(f"Window K: {args.k}s | dt_nominal: {args.dt}s | T steps: {X.shape[1]} | D: {X.shape[2]}")
    print(f"Hidden: {args.hidden} | Layers: {args.layers} | Dropout: {args.dropout}")

    best_auc = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n_batches = 0

        for Xb, Mb, yb in train_loader:
            Xb = Xb.to(device)
            Mb = Mb.to(device)
            yb = yb.to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(Xb, Mb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

            running += float(loss.item())
            n_batches += 1

        train_loss = running / max(n_batches, 1)

        metrics = evaluate(model, test_loader, device)
        auc = metrics["roc_auc"]

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"ROC_AUC={metrics['roc_auc']:.4f} | PR_AUC={metrics['pr_auc']:.4f} | "
            f"Acc={metrics['acc']:.4f} | F1={metrics['f1']:.4f}"
        )

        if auc > best_auc:
            best_auc = auc
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            ckpt = {
                "model_state_dict": model.state_dict(),
                "mean": mean,
                "std": std,
                "args": vars(args),
            }
            torch.save(ckpt, out_path)

    final = evaluate(model, test_loader, device)
    print("\n=== Final Test Metrics ===")
    print(f"ROC AUC: {final['roc_auc']:.4f}")
    print(f"PR AUC:  {final['pr_auc']:.4f}")
    print(f"Acc:     {final['acc']:.4f}")
    print(f"Prec:    {final['prec']:.4f}")
    print(f"Recall:  {final['recall']:.4f}")
    print(f"F1:      {final['f1']:.4f}")
    print("Confusion matrix [[TN FP],[FN TP]]:")
    print(final["cm"])
    print("\nSaved best checkpoint to:", args.out)


if __name__ == "__main__":
    main()
