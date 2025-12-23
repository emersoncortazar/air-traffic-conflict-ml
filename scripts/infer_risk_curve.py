from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ----------------------------
# Model definition (must match training)
# ----------------------------

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
# Feature engineering (must match training)
# ----------------------------

def add_kinematics(g: pd.DataFrame) -> pd.DataFrame:
    """
    For a single scenario trajectory sorted by time, compute:
      dx_nm, dy_nm, dz_ft
      vx_rel_nmps, vy_rel_nmps, vz_rel_ftps
      closing_rate_nmps
    """
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


def make_window_tensor(
    g: pd.DataFrame,
    t_end: float,
    k_seconds: float,
    dt_nominal: float,
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build a (1, T, D) tensor for the window (t_end - k_seconds, t_end], padding if needed.
    Returns:
      X: float32 tensor (1,T,D)
      M: float32 tensor (1,T)
    """
    T = int(math.floor(k_seconds / dt_nominal)) + 1
    feature_cols = [
        "dx_nm", "dy_nm", "dz_ft",
        "vx_rel_nmps", "vy_rel_nmps", "vz_rel_ftps",
        "closing_rate_nmps",
    ]
    D = len(feature_cols)

    w = g[(g["t_s"] > (t_end - k_seconds)) & (g["t_s"] <= t_end)].copy()
    w = w.sort_values("t_s")

    X = np.zeros((T, D), dtype=np.float32)
    M = np.zeros((T,), dtype=np.float32)

    if len(w) == 0:
        # no observations: all mask 0, keep zeros
        pass
    else:
        feats = w[feature_cols].to_numpy(dtype=np.float32)
        L = min(len(feats), T)
        X[:L, :] = feats[:L, :]
        M[:L] = 1.0
        if L < T:
            X[L:, :] = feats[L - 1, :]

    # normalize
    X = (X - mean) / std

    Xt = torch.from_numpy(X).unsqueeze(0)  # (1,T,D)
    Mt = torch.from_numpy(M).unsqueeze(0)  # (1,T)
    return Xt, Mt


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", type=str, default="data/trajectories.csv")
    ap.add_argument("--labels", type=str, default="data/labels.csv")
    ap.add_argument("--ckpt", type=str, default="models/sequence_gru_calibrated.pt")
    ap.add_argument("--scenario", type=int, default=None, help="scenario_id; if omitted, auto-pick a conflict scenario")
    ap.add_argument("--dt", type=float, default=None, help="override dt_nominal (usually not needed)")
    ap.add_argument("--k", type=float, default=None, help="override K seconds (usually not needed)")
    ap.add_argument("--stride", type=int, default=1, help="evaluate every N rows (1 = every row)")
    ap.add_argument("--max_points", type=int, default=200, help="cap printed rows")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    saved_args = ckpt.get("args", {}) or {}

    mean = np.array(ckpt["mean"], dtype=np.float32)
    std = np.array(ckpt["std"], dtype=np.float32)

    T_cal = float(ckpt.get("temperature", 1.0))

    k_seconds = float(args.k) if args.k is not None else float(saved_args.get("k", 20.0))
    dt_nominal = float(args.dt) if args.dt is not None else float(saved_args.get("dt", 1.0))
    hidden = int(saved_args.get("hidden", 64))
    layers = int(saved_args.get("layers", 1))
    dropout = float(saved_args.get("dropout", 0.1))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    traj_df = pd.read_csv(args.traj)
    labels_df = pd.read_csv(args.labels)

    traj_df["scenario_id"] = traj_df["scenario_id"].astype(int)
    labels_df["scenario_id"] = labels_df["scenario_id"].astype(int)

    if args.scenario is None:
        # pick a conflict scenario if possible; else smallest id
        conflict_ids = labels_df[labels_df["conflict"] == True]["scenario_id"].tolist()
        if len(conflict_ids) == 0:
            sid = int(labels_df["scenario_id"].min())
            picked = "no conflicts found; picked min scenario_id"
        else:
            sid = int(conflict_ids[0])
            picked = "picked first conflict=True scenario"
    else:
        sid = int(args.scenario)
        picked = "user provided"

    g = traj_df[traj_df["scenario_id"] == sid].copy()
    if len(g) == 0:
        raise SystemExit(f"No rows found for scenario_id={sid} in {args.traj}")

    g = add_kinematics(g)
    g = g.sort_values("t_s")

    # Load model
    model = GRUClassifier(input_dim=7, hidden_dim=hidden, num_layers=layers, dropout=dropout).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    times = g["t_s"].to_numpy(dtype=np.float32)

    # Evaluate at each time step (with stride)
    out_rows = []
    for i in range(0, len(times), max(1, args.stride)):
        t_end = float(times[i])

        Xt, Mt = make_window_tensor(
            g=g,
            t_end=t_end,
            k_seconds=k_seconds,
            dt_nominal=dt_nominal,
            mean=mean,
            std=std,
        )
        Xt = Xt.to(device)
        Mt = Mt.to(device)

        with torch.no_grad():
            logit = model(Xt, Mt).detach().cpu().numpy()[0].astype(np.float64)

        # apply temperature scaling: sigmoid(logit / T)
        p = float(sigmoid(np.array([logit / T_cal]))[0])

        out_rows.append({
            "scenario_id": sid,
            "t_s": t_end,
            "p_conflict": p,
            "temperature": T_cal,
            "k_seconds": k_seconds,
            "dt_nominal": dt_nominal,
        })

    out_df = pd.DataFrame(out_rows)

    # Save
    out_path = Path("data") / f"risk_curve_scenario_{sid}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    # Print summary
    print("\n=== Risk Curve Inference ===")
    print(f"Scenario: {sid} ({picked})")
    # show label if available
    if sid in set(labels_df["scenario_id"].tolist()):
        row = labels_df[labels_df["scenario_id"] == sid].iloc[0].to_dict()
        print(f"Label conflict: {row.get('conflict')} | encounter_type: {row.get('encounter_type')} | ttc_s: {row.get('ttc_s')}")
    print(f"K window: {k_seconds}s | dt_nominal: {dt_nominal}s | temperature: {T_cal:.6f}")
    print(f"Evaluations: {len(out_df)} | Saved: {out_path}")

    # Print head/tail
    n = min(len(out_df), args.max_points)
    print("\nFirst rows:")
    print(out_df.head(min(10, n)).to_string(index=False))
    if len(out_df) > 10:
        print("\nLast rows:")
        print(out_df.tail(min(10, n)).to_string(index=False))


if __name__ == "__main__":
    main()
