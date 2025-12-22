"""
Randomized dataset generator for air-traffic conflict risk modeling.

Outputs (in ./data by default):
- trajectories.csv : per-timestep observed states (optionally noisy + with dropped rows)
- labels.csv       : per-scenario labels + scenario parameters

Run from repo root (Windows):
  set PYTHONPATH=%CD%\src
  python scripts\generate_random_dataset.py --n 5000 --noisy

Then:
  python scripts\build_features.py
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

import numpy as np

from atcml.sim.dynamics import aircraft_state
from atcml.sim.simulate import simulate_pair
from atcml.sim.conflict import conflict_within_horizon
from atcml.sim.encounter_generators import make_head_on, make_crossing
from atcml.sim.noise_models import apply_observation_noise


# -------------------------
# CSV utilities
# -------------------------
def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# -------------------------
# Encounter generators
# -------------------------
def make_overtake(
    initial_sep_nm: float,
    speed_a_kts: float,
    speed_b_kts: float,
    alt_ft: float,
) -> tuple[aircraft_state, aircraft_state]:
    """
    Overtake: both aircraft fly east (0 deg) along x-axis.
    A starts behind B by initial_sep_nm and is faster, potentially catching up.
    """
    a = aircraft_state(
        x_nm=0.0, y_nm=0.0, alt_ft=alt_ft,
        speed_kts=speed_a_kts, heading_deg=0.0, vz_fpm=0.0
    )
    b = aircraft_state(
        x_nm=initial_sep_nm, y_nm=0.0, alt_ft=alt_ft,
        speed_kts=speed_b_kts, heading_deg=0.0, vz_fpm=0.0
    )
    return a, b


def make_angle_encounter(
    angle_deg: float,
    speed_a_kts: float,
    speed_b_kts: float,
    alt_ft: float,
    t_intersect_s: float,
    time_offset_s: float,
    cross_track_offset_nm: float,
) -> tuple[aircraft_state, aircraft_state]:
    """
    General angle encounter (acute or obtuse): velocity vectors differ by angle_deg.
    A flies heading 0 deg (east).
    B flies heading angle_deg (0..180). For obtuse angles, B has negative x component.

    We "construct" an encounter around an intersection point at the origin:
    - Place A so it would reach the origin at t_intersect_s
    - Place B so it would reach the origin at (t_intersect_s + time_offset_s)
    - Add a perpendicular (cross-track) offset to B to create near-misses/nonconflicts

    This is a clean, controllable way to generate both conflicts and non-conflicts.
    """
    # Convert speeds to NM/s (because your sim uses NM and seconds)
    vA_nmps = speed_a_kts / 3600.0
    vB_nmps = speed_b_kts / 3600.0

    # Unit direction vectors
    a_heading_rad = math.radians(0.0)
    b_heading_rad = math.radians(angle_deg)

    uA = (math.cos(a_heading_rad), math.sin(a_heading_rad))  # (1,0)
    uB = (math.cos(b_heading_rad), math.sin(b_heading_rad))

    # Backtrack along velocity direction so that position at t=0 reaches origin at desired time
    dA_nm = vA_nmps * max(0.0, t_intersect_s)
    dB_nm = vB_nmps * max(0.0, t_intersect_s + time_offset_s)

    # Initial positions along their rays pointing toward origin
    a_x0 = -uA[0] * dA_nm
    a_y0 = -uA[1] * dA_nm

    b_x0 = -uB[0] * dB_nm
    b_y0 = -uB[1] * dB_nm

    # Apply cross-track offset to B (perpendicular to its direction)
    # Perp unit vector to uB is (-uB_y, uB_x)
    perpB = (-uB[1], uB[0])
    b_x0 += perpB[0] * cross_track_offset_nm
    b_y0 += perpB[1] * cross_track_offset_nm

    a = aircraft_state(
        x_nm=a_x0, y_nm=a_y0, alt_ft=alt_ft,
        speed_kts=speed_a_kts, heading_deg=0.0, vz_fpm=0.0
    )
    b = aircraft_state(
        x_nm=b_x0, y_nm=b_y0, alt_ft=alt_ft,
        speed_kts=speed_b_kts, heading_deg=angle_deg, vz_fpm=0.0
    )
    return a, b


# -------------------------
# Sampling logic
# -------------------------
def sample_angle_deg(rng: random.Random) -> float:
    """
    Sample a random angle in (0,180) but avoid being too close to 90 (crossing)
    and too close to 180 (head-on), because we keep those as separate types.

    You can relax these exclusions later if you want a continuous spectrum.
    """
    while True:
        angle = rng.uniform(10.0, 170.0)  # avoid near-0 and near-180 extremes
        if abs(angle - 90.0) < 5.0:
            continue  # keep crossing distinct
        if angle > 175.0:
            continue  # keep head-on distinct
        return angle


def sample_scenario(rng: random.Random) -> tuple[str, aircraft_state, aircraft_state, dict]:
    """
    Returns:
      encounter_type, stateA, stateB, params_dict
    """
    encounter_type = rng.choice(["head_on", "crossing", "overtake", "angle"])

    # Altitudes: keep simple (same altitude) for now; add vertical profiles later
    alt_ft = rng.choice([28000.0, 30000.0, 32000.0])

    if encounter_type == "head_on":
        sep_nm = rng.uniform(8.0, 50.0)
        speed_kts = rng.uniform(250.0, 480.0)
        a, b = make_head_on(separation_nm=sep_nm, speed_kts=speed_kts, alt_ft=alt_ft)
        params = {"sep_nm": sep_nm, "speed_kts": speed_kts, "alt_ft": alt_ft}
        return "head_on", a, b, params

    if encounter_type == "crossing":
        start_offset_nm = rng.uniform(6.0, 25.0)
        b_extra_offset_nm = rng.uniform(0.0, 25.0)
        speed_kts = rng.uniform(250.0, 480.0)
        a, b = make_crossing(
            start_offset_nm=start_offset_nm,
            speed_kts=speed_kts,
            alt_ft_a=alt_ft,
            alt_ft_b=alt_ft,
            b_extra_offset_nm=b_extra_offset_nm,
        )
        params = {
            "start_offset_nm": start_offset_nm,
            "b_extra_offset_nm": b_extra_offset_nm,
            "speed_kts": speed_kts,
            "alt_ft": alt_ft,
        }
        return "crossing", a, b, params

    if encounter_type == "overtake":
        initial_sep_nm = rng.uniform(2.0, 25.0)
        speed_b_kts = rng.uniform(220.0, 420.0)
        speed_a_kts = speed_b_kts + rng.uniform(20.0, 220.0)  # A usually faster
        a, b = make_overtake(
            initial_sep_nm=initial_sep_nm,
            speed_a_kts=speed_a_kts,
            speed_b_kts=speed_b_kts,
            alt_ft=alt_ft,
        )
        params = {
            "initial_sep_nm": initial_sep_nm,
            "speed_a_kts": speed_a_kts,
            "speed_b_kts": speed_b_kts,
            "alt_ft": alt_ft,
        }
        return "overtake", a, b, params

    # angle encounter (acute or obtuse)
    angle_deg = sample_angle_deg(rng)
    speed_a_kts = rng.uniform(220.0, 480.0)
    speed_b_kts = rng.uniform(220.0, 480.0)

    # We center around a notional intersection time, then randomize timing + miss distance.
    t_intersect_s = rng.uniform(40.0, 160.0)

    # If time_offset_s = 0 and cross_track_offset_nm = 0, they converge at the origin together.
    # We randomize these to create a mix of conflicts/nonconflicts.
    time_offset_s = rng.uniform(-40.0, 40.0)           # early/late arrival
    cross_track_offset_nm = rng.uniform(-10.0, 10.0)   # lateral miss distance

    a, b = make_angle_encounter(
        angle_deg=angle_deg,
        speed_a_kts=speed_a_kts,
        speed_b_kts=speed_b_kts,
        alt_ft=alt_ft,
        t_intersect_s=t_intersect_s,
        time_offset_s=time_offset_s,
        cross_track_offset_nm=cross_track_offset_nm,
    )

    params = {
        "angle_deg": angle_deg,
        "speed_a_kts": speed_a_kts,
        "speed_b_kts": speed_b_kts,
        "alt_ft": alt_ft,
        "t_intersect_s": t_intersect_s,
        "time_offset_s": time_offset_s,
        "cross_track_offset_nm": cross_track_offset_nm,
    }
    return "angle", a, b, params


# -------------------------
# Main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000, help="number of scenarios")
    ap.add_argument("--dt", type=float, default=1.0, help="timestep seconds")
    ap.add_argument("--horizon", type=float, default=180.0, help="horizon seconds")
    ap.add_argument("--seed", type=int, default=0, help="random seed")

    ap.add_argument("--noisy", action="store_true", help="apply observation noise + dropping")
    ap.add_argument("--sigma_xy_nm", type=float, default=0.05, help="position noise std (NM)")
    ap.add_argument("--sigma_alt_ft", type=float, default=50.0, help="alt noise std (ft)")
    ap.add_argument("--drop_prob", type=float, default=0.10, help="drop probability per timestep")

    ap.add_argument("--out_dir", type=str, default="data", help="output directory")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    traj_rows: list[dict] = []
    label_rows: list[dict] = []

    for scenario_id in range(args.n):
        encounter_type, a0, b0, params = sample_scenario(rng)

        # Label from clean dynamics ("teacher"): conflict + time-to-first-conflict
        conflict, ttc = conflict_within_horizon(a0, b0, dt_s=args.dt, horizon_s=args.horizon)

        # Generate clean trajectory rows
        rows = simulate_pair(a0, b0, dt_s=args.dt, horizon_s=args.horizon)

        # Optionally apply observation noise + missingness ("student" inputs)
        if args.noisy:
            rows = apply_observation_noise(
                rows,
                sigma_xy_nm=args.sigma_xy_nm,
                sigma_alt_ft=args.sigma_alt_ft,
                drop_prob=args.drop_prob,
                seed=args.seed + scenario_id,  # vary per scenario but reproducible
            )

        # Attach scenario metadata to each trajectory row
        for r in rows:
            r["scenario_id"] = int(scenario_id)
            r["encounter_type"] = encounter_type
        traj_rows.extend(rows)

        # Label row (store parameters for slicing/debugging later)
        label = {
            "scenario_id": int(scenario_id),
            "encounter_type": encounter_type,
            "conflict": bool(conflict),
            "ttc_s": float(ttc),
            "dt_s": float(args.dt),
            "horizon_s": float(args.horizon),
        }
        # Add scenario parameters (varies by encounter)
        label.update({k: float(v) for k, v in params.items()})
        label_rows.append(label)

        if (scenario_id + 1) % max(1, args.n // 10) == 0:
            print(f"[{scenario_id+1}/{args.n}] trajectories so far: {len(traj_rows)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    traj_path = out_dir / "trajectories.csv"
    labels_path = out_dir / "labels.csv"

    # Keep trajectory schema stable
    traj_fields = [
        "scenario_id", "encounter_type", "t_s",
        "a_x_nm", "a_y_nm", "a_alt_ft",
        "b_x_nm", "b_y_nm", "b_alt_ft",
    ]
    write_csv(traj_path, traj_rows, traj_fields)

    # Labels: union of keys
    label_fields = sorted({k for row in label_rows for k in row.keys()})
    write_csv(labels_path, label_rows, label_fields)

    print("\nWrote:", traj_path, "rows:", len(traj_rows))
    print("Wrote:", labels_path, "rows:", len(label_rows))
    print("Example label row:", label_rows[0])


if __name__ == "__main__":
    main()
