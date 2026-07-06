#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Open-loop precision harness for the GR00T-WBC base via groot_mover.

Drives a sequence of small/medium forward / strafe / yaw targets through
GrootMover, then (if the adapter's --log-pose CSV is given) computes the ACTUAL
base displacement per move and prints a goal-vs-actual table — the precision
number that does not exist today.

Run alongside the adapter in SIM:
  adapter ... --interface sim --log-pose /tmp/groot_pose.csv
  python measure_precision.py --pose-csv /tmp/groot_pose.csv

Both this script and the adapter stamp time.time() on the same host, so the
move timestamps recorded here index directly into the pose CSV.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, asdict

import groot_mover as gm

# (kind, magnitude) — magnitude in cm for linear, degrees for yaw.
DEFAULT_SEQUENCE = [
    ("forward", 3), ("forward", 5), ("forward", 10), ("forward", 20), ("forward", 40),
    ("back", 10),
    ("left", 10), ("right", 10),
    ("yaw_left", 2), ("yaw_left", 5), ("yaw_left", 15), ("yaw_left", 45),
    ("yaw_right", 15),
]


@dataclass
class MoveRecord:
    kind: str
    target: float        # cm (linear) or deg (yaw)
    expected: float      # cm or deg, from the conversion (may overshoot if floored)
    floored: bool
    t_start: float
    t_end: float


def drive(mover: gm.GrootMover, sequence, settle_s: float) -> list:
    records = []
    for kind, mag in sequence:
        print(f"\n=== {kind} {mag} ===")
        t0 = time.time()
        if kind == "forward":
            plan = mover.move_forward_cm(mag)
        elif kind == "back":
            plan = mover.move_forward_cm(-mag)
        elif kind == "left":
            plan = mover.strafe_cm(mag)
        elif kind == "right":
            plan = mover.strafe_cm(-mag)
        elif kind == "yaw_left":
            plan = mover.rotate_deg(mag)
        elif kind == "yaw_right":
            plan = mover.rotate_deg(-mag)
        else:
            raise ValueError(kind)
        t1 = time.time()
        is_yaw = kind.startswith("yaw")
        exp = math.degrees(plan.expected) if is_yaw else plan.expected * 100.0
        records.append(MoveRecord(kind, float(mag), exp, plan.floored, t0, t1))
        time.sleep(settle_s)
    mover.stop()
    return records


# ----------------------------------------------------------------- analysis
def _yaw_of(qw, qx, qy, qz):
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def _load_pose_csv(path):
    rows = []
    with open(path) as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 8:
                continue
            t, x, y, z, qw, qx, qy, qz = (float(v) for v in parts[:8])
            rows.append((t, x, y, z, qw, qx, qy, qz))
    return rows


def _nearest(rows, t):
    return min(rows, key=lambda r: abs(r[0] - t)) if rows else None


def analyze(records, pose_csv: str) -> None:
    rows = _load_pose_csv(pose_csv)
    if not rows:
        print(f"\n[analyze] no pose rows in {pose_csv}; is the adapter running with "
              f"--log-pose in sim?")
        return

    print("\n" + "=" * 78)
    print(f"{'move':>12} {'target':>9} {'expected':>9} {'actual':>9} {'err':>8} {'flr':>4}")
    print("-" * 78)
    for r in records:
        a = _nearest(rows, r.t_start)
        b = _nearest(rows, r.t_end)
        if a is None or b is None:
            continue
        if r.kind.startswith("yaw"):
            actual = math.degrees(_yaw_of(b[4], b[5], b[6], b[7])
                                   - _yaw_of(a[4], a[5], a[6], a[7]))
            actual = (actual + 180.0) % 360.0 - 180.0   # wrap
            unit = "deg"
        else:
            dx, dy = b[1] - a[1], b[2] - a[2]
            planar = math.hypot(dx, dy) * 100.0          # cm
            sign = 1.0 if (r.expected >= 0) else -1.0
            actual = sign * planar
            unit = "cm"
        err = actual - r.expected
        print(f"{r.kind:>12} {r.target:>8.0f}{unit[:1]} {r.expected:>8.1f}{unit[:1]} "
              f"{actual:>8.1f}{unit[:1]} {err:>+7.1f}{unit[:1]} {'Y' if r.floored else '.':>4}")
    print("=" * 78)
    print("note: open-loop, no odometry feedback. 'actual' = sim ground-truth base "
          "displacement. err = actual - expected.")


def main() -> int:
    p = argparse.ArgumentParser(description="GR00T-WBC open-loop precision harness")
    p.add_argument("--cmd-file", default=gm.CMD_FILE)
    p.add_argument("--pose-csv", default=None,
                   help="adapter --log-pose CSV; if given, analyze after driving")
    p.add_argument("--warmup-s", type=float, default=5.0,
                   help="wait for the adapter to activate + stand before driving")
    p.add_argument("--settle-s", type=float, default=1.5,
                   help="pause after each move (re-perceive analog)")
    p.add_argument("--fwd-cruise", type=float, default=gm.FWD_CRUISE)
    p.add_argument("--yaw-cruise", type=float, default=gm.YAW_CRUISE)
    p.add_argument("--min-duration", type=float, default=gm.MIN_DURATION)
    p.add_argument("--save-moves", default=None, help="write the move log JSON here")
    p.add_argument("--analyze-only", action="store_true",
                   help="skip driving; analyze --save-moves + --pose-csv")
    args = p.parse_args()

    if args.analyze_only:
        with open(args.save_moves) as f:
            records = [MoveRecord(**m) for m in json.load(f)]
        analyze(records, args.pose_csv)
        return 0

    mover = gm.GrootMover(cmd_file=args.cmd_file, fwd_cruise=args.fwd_cruise,
                          yaw_cruise=args.yaw_cruise, min_duration=args.min_duration)
    print(f"[warmup] holding RL_FULL for {args.warmup_s:.1f}s while the base stands...")
    mover.initialize()
    time.sleep(max(0.0, args.warmup_s - 1.5))

    records = drive(mover, DEFAULT_SEQUENCE, args.settle_s)

    if args.save_moves:
        with open(args.save_moves, "w") as f:
            json.dump([asdict(r) for r in records], f, indent=2)
        print(f"[saved] move log -> {args.save_moves}")

    if args.pose_csv:
        time.sleep(0.5)
        analyze(records, args.pose_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
