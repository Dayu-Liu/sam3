#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Isolated small-move test for the GR00T-WBC forward-step fix (robot-side).

Drives the SAME RemoteMover path box_demo uses (robot -> HTTP -> 5080 GR00T base),
replaying the kind of small cm/deg adjustments box_demo emits while approaching
the box — so you can watch each command actually step FORWARD (never backward)
before running the full grasp pipeline. No camera / SAM3 / box needed.

Prereq: the 5080 base (merger + adapter) + HTTP bridge on :5001 are up (see
test_box_demo2.md §3 terminals ② + ③). Robot reachable to the 5080 over DDS.

Run on the robot:
  conda activate robojudo
  cd /home/unitree/zihou/box_demo_2
  python test_small_moves.py --ipc-url http://192.168.123.222:5001

A/B the fix with the same knobs as box_demo, e.g.:
  python test_small_moves.py --warmup-time 0     # old-style (expect some backward)
  python test_small_moves.py --warmup-time 0.6   # new default (expect always forward)
"""
import argparse
import time

from remote_mover import RemoteMover

# the small adjustments box_demo emits during approach: (kind, value)
# value: cm for forward/strafe, deg for rotate. Includes a small backward.
SEQ = [
    ("forward", 4), ("forward", 6), ("forward", 8), ("forward", 10),
    ("strafe", 5), ("strafe", -5),
    ("rotate", 10), ("rotate", -10),
    ("forward", -6),   # intentional backward should still work
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ipc-url", default="http://192.168.123.222:5001",
                    help="5080 HTTP IPC bridge (agile_http_ipc_server :5001)")
    # None -> use groot_mover module defaults (the shipped fix). Override to A/B.
    ap.add_argument("--warmup-time", type=float, default=None)
    ap.add_argument("--settle-before", type=float, default=None)
    ap.add_argument("--min-duration", type=float, default=None)
    ap.add_argument("--min-distance", type=float, default=None)
    ap.add_argument("--dist-gain", type=float, default=None)
    ap.add_argument("--pause", type=float, default=3.0,
                    help="observe/settle time after each move (s)")
    ap.add_argument("--auto", action="store_true",
                    help="run the sequence hands-free (default: Enter-gated per move)")
    a = ap.parse_args()

    kw = {}
    for name, val in (("warmup_time", a.warmup_time), ("settle_before_s", a.settle_before),
                      ("min_duration", a.min_duration), ("min_distance", a.min_distance),
                      ("dist_gain", a.dist_gain)):
        if val is not None:
            kw[name] = val

    mv = RemoteMover(a.ipc_url, **kw)
    print("=" * 64)
    print("small-move test  ipc-url =", a.ipc_url)
    print(f"  knobs: warmup={mv.warmup_time}s@{mv.warmup_speed} settle_before={mv.settle_before_s}s "
          f"min_dur={mv.min_duration}s min_dist={mv.min_distance}m v_floor={mv.v_floor} "
          f"dist_gain={mv.dist_gain}")
    print("  watch: 每条命令后 base 是否【净前进】(warmup 期会先小步起来再走);绝不后退")
    print("=" * 64)

    mv.initialize()          # RL_FULL
    time.sleep(1.5)
    try:
        for kind, val in SEQ:
            unit = "cm" if kind != "rotate" else "deg"
            tag = f"{kind} {val:+d}{unit}"
            if a.auto:
                print(f"\n--- {tag} ---")
            else:
                input(f"\n[Enter] 执行: {tag}   (急停: Ctrl+C 或遥控 select)")
            if kind == "forward":
                mv.move_forward_cm(val)
            elif kind == "strafe":
                mv.strafe_cm(val)
            elif kind == "rotate":
                mv.rotate_deg(val)
            time.sleep(a.pause)
            mv.stop()
        print("\n完成。底座保持 RL_FULL 平衡。")
    except KeyboardInterrupt:
        print("\n[中断] 发 DAMP。")
        mv.damp()
        return
    mv.stop()


if __name__ == "__main__":
    main()
