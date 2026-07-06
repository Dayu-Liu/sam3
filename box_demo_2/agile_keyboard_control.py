#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keyboard command writer for the AGILE box_demo_2 lower-body pipeline.

Writes /tmp/robojudo_ext_cmd.json.  agile_lowcmd_pipeline.py reads the same IPC
file and feeds AGILE's [vx, vy, wz, absolute_height] command.

Keys:
  w/s: forward/backward
  a/d: left/right strafe
  q/e: yaw left/right
  z/x: lower/raise base height by --height-step
  c:   set pick-height preset
  r:   return to stand height
  space: stop velocity, keep current height
  f:   RL_FULL mode
  l:   RL_LOWER mode (legs balance, upper body free for arm_sdk)
  o:   damping/e-stop request
  Ctrl+C: exit
"""

from __future__ import annotations

import argparse
import json
import os
import select
import sys
import termios
import time
import tty

CMD_FILE = "/tmp/robojudo_ext_cmd.json"
STAND_HEIGHT = 0.72
MIN_HEIGHT = 0.40
MAX_HEIGHT = 0.72


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def write_cmd(fsm: str, vx: float, vy: float, wz: float, height: float,
              estop: bool = False):
    cmd = {
        "fsm": "DAMP" if estop else fsm,
        "velocity": {"forward": vx, "lateral": vy, "yaw": wz},
        "height": clamp(height, MIN_HEIGHT, MAX_HEIGHT),
        "units": "agile",
        "timestamp": time.time(),
    }
    if estop:
        cmd["estop"] = True
    tmp = f"{CMD_FILE}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cmd, f)
    os.replace(tmp, CMD_FILE)


def key_ready(timeout: float) -> bool:
    r, _, _ = select.select([sys.stdin], [], [], timeout)
    return bool(r)


def main():
    p = argparse.ArgumentParser(description="Keyboard IPC control for AGILE box_demo_2")
    p.add_argument("--vx", type=float, default=0.20, help="W/S speed in m/s")
    p.add_argument("--vy", type=float, default=0.12, help="A/D speed in m/s")
    p.add_argument("--wz", type=float, default=0.15, help="Q/E yaw rate in rad/s")
    p.add_argument("--height-step", type=float, default=0.02, help="Z/X height increment in m")
    p.add_argument("--pick-height", type=float, default=0.55, help="C key preset height in m")
    p.add_argument("--refresh-hz", type=float, default=20.0)
    p.add_argument("--key-timeout", type=float, default=0.25, help="velocity expires after no key input")
    args = p.parse_args()

    height = STAND_HEIGHT
    fsm = "RL_FULL"
    vx = vy = wz = 0.0
    last_motion_key = 0.0
    dirty = True
    dt = 1.0 / args.refresh_hz

    print("=" * 64)
    print("AGILE keyboard control -> /tmp/robojudo_ext_cmd.json")
    print("  w/s forward/back, a/d strafe, q/e yaw")
    print("  z/x height -/+ 2cm, c pick-height, r stand, space stop")
    print("  f RL_FULL, l RL_LOWER, o damping/e-stop, Ctrl+C exit")
    print("=" * 64)

    old = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while True:
            if key_ready(dt):
                ch = sys.stdin.read(1)
                if ch == "\x03":
                    raise KeyboardInterrupt
                if ch == "w":
                    vx, vy, wz = args.vx, 0.0, 0.0
                    last_motion_key = time.time()
                elif ch == "s":
                    vx, vy, wz = -args.vx, 0.0, 0.0
                    last_motion_key = time.time()
                elif ch == "a":
                    vx, vy, wz = 0.0, args.vy, 0.0
                    last_motion_key = time.time()
                elif ch == "d":
                    vx, vy, wz = 0.0, -args.vy, 0.0
                    last_motion_key = time.time()
                elif ch == "q":
                    vx, vy, wz = 0.0, 0.0, args.wz
                    last_motion_key = time.time()
                elif ch == "e":
                    vx, vy, wz = 0.0, 0.0, -args.wz
                    last_motion_key = time.time()
                elif ch == "z":
                    height = clamp(height - args.height_step, MIN_HEIGHT, MAX_HEIGHT)
                    dirty = True
                elif ch == "x":
                    height = clamp(height + args.height_step, MIN_HEIGHT, MAX_HEIGHT)
                    dirty = True
                elif ch == "c":
                    height = clamp(args.pick_height, MIN_HEIGHT, MAX_HEIGHT)
                    dirty = True
                elif ch == "r":
                    height = STAND_HEIGHT
                    dirty = True
                elif ch == " ":
                    vx = vy = wz = 0.0
                    dirty = True
                elif ch == "f":
                    fsm = "RL_FULL"
                    dirty = True
                elif ch == "l":
                    fsm = "RL_LOWER"
                    vx = vy = wz = 0.0
                    dirty = True
                elif ch == "o":
                    write_cmd(fsm, 0.0, 0.0, 0.0, height, estop=True)
                    print("\n[DAMP] e-stop request written")
                    continue

            motion_active = time.time() - last_motion_key <= args.key_timeout
            if not motion_active and (vx != 0.0 or vy != 0.0 or wz != 0.0):
                vx = vy = wz = 0.0
                dirty = True

            if motion_active or dirty:
                write_cmd(fsm, vx, vy, wz, height)
                dirty = False
            print(
                f"\rmode={fsm:8s} vx={vx:+.2f} vy={vy:+.2f} wz={wz:+.2f} height={height:.2f}m",
                end="",
                flush=True,
            )
    except KeyboardInterrupt:
        print("\nexit: stop velocity, keep current height")
        write_cmd(fsm, 0.0, 0.0, 0.0, height)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)


if __name__ == "__main__":
    main()
