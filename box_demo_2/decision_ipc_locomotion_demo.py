#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive decision-layer IPC demo for locomotion-only tests.

Type numeric commands such as:

  1 1.0      forward 1.0 m
  3 0.4      left strafe 0.4 m
  5 90       yaw left 90 deg
  7 0.55     set base height to 0.55 m

The script converts distance/angle requests into velocity plus duration and
refreshes IPC files while the command is active.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    import msvcrt
except ImportError:  # pragma: no cover - Linux robot host path
    msvcrt = None

try:
    import select
    import termios
    import tty
except ImportError:  # pragma: no cover - Windows local inspection path
    select = None
    termios = None
    tty = None

LEGACY_CMD_FILE = Path("/tmp/robojudo_ext_cmd.json")
SIM_STATE_DIR = Path("/tmp/agile_sim2sim")
STAND_HEIGHT = 0.74
MIN_HEIGHT = 0.40
MAX_HEIGHT = 0.80


@dataclass
class SolvedMotion:
    kind: str
    target: float
    vx: float
    vy: float
    wz: float
    duration: float
    expected_delta: float
    note: str = ""


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def sim_command_path(state_dir: Path) -> Path:
    return state_dir / "command.json"


def make_payloads(
    *,
    fsm: str,
    vx: float,
    vy: float,
    wz: float,
    height: float,
    duration: float | None,
    source: str,
    request: dict | None = None,
) -> tuple[dict, dict]:
    t = time.time()
    height = clamp(height, MIN_HEIGHT, MAX_HEIGHT)
    sim = {
        "vx": float(vx),
        "vy": float(vy),
        "wz": float(wz),
        "height": height,
        "fsm": fsm,
        "timestamp": t,
        "source": source,
    }
    legacy = {
        "fsm": fsm,
        "velocity": {"forward": float(vx), "lateral": float(vy), "yaw": float(wz)},
        "height": height,
        "units": "agile",
        "timestamp": t,
        "source": source,
    }
    if fsm == "LIMP":
        legacy["limp"] = True
        sim["limp"] = True
    if fsm == "DAMP":
        legacy["estop"] = True
        sim["estop"] = True
    if duration is not None and duration > 0:
        sim["duration"] = float(duration)
        sim["expires_at"] = t + float(duration)
        legacy["duration"] = float(duration)
    if request:
        sim["request"] = request
        legacy["request"] = request
    return sim, legacy


class IpcWriter:
    def __init__(self, cmd_file: Path, state_dir: Path, write_mode: str):
        self.cmd_file = cmd_file
        self.state_dir = state_dir
        self.write_mode = write_mode

    def write(self, sim_payload: dict, legacy_payload: dict) -> None:
        if self.write_mode in ("sim", "both"):
            atomic_write_json(sim_command_path(self.state_dir), sim_payload)
        if self.write_mode in ("legacy", "both"):
            atomic_write_json(self.cmd_file, legacy_payload)


class MotionInterruptWatcher:
    """Watch for a single-key stop while a motion command is being refreshed."""

    def __init__(self):
        self._fd: int | None = None
        self._old_termios = None
        self._raw_enabled = False

    def __enter__(self):
        if msvcrt is not None:
            return self
        if select is None or termios is None or tty is None:
            return self
        if not sys.stdin.isatty():
            return self
        self._fd = sys.stdin.fileno()
        self._old_termios = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self._raw_enabled = True
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._raw_enabled and self._fd is not None and self._old_termios is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_termios)
        return False

    def stop_requested(self) -> bool:
        if msvcrt is not None:
            while msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch in ("0", " "):
                    return True
                if ch == "\x03":
                    raise KeyboardInterrupt
            return False

        if not self._raw_enabled or select is None:
            return False
        ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not ready:
            return False
        ch = sys.stdin.read(1)
        if ch in ("0", " "):
            return True
        if ch == "\x03":
            raise KeyboardInterrupt
        return False


def solve_axis_motion(
    *,
    kind: str,
    distance: float,
    speed: float,
    max_speed: float,
    min_duration: float,
    vx_sign: float = 0.0,
    vy_sign: float = 0.0,
) -> SolvedMotion:
    distance = float(distance)
    if abs(distance) < 1e-6:
        return SolvedMotion(kind, distance, 0.0, 0.0, 0.0, 0.0, 0.0, "zero distance")

    direction = 1.0 if distance >= 0.0 else -1.0
    speed_abs = clamp(abs(speed), 1e-6, abs(max_speed))
    duration = abs(distance) / speed_abs
    note = ""
    if duration < min_duration:
        duration = min_duration
        speed_abs = abs(distance) / duration
        note = f"short request: speed reduced to preserve distance over {duration:.2f}s"

    vx = vx_sign * direction * speed_abs
    vy = vy_sign * direction * speed_abs
    expected = math.copysign(speed_abs * duration, distance)
    return SolvedMotion(kind, distance, vx, vy, 0.0, duration, expected, note)


def solve_yaw_motion(angle_deg: float, yaw_rate: float, max_yaw: float, min_duration: float) -> SolvedMotion:
    angle_rad = math.radians(float(angle_deg))
    if abs(angle_rad) < 1e-6:
        return SolvedMotion("yaw", angle_deg, 0.0, 0.0, 0.0, 0.0, 0.0, "zero angle")

    direction = 1.0 if angle_rad >= 0.0 else -1.0
    rate_abs = clamp(abs(yaw_rate), 1e-6, abs(max_yaw))
    duration = abs(angle_rad) / rate_abs
    note = ""
    if duration < min_duration:
        duration = min_duration
        rate_abs = abs(angle_rad) / duration
        note = f"short turn: yaw rate reduced to preserve angle over {duration:.2f}s"

    wz = direction * rate_abs
    expected_deg = math.degrees(wz * duration)
    return SolvedMotion("yaw", angle_deg, 0.0, 0.0, wz, duration, expected_deg, note)


def print_help() -> None:
    print(
        """
Numeric decision commands:
  1 <m> [speed]       forward distance, e.g. 1 1.0
  2 <m> [speed]       backward distance
  3 <m> [speed]       left strafe distance
  4 <m> [speed]       right strafe distance
  5 <deg> [rate]      yaw left, degrees
  6 <deg> [rate]      yaw right, degrees
  7 <height> [hold]   set/hold base height, meters
  8                  stand height
  9                  crouch preset
  0                  stop; also interrupts an active motion immediately

Aliases:
  f/b/l/r <m> [speed], yaw <deg> [rate], h <m>, stand, squat, stop
  demo               run a small forward-left-right-turn sequence
  Ctrl+C             write LIMP by default, so the adapter releases motor stiffness
  status             show current settings
  help, q
"""
    )


def publish_hold(writer: IpcWriter, args, *, fsm: str, height: float, hold_s: float, source: str, request: dict) -> None:
    hold_s = max(0.0, float(hold_s))
    deadline = time.time() + hold_s
    while True:
        sim, legacy = make_payloads(
            fsm=fsm,
            vx=0.0,
            vy=0.0,
            wz=0.0,
            height=height,
            duration=hold_s if hold_s > 0 else None,
            source=source,
            request=request,
        )
        writer.write(sim, legacy)
        if hold_s <= 0 or time.time() >= deadline:
            break
        time.sleep(1.0 / args.refresh_hz)


def run_motion(writer: IpcWriter, args, motion: SolvedMotion, *, fsm: str, height: float, source: str) -> None:
    if motion.duration <= 0:
        print(f"[skip] {motion.note or 'zero command'}")
        return

    request = asdict(motion)
    print(
        f"[send] {motion.kind}: target={motion.target:+.3f}, "
        f"cmd=({motion.vx:+.3f}, {motion.vy:+.3f}, {motion.wz:+.3f}), "
        f"duration={motion.duration:.2f}s, expected={motion.expected_delta:+.3f}"
    )
    if motion.note:
        print(f"       note: {motion.note}")
    print("       press 0 during this motion to interrupt and write zero velocity")

    deadline = time.time() + motion.duration
    interrupted = False
    with MotionInterruptWatcher() as watcher:
        while time.time() < deadline:
            if watcher.stop_requested():
                interrupted = True
                break
            remaining = max(0.0, deadline - time.time())
            sim, legacy = make_payloads(
                fsm=fsm,
                vx=motion.vx,
                vy=motion.vy,
                wz=motion.wz,
                height=height,
                duration=remaining,
                source=source,
                request=request,
            )
            writer.write(sim, legacy)
            time.sleep(1.0 / args.refresh_hz)

    stop_request = {"kind": "stop_after_motion"}
    stop_source = f"{source}_stop"
    if interrupted:
        stop_request = {
            "kind": "interrupt_stop",
            "interrupted_motion": request,
        }
        stop_source = f"{source}_interrupt_stop"
        print("\n[interrupt] 0 pressed; zero velocity written")

    publish_hold(
        writer,
        args,
        fsm=fsm,
        height=height,
        hold_s=args.stop_hold_s,
        source=stop_source,
        request=stop_request,
    )
    if not interrupted:
        print("[done] stop command written")


def parse_float(tokens: list[str], idx: int, default: float | None = None) -> float:
    if idx >= len(tokens):
        if default is None:
            raise ValueError("missing numeric argument")
        return default
    return float(tokens[idx])


def run_demo(writer: IpcWriter, args, fsm: str, height: float) -> None:
    sequence = [
        solve_axis_motion(kind="forward", distance=0.8, speed=args.fwd_speed, max_speed=args.fwd_max, min_duration=args.min_duration, vx_sign=1.0),
        solve_axis_motion(kind="left", distance=0.25, speed=args.lat_speed, max_speed=args.lat_max, min_duration=args.min_duration, vy_sign=1.0),
        solve_axis_motion(kind="right", distance=0.25, speed=args.lat_speed, max_speed=args.lat_max, min_duration=args.min_duration, vy_sign=-1.0),
        solve_yaw_motion(45.0, args.yaw_rate, args.yaw_max, args.min_duration),
    ]
    for motion in sequence:
        run_motion(writer, args, motion, fsm=fsm, height=height, source="decision_demo_sequence")
        time.sleep(0.4)


def publish_shutdown_action(writer: IpcWriter, args, *, current_fsm: str, height: float, reason: str) -> None:
    action = args.ctrl_c_action
    if action == "limp":
        fsm = "LIMP"
    elif action == "damp":
        fsm = "DAMP"
    else:
        fsm = current_fsm
    publish_hold(
        writer,
        args,
        fsm=fsm,
        height=height,
        hold_s=args.ctrl_c_hold_s,
        source=f"decision_demo_{reason}_{action}",
        request={"kind": reason, "action": action},
    )
    print(f"[{reason}] action={action} written")


def main() -> int:
    global MIN_HEIGHT, MAX_HEIGHT

    parser = argparse.ArgumentParser(description="Decision-layer numeric IPC locomotion demo")
    parser.add_argument("--cmd-file", type=Path, default=LEGACY_CMD_FILE)
    parser.add_argument("--state-dir", type=Path, default=SIM_STATE_DIR)
    parser.add_argument("--write", choices=("legacy", "sim", "both"), default="both")
    parser.add_argument("--refresh-hz", type=float, default=20.0)
    parser.add_argument("--fwd-speed", type=float, default=0.25)
    parser.add_argument("--lat-speed", type=float, default=0.15)
    parser.add_argument("--yaw-rate", type=float, default=0.35)
    parser.add_argument("--fwd-max", type=float, default=0.50)
    parser.add_argument("--lat-max", type=float, default=0.30)
    parser.add_argument("--yaw-max", type=float, default=0.60)
    parser.add_argument("--min-duration", type=float, default=0.60)
    parser.add_argument("--height", type=float, default=STAND_HEIGHT)
    parser.add_argument("--stand-height", type=float, default=STAND_HEIGHT)
    parser.add_argument("--min-height", type=float, default=MIN_HEIGHT)
    parser.add_argument("--max-height", type=float, default=MAX_HEIGHT)
    parser.add_argument("--crouch-height", type=float, default=0.55)
    parser.add_argument("--height-hold-s", type=float, default=1.0)
    parser.add_argument("--stop-hold-s", type=float, default=0.3)
    parser.add_argument("--ctrl-c-action", choices=("limp", "damp", "stop"), default="limp")
    parser.add_argument("--ctrl-c-hold-s", type=float, default=1.0)
    args = parser.parse_args()

    MIN_HEIGHT = float(args.min_height)
    MAX_HEIGHT = float(args.max_height)

    writer = IpcWriter(args.cmd_file.expanduser(), args.state_dir.expanduser(), args.write)
    height = clamp(float(args.height), MIN_HEIGHT, MAX_HEIGHT)
    fsm = "RL_FULL"

    print("=" * 72)
    print("Decision IPC locomotion demo")
    print(f"  write:      {args.write}")
    print(f"  legacy IPC: {writer.cmd_file}")
    print(f"  sim IPC:    {sim_command_path(writer.state_dir)}")
    print(f"  speed:      fwd={args.fwd_speed:.2f}m/s lat={args.lat_speed:.2f}m/s yaw={args.yaw_rate:.2f}rad/s")
    print(f"  height:     {height:.2f}m")
    print("=" * 72)
    print_help()

    publish_hold(writer, args, fsm=fsm, height=height, hold_s=0.0, source="decision_demo_init", request={"kind": "init"})

    try:
        while True:
            try:
                line = input("decision> ").strip()
            except EOFError:
                line = "q"
            if not line:
                continue

            try:
                tokens = shlex.split(line)
                cmd = tokens[0].lower()

                if cmd in ("q", "quit", "exit"):
                    publish_hold(writer, args, fsm=fsm, height=height, hold_s=0.0, source="decision_demo_exit", request={"kind": "exit"})
                    print("bye")
                    return 0
                if cmd in ("help", "?"):
                    print_help()
                    continue
                if cmd == "status":
                    print(
                        f"mode={fsm} height={height:.2f} write={args.write} "
                        f"fwd={args.fwd_speed:.2f} lat={args.lat_speed:.2f} yaw={args.yaw_rate:.2f}"
                    )
                    continue
                if cmd in ("0", "stop", "space"):
                    publish_hold(writer, args, fsm=fsm, height=height, hold_s=args.stop_hold_s, source="decision_demo_stop", request={"kind": "stop"})
                    print("[stop] zero velocity written")
                    continue
                if cmd in ("mode", "fsm"):
                    value = tokens[1].lower()
                    if value in ("full", "rl_full", "walk"):
                        fsm = "RL_FULL"
                    elif value in ("lower", "rl_lower"):
                        fsm = "RL_LOWER"
                    elif value in ("damp", "damping"):
                        fsm = "DAMP"
                    elif value in ("limp", "soft"):
                        fsm = "LIMP"
                    else:
                        raise ValueError("mode must be full/lower/damp/limp")
                    publish_hold(writer, args, fsm=fsm, height=height, hold_s=0.0, source="decision_demo_mode", request={"kind": "mode", "fsm": fsm})
                    print(f"[mode] {fsm}")
                    continue
                if cmd == "demo":
                    run_demo(writer, args, fsm, height)
                    continue

                if cmd in ("7", "h", "height"):
                    height = clamp(parse_float(tokens, 1), MIN_HEIGHT, MAX_HEIGHT)
                    hold_s = parse_float(tokens, 2, args.height_hold_s)
                    publish_hold(writer, args, fsm=fsm, height=height, hold_s=hold_s, source="decision_demo_height", request={"kind": "height", "height": height})
                    print(f"[height] {height:.2f}m")
                    continue
                if cmd in ("8", "stand"):
                    height = clamp(args.stand_height, MIN_HEIGHT, MAX_HEIGHT)
                    publish_hold(writer, args, fsm=fsm, height=height, hold_s=args.height_hold_s, source="decision_demo_stand", request={"kind": "stand"})
                    print(f"[stand] {height:.2f}m")
                    continue
                if cmd in ("9", "squat", "crouch"):
                    height = clamp(parse_float(tokens, 1, args.crouch_height), MIN_HEIGHT, MAX_HEIGHT)
                    hold_s = parse_float(tokens, 2, args.height_hold_s)
                    publish_hold(writer, args, fsm=fsm, height=height, hold_s=hold_s, source="decision_demo_squat", request={"kind": "squat", "height": height})
                    print(f"[squat] {height:.2f}m")
                    continue

                if cmd in ("1", "f", "forward"):
                    dist = abs(parse_float(tokens, 1))
                    speed = parse_float(tokens, 2, args.fwd_speed)
                    motion = solve_axis_motion(kind="forward", distance=dist, speed=speed, max_speed=args.fwd_max, min_duration=args.min_duration, vx_sign=1.0)
                elif cmd in ("2", "b", "back", "backward"):
                    dist = -abs(parse_float(tokens, 1))
                    speed = parse_float(tokens, 2, args.fwd_speed)
                    motion = solve_axis_motion(kind="backward", distance=dist, speed=speed, max_speed=args.fwd_max, min_duration=args.min_duration, vx_sign=1.0)
                elif cmd in ("3", "l", "left"):
                    dist = abs(parse_float(tokens, 1))
                    speed = parse_float(tokens, 2, args.lat_speed)
                    motion = solve_axis_motion(kind="left", distance=dist, speed=speed, max_speed=args.lat_max, min_duration=args.min_duration, vy_sign=1.0)
                elif cmd in ("4", "r", "right"):
                    dist = -abs(parse_float(tokens, 1))
                    speed = parse_float(tokens, 2, args.lat_speed)
                    motion = solve_axis_motion(kind="right", distance=dist, speed=speed, max_speed=args.lat_max, min_duration=args.min_duration, vy_sign=1.0)
                elif cmd in ("5", "yaw_left", "turn_left"):
                    angle = abs(parse_float(tokens, 1))
                    rate = parse_float(tokens, 2, args.yaw_rate)
                    motion = solve_yaw_motion(angle, rate, args.yaw_max, args.min_duration)
                elif cmd in ("6", "yaw_right", "turn_right"):
                    angle = -abs(parse_float(tokens, 1))
                    rate = parse_float(tokens, 2, args.yaw_rate)
                    motion = solve_yaw_motion(angle, rate, args.yaw_max, args.min_duration)
                elif cmd in ("yaw", "turn"):
                    angle = parse_float(tokens, 1)
                    rate = parse_float(tokens, 2, args.yaw_rate)
                    motion = solve_yaw_motion(angle, rate, args.yaw_max, args.min_duration)
                else:
                    print(f"unknown command: {cmd}. type help")
                    continue

                run_motion(writer, args, motion, fsm=fsm, height=height, source="decision_demo_motion")
            except Exception as exc:
                print(f"[error] {exc}")
    except KeyboardInterrupt:
        print()
        publish_shutdown_action(writer, args, current_fsm=fsm, height=height, reason="ctrl_c")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
