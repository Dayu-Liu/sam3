#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GR00T Decoupled WBC adapter for box_demo_2.

This process lets the existing box demo keep its command and arm merge path:

  /tmp/robojudo_ext_cmd.json -> GR00T-WBC policy -> rt/lowcmd_rl
  rt/lowcmd_rl + rt/arm_sdk -> merge_lowcmd_arm_sdk.py -> rt/lowcmd

Run this instead of the official GR00T run_g1_control_loop.py when using
box_demo_2. The official loop publishes rt/lowcmd directly, while this adapter
only publishes rt/lowcmd_rl so the original arm_sdk overlay still works.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CMD_FILE = "/tmp/robojudo_ext_cmd.json"

DEFAULT_BASE_HEIGHT = 0.74
DEFAULT_MIN_HEIGHT = 0.40
DEFAULT_MAX_HEIGHT = 0.80

# GR00T-WBC auto-switches Balance<->Walk at ||[vx,vy,wz]|| < 0.05
# (g1_gear_wbc_policy.py:223). A command inside (0, 0.05) makes the robot
# silently balance instead of step -> the open-loop distance is wrong with no
# error signal. Commands in that dead-band are snapped to an explicit no-move.
WALK_THRESHOLD_NOMOVE = 0.05

TOTAL_HG_MOTORS = 35
BODY_MOTORS = 29
ARM_SDK_ENABLE_SLOT = 29

TOTAL_HG_MOTORS = 35
BODY_MOTORS = 29
ARM_SDK_ENABLE_SLOT = 29


@dataclass
class ExternalCommand:
    fsm: str
    vx: float
    vy: float
    wz: float
    height: float
    fresh: bool
    estop: bool = False


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def approach(current: float, target: float, max_delta: float) -> float:
    if target > current:
        return min(target, current + max_delta)
    return max(target, current - max_delta)


def _load_ddsc() -> None:
    ddsc = (
        Path(os.environ.get("CYCLONEDDS_HOME", Path.home() / "cyclonedds-0.10-install"))
        / "lib"
        / "libddsc.so.0"
    )
    if ddsc.is_file():
        ctypes.CDLL(str(ddsc), mode=ctypes.RTLD_GLOBAL)


def read_external_command(
    cmd_file: Path,
    last_height: float,
    stale_s: float,
    fwd_max: float,
    lat_max: float,
    yaw_max: float,
    min_height: float,
    max_height: float,
) -> ExternalCommand:
    now = time.time()
    try:
        with cmd_file.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        return ExternalCommand("RL_FULL", 0.0, 0.0, 0.0, last_height, False)
    except Exception as exc:
        print(f"[WARN] cannot read command file {cmd_file}: {exc}")
        return ExternalCommand("RL_FULL", 0.0, 0.0, 0.0, last_height, False)

    age = now - float(raw.get("timestamp", 0.0))
    fresh = age <= stale_s
    fsm = str(raw.get("fsm") or "RL_FULL")
    height = clamp(float(raw.get("height", last_height)), min_height, max_height)

    if bool(raw.get("limp")) or fsm == "LIMP":
        return ExternalCommand("LIMP", 0.0, 0.0, 0.0, height, fresh, estop=True)

    if bool(raw.get("estop")) or fsm == "DAMP":
        return ExternalCommand("DAMP", 0.0, 0.0, 0.0, height, fresh, estop=True)

    if not fresh or fsm == "RL_LOWER":
        return ExternalCommand(fsm, 0.0, 0.0, 0.0, height, fresh)

    vel = raw.get("velocity") or {}
    vx = float(vel.get("forward", 0.0))
    vy = float(vel.get("lateral", 0.0))
    wz = float(vel.get("yaw", 0.0))

    # Existing RobotMover writes normalized values. agile_keyboard_control.py
    # writes physical units and marks them with units=agile.
    if raw.get("units") != "agile":
        vx *= fwd_max
        vy *= lat_max
        wz *= yaw_max

    vx = clamp(vx, -fwd_max, fwd_max)
    vy = clamp(vy, -lat_max, lat_max)
    wz = clamp(wz, -yaw_max, yaw_max)

    # Dead-band guard: snap a sub-Walk-threshold command to a clean no-move so it
    # never masquerades as motion (groot_mover.py floors above this, so its small
    # precise moves are unaffected).
    if 0.0 < math.sqrt(vx * vx + vy * vy + wz * wz) < WALK_THRESHOLD_NOMOVE:
        vx = vy = wz = 0.0

    return ExternalCommand(
        fsm=fsm,
        vx=vx,
        vy=vy,
        wz=wz,
        height=height,
        fresh=fresh,
    )


def import_groot_stack(repo: Path):
    repo = repo.expanduser().resolve()
    if not (repo / "decoupled_wbc").is_dir():
        raise SystemExit(f"GR00T-WholeBodyControl repo not found: {repo}")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    from decoupled_wbc.control.envs.g1.g1_env import G1Env
    from decoupled_wbc.control.main.teleop.configs.configs import ControlLoopConfig
    from decoupled_wbc.control.policy.wbc_policy_factory import get_wbc_policy
    from decoupled_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model

    return {
        "G1Env": G1Env,
        "ControlLoopConfig": ControlLoopConfig,
        "get_wbc_policy": get_wbc_policy,
        "instantiate_g1_robot_model": instantiate_g1_robot_model,
    }


class LowCmdRlPublisher:
    """Publish GR00T-WBC body targets to rt/lowcmd_rl instead of rt/lowcmd."""

    def __init__(self, wbc_config: dict[str, Any], topic: str = "rt/lowcmd_rl"):
        from unitree_sdk2py.core.channel import ChannelPublisher
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
        from unitree_sdk2py.utils.crc import CRC

        self.config = wbc_config
        self.topic = topic
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.pub = ChannelPublisher(topic, LowCmd_)
        self.pub.Init()
        self.crc = CRC()

        self.robot_kp = list(wbc_config["MOTOR_KP"])
        self.robot_kd = list(wbc_config["MOTOR_KD"])
        self.joint2motor = list(wbc_config["JOINT2MOTOR"])
        self.motor2joint = list(wbc_config["MOTOR2JOINT"])
        self.default_motor_angles = list(wbc_config.get("DEFAULT_MOTOR_ANGLES", [0.0] * BODY_MOTORS))
        self.weak_motor_indices = set(wbc_config.get("WeakMotorJointIndex", {}).values())
        self.pos_stop = float(wbc_config["UNITREE_LEGGED_CONST"]["PosStopF"])
        self.vel_stop = float(wbc_config["UNITREE_LEGGED_CONST"]["VelStopF"])
        self.mode_pr = int(wbc_config["UNITREE_LEGGED_CONST"]["MODE_PR"])
        self.mode_machine = int(wbc_config["UNITREE_LEGGED_CONST"]["MODE_MACHINE"])
        self._init_lowcmd()

    def _mode_for_motor(self, motor_index: int) -> int:
        return 0x01 if motor_index in self.weak_motor_indices else 0x0A

    def _init_lowcmd(self) -> None:
        if hasattr(self.low_cmd, "level_flag"):
            self.low_cmd.level_flag = 0xFF
        if hasattr(self.low_cmd, "gpio"):
            self.low_cmd.gpio = 0
        self.low_cmd.mode_pr = self.mode_pr
        self.low_cmd.mode_machine = self.mode_machine
        for i in range(min(TOTAL_HG_MOTORS, len(self.low_cmd.motor_cmd))):
            m = self.low_cmd.motor_cmd[i]
            m.mode = self._mode_for_motor(i) if i < BODY_MOTORS else 0x01
            m.q = self.pos_stop
            m.dq = self.vel_stop
            m.tau = 0.0
            m.kp = 0.0
            m.kd = 0.0

    def publish_body_targets(self, body_q, low_state=None) -> None:
        self.low_cmd.mode_pr = self.mode_pr
        self.low_cmd.mode_machine = self.mode_machine

        # Fill all 35 HG command slots. Body motors 0..28 get GR00T-WBC
        # targets. Extra slots are passive. Slot 29 must stay q=0.0 so the
        # merger never treats the RL stream as arm_sdk ownership.
        for i in range(min(TOTAL_HG_MOTORS, len(self.low_cmd.motor_cmd))):
            m = self.low_cmd.motor_cmd[i]
            m.mode = self._mode_for_motor(i) if i < BODY_MOTORS else 0x01
            if i < BODY_MOTORS:
                joint_index = self.motor2joint[i]
                motor_index = self.joint2motor[i]
                out = self.low_cmd.motor_cmd[motor_index]
                out.mode = self._mode_for_motor(motor_index)
                if joint_index == -1:
                    out.q = float(self.default_motor_angles[motor_index])
                else:
                    out.q = float(body_q[joint_index])
                out.dq = 0.0
                out.tau = 0.0
                out.kp = float(self.robot_kp[motor_index])
                out.kd = float(self.robot_kd[motor_index])
            else:
                q = 0.0
                if low_state is not None and i != ARM_SDK_ENABLE_SLOT:
                    try:
                        q = float(low_state.motor_state[i].q)
                    except Exception:
                        q = 0.0
                m.q = 0.0 if i == ARM_SDK_ENABLE_SLOT else q
                m.dq = 0.0
                m.tau = 0.0
                m.kp = 0.0
                m.kd = 0.0

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.pub.Write(self.low_cmd)

    def publish_damping(self, low_state, kd_value: float) -> None:
        self.low_cmd.mode_pr = self.mode_pr
        self.low_cmd.mode_machine = self.mode_machine
        for i in range(min(TOTAL_HG_MOTORS, len(self.low_cmd.motor_cmd))):
            m = self.low_cmd.motor_cmd[i]
            m.mode = self._mode_for_motor(i) if i < BODY_MOTORS else 0x01
            if i == ARM_SDK_ENABLE_SLOT:
                q = 0.0
            else:
                try:
                    q = float(low_state.motor_state[i].q)
                except Exception:
                    q = 0.0
            m.q = q
            m.dq = 0.0
            m.tau = 0.0
            m.kp = 0.0
            m.kd = float(kd_value)
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.pub.Write(self.low_cmd)

    def publish_limp(self) -> None:
        self.low_cmd.mode_pr = self.mode_pr
        self.low_cmd.mode_machine = self.mode_machine
        for i in range(min(TOTAL_HG_MOTORS, len(self.low_cmd.motor_cmd))):
            m = self.low_cmd.motor_cmd[i]
            m.mode = self._mode_for_motor(i) if i < BODY_MOTORS else 0x01
            m.q = 0.0 if i == ARM_SDK_ENABLE_SLOT else self.pos_stop
            m.dq = self.vel_stop
            m.tau = 0.0
            m.kp = 0.0
            m.kd = 0.0
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.pub.Write(self.low_cmd)


def _activate_policy_once(wbc_policy) -> None:
    if hasattr(wbc_policy, "activate_policy"):
        wbc_policy.activate_policy()
        return
    try:
        wbc_policy.lower_body_policy.use_policy_action = True
    except Exception:
        pass


def _latest_low_state(env):
    try:
        return env.body().body_state_processor.robot_low_state
    except Exception:
        return None


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="GR00T-WBC rt/lowcmd_rl adapter for box_demo_2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--groot-repo", type=Path, default=Path(os.environ.get("GROOT_WBC_REPO", "~/GR00T-WholeBodyControl")))
    p.add_argument("--interface", "--iface", dest="interface", default=os.environ.get("UNITREE_DDS_INTERFACE", "real"))
    p.add_argument("--domain", type=int, default=0)
    p.add_argument("--cmd-file", type=Path, default=Path(CMD_FILE))
    p.add_argument("--publish-topic", default="rt/lowcmd_rl")
    p.add_argument("--hz", type=float, default=50.0)
    p.add_argument("--cmd-stale-s", type=float, default=0.40)
    p.add_argument("--fwd-max", type=float, default=0.50)
    p.add_argument("--lat-max", type=float, default=0.30)
    p.add_argument("--yaw-max", type=float, default=0.60)
    p.add_argument("--stand-height", type=float, default=DEFAULT_BASE_HEIGHT)
    p.add_argument("--min-height", type=float, default=DEFAULT_MIN_HEIGHT)
    p.add_argument("--max-height", type=float, default=DEFAULT_MAX_HEIGHT)
    p.add_argument("--height-rate", type=float, default=0.20, help="height slew rate in m/s")
    p.add_argument("--wbc-model-path", default="policy/GR00T-WholeBodyControl-Balance.onnx,policy/GR00T-WholeBodyControl-Walk.onnx")
    p.add_argument("--control-frequency", type=int, default=50)
    p.add_argument("--sim-frequency", type=int, default=200)
    p.add_argument("--sim-sync-mode", action="store_true", help="step MuJoCo in this process instead of a sim thread")
    p.add_argument("--onscreen", action="store_true", help="show MuJoCo viewer when --interface resolves to sim")
    p.add_argument("--offscreen", action="store_true", help="enable offscreen rendering when supported")
    p.add_argument("--upper-body-joint-speed", type=float, default=1000.0)
    p.add_argument("--enable-waist", action="store_true", help="put waist in both upper/lower groups like official option")
    p.add_argument("--high-elbow-pose", action="store_true")
    p.add_argument("--damping-kd", type=float, default=8.0)
    p.add_argument("--shutdown-action", choices=("limp", "damp", "none"), default="limp",
                   help="command sent when this adapter exits by Ctrl+C")
    p.add_argument("--shutdown-s", type=float, default=1.0, help="seconds to publish shutdown command")
    p.add_argument("--torch-threads", type=int, default=int(os.environ.get("TORCH_THREADS", "1")))
    p.add_argument("--dry-run", action="store_true", help="compute policy but do not publish rt/lowcmd_rl")
    p.add_argument("--no-auto-activate-policy", action="store_true", help="leave WBC lower policy in current-q hold mode")
    p.add_argument("--disable-joint-safety", action="store_true", help="skip GR00T JointSafetyMonitor before publishing")
    p.add_argument("--print-every-s", type=float, default=1.0)
    p.add_argument("--log-pose", type=Path, default=None,
                   help="append sim ground-truth floating_base_pose to a CSV each "
                        "tick (for open-loop precision tests; sim only)")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", str(args.torch_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.torch_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(args.torch_threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(args.torch_threads))

    _load_ddsc()

    import numpy as np
    import torch

    torch.set_num_threads(max(1, int(args.torch_threads)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    G = import_groot_stack(args.groot_repo)
    ControlLoopConfig = G["ControlLoopConfig"]
    instantiate_g1_robot_model = G["instantiate_g1_robot_model"]
    G1Env = G["G1Env"]
    get_wbc_policy = G["get_wbc_policy"]

    config = ControlLoopConfig(
        interface=args.interface,
        control_frequency=int(args.control_frequency),
        sim_frequency=int(args.sim_frequency),
        sim_sync_mode=bool(args.sim_sync_mode),
        wbc_model_path=args.wbc_model_path,
        enable_waist=bool(args.enable_waist),
        with_hands=False,
        high_elbow_pose=bool(args.high_elbow_pose),
        upper_body_joint_speed=float(args.upper_body_joint_speed),
        verbose=False,
        enable_onscreen=bool(args.onscreen),
        enable_offscreen=bool(args.offscreen),
    )
    wbc_config = config.load_wbc_yaml()
    wbc_config["DOMAIN_ID"] = int(args.domain)

    waist_location = "lower_and_upper_body" if args.enable_waist else "lower_body"
    robot_model = instantiate_g1_robot_model(
        waist_location=waist_location,
        high_elbow_pose=bool(args.high_elbow_pose),
    )
    env = G1Env(
        env_name=config.env_name,
        robot_model=robot_model,
        config=wbc_config,
        wbc_version=config.wbc_version,
    )
    if env.sim and not config.sim_sync_mode:
        env.start_simulator()
    wbc_policy = get_wbc_policy(
        "g1",
        robot_model,
        wbc_config,
        init_time=time.monotonic(),
    )
    if not args.no_auto_activate_policy:
        _activate_policy_once(wbc_policy)

    publisher = LowCmdRlPublisher(wbc_config=wbc_config, topic=args.publish_topic)

    dt = 1.0 / float(args.hz)
    height_cmd = clamp(float(args.stand_height), args.min_height, args.max_height)
    last_print = 0.0

    pose_log = None
    if args.log_pose is not None:
        pose_log = open(args.log_pose, "w")
        pose_log.write("time,x,y,z,qw,qx,qy,qz,fsm,vx,vy,wz\n")

    print("=" * 72)
    print("GR00T-WBC adapter for box_demo_2")
    print(f"  repo:        {args.groot_repo.expanduser().resolve()}")
    print(f"  interface:   {config.interface} ({config.env_type}), domain={args.domain}")
    print(f"  cmd file:    {args.cmd_file}")
    print(f"  publish:     {'DRY-RUN' if args.dry_run else args.publish_topic} @ {args.hz:.1f}Hz")
    print(f"  limits:      fwd={args.fwd_max:.2f} lat={args.lat_max:.2f} yaw={args.yaw_max:.2f}")
    print(f"  height:      {args.min_height:.2f}..{args.max_height:.2f}m, rate={args.height_rate:.2f}m/s")
    print(f"  policy:      {'auto-activated' if not args.no_auto_activate_policy else 'current-q hold'}")
    print("  merger:      keep merge_lowcmd_arm_sdk.py running to publish final rt/lowcmd")
    print("=" * 72)

    try:
        while True:
            t0 = time.monotonic()
            if env.sim and config.sim_sync_mode:
                env.step_simulator()

            cmd = read_external_command(
                cmd_file=args.cmd_file,
                last_height=height_cmd,
                stale_s=float(args.cmd_stale_s),
                fwd_max=float(args.fwd_max),
                lat_max=float(args.lat_max),
                yaw_max=float(args.yaw_max),
                min_height=float(args.min_height),
                max_height=float(args.max_height),
            )
            height_cmd = approach(height_cmd, cmd.height, float(args.height_rate) * dt)

            try:
                obs = env.observe()
            except Exception as exc:
                now = time.monotonic()
                if now - last_print >= float(args.print_every_s):
                    print(f"[WAIT] no valid GR00T-WBC observation yet: {exc}")
                    last_print = now
                time.sleep(dt)
                continue

            low_state = _latest_low_state(env)

            if pose_log is not None:
                try:
                    p7 = obs["floating_base_pose"]
                    pose_log.write(
                        "%.4f,%.5f,%.5f,%.5f,%.6f,%.6f,%.6f,%.6f,%s,%.4f,%.4f,%.4f\n"
                        % (time.time(), float(p7[0]), float(p7[1]), float(p7[2]),
                           float(p7[3]), float(p7[4]), float(p7[5]), float(p7[6]),
                           cmd.fsm, cmd.vx, cmd.vy, cmd.wz))
                    pose_log.flush()
                except Exception:
                    pass

            if cmd.fsm == "LIMP":
                if not args.dry_run:
                    publisher.publish_limp()
            elif cmd.estop or cmd.fsm == "DAMP":
                if not args.dry_run and low_state is not None:
                    publisher.publish_damping(low_state, args.damping_kd)
            else:
                t_now = time.monotonic()
                wbc_policy.set_observation(obs)
                goal = {
                    "navigate_cmd": np.array([cmd.vx, cmd.vy, cmd.wz], dtype=np.float32),
                    "base_height_command": np.array([height_cmd], dtype=np.float32),
                    "target_time": t_now + dt,
                    "interpolation_garbage_collection_time": t_now - 2.0 * dt,
                }
                wbc_policy.set_goal(goal)
                action = wbc_policy.get_action(time=t_now)

                if not args.disable_joint_safety:
                    try:
                        action = env.safety_monitor.handle_violations(obs, action)["action"]
                    except Exception as exc:
                        print(f"[WARN] joint safety monitor failed; using raw action: {exc}")

                body_q = robot_model.get_body_actuated_joints(action["q"])
                if not args.dry_run:
                    publisher.publish_body_targets(body_q, low_state=low_state)

            now = time.monotonic()
            if now - last_print >= float(args.print_every_s):
                print(
                    f"[GR00T-WBC] fsm={cmd.fsm:8s} fresh={int(cmd.fresh)} "
                    f"cmd=({cmd.vx:+.2f},{cmd.vy:+.2f},{cmd.wz:+.2f},h={height_cmd:.2f})"
                )
                last_print = now

            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, dt - elapsed))
    except KeyboardInterrupt:
        print(f"\nStopping GR00T-WBC adapter; shutdown_action={args.shutdown_action}...")
        if not args.dry_run and args.shutdown_action != "none":
            for _ in range(max(1, int(float(args.shutdown_s) * args.hz))):
                if args.shutdown_action == "limp":
                    publisher.publish_limp()
                elif args.shutdown_action == "damp":
                    low_state = _latest_low_state(env)
                    if low_state is not None:
                        publisher.publish_damping(low_state, args.damping_kd)
                time.sleep(dt)
    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
