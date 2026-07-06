#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AGILE lower-body pipeline for box_demo_2 real-robot deployment.

This process replaces the original RoboJuDo run_pipeline.py in the box demo:

  /tmp/robojudo_ext_cmd.json  ->  AGILE velocity+height policy  -> rt/lowcmd_rl

The upper-body process is unchanged.  box_demo_main.py still does camera/VLM/SAM3
perception, torso-frame target conversion, IK, and publishes rt/arm_sdk.  The
existing merge_lowcmd_arm_sdk.py keeps using rt/arm_sdk to override motors
12..29 during grasp, while this file only supplies the AGILE lower-body base.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import sys
import tempfile
import threading
import time
import types
from dataclasses import dataclass
from pathlib import Path

CMD_FILE = "/tmp/robojudo_ext_cmd.json"

DEFAULT_REPO = Path(__file__).resolve().parents[1] / "cc" / "experiments" / "repos" / "WBC-AGILE"
REL_POLICY = "agile/data/policy/velocity_height_g1/unitree_g1_velocity_height_recurrent_student.pt"
REL_CONFIG = "agile/data/policy/velocity_height_g1/unitree_g1_velocity_height_recurrent_student.yaml"

STAND_HEIGHT = 0.72
MIN_HEIGHT = 0.40
MAX_HEIGHT = 0.72

NUM_MOTORS = 35
MOTOR_MODE_ENABLE = 0x01

# Unitree G1 hg motor order used by box_demo_2/read_state.py and arm_sdk.
MOTOR_BY_JOINT = {
    "left_hip_pitch_joint": 0,
    "left_hip_roll_joint": 1,
    "left_hip_yaw_joint": 2,
    "left_knee_joint": 3,
    "left_ankle_pitch_joint": 4,
    "left_ankle_roll_joint": 5,
    "right_hip_pitch_joint": 6,
    "right_hip_roll_joint": 7,
    "right_hip_yaw_joint": 8,
    "right_knee_joint": 9,
    "right_ankle_pitch_joint": 10,
    "right_ankle_roll_joint": 11,
    "waist_yaw_joint": 12,
    "waist_roll_joint": 13,
    "waist_pitch_joint": 14,
    "left_shoulder_pitch_joint": 15,
    "left_shoulder_roll_joint": 16,
    "left_shoulder_yaw_joint": 17,
    "left_elbow_joint": 18,
    "left_wrist_roll_joint": 19,
    "left_wrist_pitch_joint": 20,
    "left_wrist_yaw_joint": 21,
    "right_shoulder_pitch_joint": 22,
    "right_shoulder_roll_joint": 23,
    "right_shoulder_yaw_joint": 24,
    "right_elbow_joint": 25,
    "right_wrist_roll_joint": 26,
    "right_wrist_pitch_joint": 27,
    "right_wrist_yaw_joint": 28,
}


@dataclass
class HardwareSimState:
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    root_pos: torch.Tensor
    root_quat: torch.Tensor
    root_lin_vel: torch.Tensor
    root_ang_vel: torch.Tensor
    joint_effort: torch.Tensor | None = None
    anchor_body_pos: torch.Tensor | None = None
    anchor_body_quat: torch.Tensor | None = None


@dataclass
class JointCommand:
    position: torch.Tensor
    kp: torch.Tensor
    kd: torch.Tensor


class _DummyMuJocoSimulation:
    pass


def _load_ddsc() -> None:
    ddsc = (
        Path(os.environ.get("CYCLONEDDS_HOME", Path.home() / "cyclonedds-0.10-install"))
        / "lib"
        / "libddsc.so.0"
    )
    if ddsc.is_file():
        ctypes.CDLL(str(ddsc), mode=ctypes.RTLD_GLOBAL)


def _motion_mode_name(result) -> str:
    if isinstance(result, dict):
        return str(result.get("name", "") or "")
    return str(getattr(result, "name", "") or "")


def release_unitree_motion_mode(timeout_s: float, retry_s: float, client_timeout_s: float) -> bool:
    """Release Unitree high-level motion service before low-level control.

    This mirrors Unitree/SONIC low-level examples: CheckMode() reports active
    sport/ai/advanced motion services, and ReleaseMode() deactivates them so
    our rt/lowcmd path is not fighting a high-level controller.
    """
    try:
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
    except Exception as exc:
        print(f"[motion] MotionSwitcherClient unavailable: {exc}")
        return False

    msc = MotionSwitcherClient()
    msc.SetTimeout(float(client_timeout_s))
    msc.Init()

    deadline = time.monotonic() + max(0.0, float(timeout_s))
    retry_s = max(0.1, float(retry_s))
    last_name = ""
    while True:
        try:
            status, result = msc.CheckMode()
        except Exception as exc:
            print(f"[motion] CheckMode failed: {exc}")
            return False

        name = _motion_mode_name(result)
        last_name = name
        if not name:
            print("[motion] Unitree motion service released; low-level command path is available.")
            return True

        print(f"[motion] active Unitree motion mode '{name}' (status={status}); calling ReleaseMode()")
        try:
            ret = msc.ReleaseMode()
        except Exception as exc:
            print(f"[motion] ReleaseMode failed: {exc}")
            return False
        if ret == 0:
            print("[motion] ReleaseMode succeeded.")
        else:
            print(f"[motion] ReleaseMode returned error code {ret}; retrying.")

        if time.monotonic() >= deadline:
            print(f"[motion] timed out while releasing Unitree motion mode '{last_name}'.")
            return False
        time.sleep(retry_s)


def _install_agile_stubs() -> None:
    """Let AGILE import without mujoco/pandas/matplotlib on the robot."""
    sim_mod = types.ModuleType("agile.sim2mujoco.simulation")
    sim_mod.JointCommand = JointCommand
    sim_mod.SimState = HardwareSimState
    sim_mod.MuJocoSimulation = _DummyMuJocoSimulation
    sys.modules["agile.sim2mujoco.simulation"] = sim_mod

    sched_mod = types.ModuleType("agile.sim2mujoco.command_scheduler")
    sched_mod.Sim2MuJoCoCommandScheduler = type("Sim2MuJoCoCommandScheduler", (), {})
    sys.modules["agile.sim2mujoco.command_scheduler"] = sched_mod

    log_mod = types.ModuleType("agile.sim2mujoco.data_logger")
    log_mod.Sim2MuJoCoDataLogger = type("Sim2MuJoCoDataLogger", (), {})
    sys.modules["agile.sim2mujoco.data_logger"] = log_mod


def import_agile(repo: Path):
    if not (repo / "agile" / "sim2mujoco").is_dir():
        raise SystemExit(f"AGILE repo not found: {repo}")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    _install_agile_stubs()
    from agile.sim2mujoco.actions import ActionProcessor
    from agile.sim2mujoco.commands import CommandManager
    from agile.sim2mujoco.observations import ObservationProcessor
    from agile.sim2mujoco.policy import PolicyWrapper

    return types.SimpleNamespace(
        ActionProcessor=ActionProcessor,
        CommandManager=CommandManager,
        ObservationProcessor=ObservationProcessor,
        PolicyWrapper=PolicyWrapper,
    )


class LowStateBuffer:
    def __init__(self):
        self._msg = None
        self._stamp = 0.0
        self._lock = threading.Lock()

    def update(self, msg) -> None:
        with self._lock:
            self._msg = msg
            self._stamp = time.monotonic()

    def snapshot(self):
        with self._lock:
            return self._msg, self._stamp

    def wait(self, timeout_s: float = 5.0):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            msg, _ = self.snapshot()
            if msg is not None and getattr(msg, "tick", 0) != 0:
                return msg
            time.sleep(0.01)
        raise TimeoutError("no valid rt/lowstate received")


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def policy_joint_names(config: dict) -> list[str]:
    return list(config["articulations"]["robot"]["joint_names"])


def motor_to_policy_arrays(msg, joint_names: list[str], device: torch.device):
    q, dq, tau = [], [], []
    for name in joint_names:
        mi = MOTOR_BY_JOINT[name]
        ms = msg.motor_state[mi]
        q.append(float(ms.q))
        dq.append(float(ms.dq))
        tau.append(float(getattr(ms, "tau_est", 0.0)))
    return (
        torch.tensor(q, dtype=torch.float32, device=device),
        torch.tensor(dq, dtype=torch.float32, device=device),
        torch.tensor(tau, dtype=torch.float32, device=device),
    )


def lowstate_to_sim_state(msg, joint_names: list[str], device: torch.device) -> HardwareSimState:
    q, dq, tau = motor_to_policy_arrays(msg, joint_names, device)
    imu = msg.imu_state
    quat = torch.tensor(list(imu.quaternion), dtype=torch.float32, device=device)
    gyro = torch.tensor(list(imu.gyroscope), dtype=torch.float32, device=device)
    return HardwareSimState(
        joint_pos=q,
        joint_vel=dq,
        root_pos=torch.zeros(3, dtype=torch.float32, device=device),
        root_quat=quat,
        root_lin_vel=torch.zeros(3, dtype=torch.float32, device=device),
        root_ang_vel=gyro,  # Unitree IMU gyroscope is already body-frame.
        joint_effort=tau,
    )


def zero_policy_recurrent_state(policy) -> int:
    n = 0
    try:
        for name, buf in policy.model.named_buffers():
            if "hidden" in name or "cell" in name:
                buf.zero_()
                n += 1
    except Exception:
        pass
    return n


def read_external_command(last_height: float, stale_s: float, fwd_max: float,
                          lat_max: float, yaw_max: float):
    now = time.time()
    try:
        with open(CMD_FILE, "r", encoding="utf-8") as f:
            cmd = json.load(f)
    except FileNotFoundError:
        return "RL_FULL", 0.0, 0.0, 0.0, last_height, False
    except Exception as exc:
        print(f"[WARN] bad command file: {exc}")
        return "RL_FULL", 0.0, 0.0, 0.0, last_height, False

    age = now - float(cmd.get("timestamp", 0.0))
    fresh = age <= stale_s
    fsm = cmd.get("fsm") or "RL_FULL"
    height = float(cmd.get("height", last_height))
    height = max(MIN_HEIGHT, min(MAX_HEIGHT, height))

    if bool(cmd.get("estop")) or fsm == "DAMP":
        return "DAMP", 0.0, 0.0, 0.0, height, fresh

    if not fresh or fsm == "RL_LOWER":
        return fsm, 0.0, 0.0, 0.0, height, fresh

    vel = cmd.get("velocity") or {}
    vx = float(vel.get("forward", 0.0))
    vy = float(vel.get("lateral", 0.0))
    wz = float(vel.get("yaw", 0.0))

    if cmd.get("units") != "agile":
        vx *= fwd_max
        vy *= lat_max
        wz *= yaw_max

    vx = max(-fwd_max, min(fwd_max, vx))
    vy = max(-lat_max, min(lat_max, vy))
    wz = max(-yaw_max, min(yaw_max, wz))
    return fsm, vx, vy, wz, height, fresh


def approach(current: float, target: float, max_delta: float) -> float:
    if target > current:
        return min(target, current + max_delta)
    return max(target, current - max_delta)


def _stamp_motor(cmd, idx: int, q: float, kp: float, kd: float) -> None:
    m = cmd.motor_cmd[idx]
    m.mode = MOTOR_MODE_ENABLE
    m.q = float(q)
    m.dq = 0.0
    m.kp = float(kp)
    m.kd = float(kd)
    m.tau = 0.0


def pose_by_motor(joint_names: list[str], pose) -> dict[int, float]:
    if hasattr(pose, "detach"):
        values = pose.detach().cpu().tolist()
    else:
        values = list(pose)
    return {MOTOR_BY_JOINT[name]: float(values[i]) for i, name in enumerate(joint_names)}


def build_lowcmd(msg, joint_names: list[str], joint_cmd: JointCommand,
                 upper_hold_kp: float, upper_hold_kd: float, mode_pr: int,
                 upper_target_by_motor: dict[int, float] | None = None):
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_

    out = unitree_hg_msg_dds__LowCmd_()
    out.mode_pr = int(mode_pr)
    out.mode_machine = int(getattr(msg, "mode_machine", 0))

    # Start with passive measured positions. Lower joints are replaced by AGILE
    # policy targets below. Upper joints use AGILE's default pose while no fresh
    # rt/arm_sdk overlay is active, so manual base tests do not leave the
    # torso/arms limp or hanging from the gantry.
    for i in range(NUM_MOTORS):
        q = float(msg.motor_state[i].q) if i < len(msg.motor_state) else 0.0
        _stamp_motor(out, i, q, 0.0, 0.0)

    pos = joint_cmd.position.detach().cpu().numpy()
    kp = joint_cmd.kp.detach().cpu().numpy()
    kd = joint_cmd.kd.detach().cpu().numpy()
    for ji, name in enumerate(joint_names):
        mi = MOTOR_BY_JOINT[name]
        if mi <= 11:
            _stamp_motor(out, mi, pos[ji], kp[ji], kd[ji])
        elif 12 <= mi <= 28:
            q = float(msg.motor_state[mi].q)
            if upper_target_by_motor is not None and mi in upper_target_by_motor:
                q = upper_target_by_motor[mi]
            _stamp_motor(out, mi, q, upper_hold_kp, upper_hold_kd)

    # arm_sdk enable slot: AGILE lower pipeline never requests upper override.
    _stamp_motor(out, 29, 0.0, 0.0, 0.0)
    return out


def build_damping_lowcmd(msg, kd_value: float, mode_pr: int):
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_

    out = unitree_hg_msg_dds__LowCmd_()
    out.mode_pr = int(mode_pr)
    out.mode_machine = int(getattr(msg, "mode_machine", 0))
    for i in range(NUM_MOTORS):
        q = float(msg.motor_state[i].q) if i < len(msg.motor_state) else 0.0
        _stamp_motor(out, i, q, 0.0, kd_value)
    return out


def publish_prepare(pub, crc, msg, joint_names: list[str], default_pos: torch.Tensor,
                    kp: torch.Tensor, kd: torch.Tensor, prepare_s: float,
                    hz: float, mode_pr: int, dry_run: bool,
                    upper_hold_kp: float, upper_hold_kd: float) -> None:
    if prepare_s <= 0:
        return
    n = max(1, int(round(prepare_s * hz)))
    q0 = np.array([float(msg.motor_state[MOTOR_BY_JOINT[nm]].q) for nm in joint_names], dtype=np.float32)
    q1 = default_pos.detach().cpu().numpy()
    for i in range(n):
        alpha = (i + 1) / n
        alpha = 0.5 - 0.5 * math.cos(math.pi * alpha)
        pos = torch.tensor(q0 * (1.0 - alpha) + q1 * alpha, dtype=torch.float32, device=default_pos.device)
        jc = JointCommand(position=pos, kp=kp, kd=kd)
        out = build_lowcmd(
            msg, joint_names, jc,
            upper_hold_kp=upper_hold_kp,
            upper_hold_kd=upper_hold_kd,
            mode_pr=mode_pr,
            upper_target_by_motor=pose_by_motor(joint_names, pos),
        )
        out.crc = crc.Crc(out)
        if not dry_run:
            pub.Write(out)
        time.sleep(1.0 / hz)


def main():
    p = argparse.ArgumentParser(description="AGILE lower-body rt/lowcmd_rl pipeline for box_demo_2")
    p.add_argument("--iface", default=os.environ.get("UNITREE_DDS_INTERFACE", "enP8p1s0"))
    p.add_argument("--domain", type=int, default=0)
    p.add_argument("--agile-repo", type=Path, default=Path(os.environ.get("AGILE_REPO", DEFAULT_REPO)))
    p.add_argument("--policy", type=Path, default=None)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--hz", type=float, default=50.0)
    p.add_argument("--mode-pr", type=int, default=0)
    p.add_argument("--cmd-stale-s", type=float, default=0.4)
    p.add_argument("--height-rate", type=float, default=0.20, help="max height command slew rate, m/s")
    p.add_argument("--fwd-max", type=float, default=0.50, help="legacy normalized forward=1 maps to this m/s")
    p.add_argument("--lat-max", type=float, default=0.30, help="legacy normalized lateral=1 maps to this m/s")
    p.add_argument("--yaw-max", type=float, default=0.60, help="legacy normalized yaw=1 maps to this rad/s")
    p.add_argument("--upper-hold-kp", type=float, default=40.0)
    p.add_argument("--upper-hold-kd", type=float, default=1.0)
    p.add_argument("--prepare-s", type=float, default=2.0)
    p.add_argument("--damping-kd", type=float, default=8.0)
    p.add_argument("--torch-threads", type=int, default=int(os.environ.get("TORCH_THREADS", "1")))
    p.add_argument("--dry-run", action="store_true", help="compute policy but do not publish rt/lowcmd_rl")
    p.add_argument("--no-motion-release", dest="motion_release", action="store_false",
                   help="do not call Unitree MotionSwitcherClient.ReleaseMode before live low-level control")
    p.add_argument("--motion-release-in-dry-run", action="store_true",
                   help="also call ReleaseMode during --dry-run; off by default to keep dry-run non-invasive")
    p.add_argument("--motion-release-timeout-s", type=float, default=20.0,
                   help="maximum time spent releasing Unitree high-level motion mode")
    p.add_argument("--motion-release-retry-s", type=float, default=1.0,
                   help="delay between ReleaseMode retries")
    p.add_argument("--motion-release-client-timeout-s", type=float, default=5.0,
                   help="MotionSwitcherClient RPC timeout")
    p.add_argument("--allow-motion-release-failure", action="store_true",
                   help="continue even if ReleaseMode fails; not recommended for hardware")
    p.set_defaults(motion_release=True)
    args = p.parse_args()

    global np, torch, yaml
    import numpy as np
    import torch
    import yaml
    torch.set_num_threads(max(1, int(args.torch_threads)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    _load_ddsc()
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
    from unitree_sdk2py.utils.crc import CRC

    repo = args.agile_repo.expanduser().resolve()
    policy_path = (args.policy or repo / REL_POLICY).expanduser().resolve()
    config_path = (args.config or repo / REL_CONFIG).expanduser().resolve()
    config = load_config(config_path)
    device = torch.device(args.device)
    A = import_agile(repo)

    joint_names = policy_joint_names(config)
    missing = [n for n in joint_names if n not in MOTOR_BY_JOINT]
    if missing:
        raise SystemExit(f"AGILE joint names missing motor mapping: {missing}")

    mgr = A.CommandManager(device=device, defaults={
        "linear_x": 0.0,
        "linear_y": 0.0,
        "angular_z": 0.0,
        "height": STAND_HEIGHT,
    })
    mgr.linear_x_range = (-args.fwd_max, args.fwd_max)
    mgr.linear_y_range = (-args.lat_max, args.lat_max)
    mgr.angular_z_range = (-args.yaw_max, args.yaw_max)
    mgr.height_range = (MIN_HEIGHT, MAX_HEIGHT)

    obs_processor = A.ObservationProcessor(config, joint_names, device, command_manager=mgr)
    act_processor = A.ActionProcessor(config, joint_names, device)
    policy = A.PolicyWrapper.from_config(policy_path, config, device)
    n_zeroed = zero_policy_recurrent_state(policy)

    state_buf = LowStateBuffer()
    ChannelFactoryInitialize(args.domain, args.iface)
    should_release_motion = args.motion_release and (not args.dry_run or args.motion_release_in_dry_run)
    if should_release_motion:
        released = release_unitree_motion_mode(
            args.motion_release_timeout_s,
            args.motion_release_retry_s,
            args.motion_release_client_timeout_s,
        )
        if not released and not args.allow_motion_release_failure:
            raise SystemExit(
                "failed to release Unitree high-level motion mode; refusing to start "
                "AGILE low-level control. Use --allow-motion-release-failure only "
                "for a supervised diagnostic run."
            )
    elif args.motion_release and args.dry_run:
        print("[motion] dry-run: skipping ReleaseMode; pass --motion-release-in-dry-run to test it.")
    else:
        print("[motion] ReleaseMode disabled by --no-motion-release.")
    pub = ChannelPublisher("rt/lowcmd_rl", LowCmd_)
    pub.Init()
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(state_buf.update, 10)
    crc = CRC()

    print("=" * 72)
    print("AGILE lower-body pipeline for box_demo_2")
    print(f"  iface/domain: {args.iface}/{args.domain}")
    print(f"  AGILE repo:    {repo}")
    print(f"  policy:        {policy_path}")
    print(f"  config:        {config_path}")
    print(f"  publish:       {'DRY-RUN' if args.dry_run else 'rt/lowcmd_rl'} @ {args.hz:.1f}Hz")
    print(f"  torch threads: {torch.get_num_threads()}")
    print(f"  command IPC:   {CMD_FILE}")
    print(f"  recurrent buffers zeroed: {n_zeroed}")
    print(f"  upper body:    AGILE default pose hold kp={args.upper_hold_kp:.1f} kd={args.upper_hold_kd:.1f}")
    print("  height keys:   z/x in agile_keyboard_control.py, safe range 0.40..0.72m")
    print("=" * 72)

    first_msg = state_buf.wait(timeout_s=5.0)
    default_pos = act_processor.default_joint_pos.detach().clone()
    upper_default_by_motor = pose_by_motor(joint_names, default_pos)
    publish_prepare(
        pub, crc, first_msg, joint_names, default_pos,
        act_processor.kp.detach(), act_processor.kd.detach(),
        args.prepare_s, args.hz, args.mode_pr, args.dry_run,
        args.upper_hold_kp, args.upper_hold_kd,
    )
    obs_processor.reset()
    zero_policy_recurrent_state(policy)

    dt = 1.0 / args.hz
    height_cmd = STAND_HEIGHT
    last_print = 0.0
    print("AGILE pipeline running. Ctrl+C to stop.")
    try:
        while True:
            t0 = time.monotonic()
            msg, stamp = state_buf.snapshot()
            if msg is None or time.monotonic() - stamp > 0.5:
                time.sleep(0.01)
                continue

            fsm, vx, vy, wz, target_h, fresh = read_external_command(
                height_cmd, args.cmd_stale_s, args.fwd_max, args.lat_max, args.yaw_max
            )
            height_cmd = approach(height_cmd, target_h, args.height_rate * dt)

            if fsm == "DAMP":
                out = build_damping_lowcmd(msg, args.damping_kd, args.mode_pr)
            else:
                mgr.set_command(vx, vy, wz, height_cmd)
                sim_state = lowstate_to_sim_state(msg, joint_names, device)
                obs = obs_processor.compute(sim_state)
                with torch.no_grad():
                    actions = policy(obs)
                obs_processor.set_last_action(actions)
                joint_cmd = act_processor.process(actions)
                out = build_lowcmd(
                    msg, joint_names, joint_cmd,
                    args.upper_hold_kp, args.upper_hold_kd, args.mode_pr,
                    upper_target_by_motor=upper_default_by_motor,
                )

            out.crc = crc.Crc(out)
            if not args.dry_run:
                pub.Write(out)

            now = time.monotonic()
            if now - last_print > 1.0:
                print(
                    f"[AGILE] fsm={fsm:8s} fresh={int(fresh)} "
                    f"cmd=({vx:+.2f},{vy:+.2f},{wz:+.2f},h={height_cmd:.2f}) "
                    f"tick={getattr(msg, 'tick', 0)}"
                )
                last_print = now

            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, dt - elapsed))
    except KeyboardInterrupt:
        print("\nStopping AGILE pipeline; sending damping frames...")
        for _ in range(int(0.5 * args.hz)):
            msg, _ = state_buf.snapshot()
            if msg is None:
                break
            out = build_damping_lowcmd(msg, args.damping_kd, args.mode_pr)
            out.crc = crc.Crc(out)
            if not args.dry_run:
                pub.Write(out)
            time.sleep(dt)


if __name__ == "__main__":
    main()
