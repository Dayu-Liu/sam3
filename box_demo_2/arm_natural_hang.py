#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""双臂自然下垂保持：通过 rt/arm_sdk 持续发布 RL_LOWER_HANDOFF_Q。"""

from __future__ import annotations

import os
import sys
import math
import threading
import time
from typing import Callable, Optional

import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dual_arm_target_reach import (  # noqa: E402
    ARM_JOINTS,
    DualArmController,
    G1Joint,
    RL_LOWER_HANDOFF_Q,
)
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber  # noqa: E402
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_  # noqa: E402
from unitree_sdk2py.utils.crc import CRC  # noqa: E402

DT = 0.02
DEFAULT_MOVE_DURATION = 3.0
RELEASE_BLEND_DURATION = 2.5
# merge_lowcmd_arm_sdk 默认 arm_stale_s=0.25；任何 handoff 空窗超过此值 RL 会抢回上半身
ARM_SDK_BURST_FRAMES = 30
_crc = CRC()


def smooth_ratio(ratio: float) -> float:
    r = float(np.clip(ratio, 0.0, 1.0))
    return 0.5 * (1.0 - math.cos(math.pi * r))


def read_policy_q(rl_cmd: LowCmd_) -> list[float]:
    return [float(rl_cmd.motor_cmd[int(j)].q) for j in ARM_JOINTS]


def lerp_q(q_from: list, q_to: list, ratio: float) -> list:
    r = float(np.clip(ratio, 0.0, 1.0))
    return [q_from[i] * (1.0 - r) + q_to[i] * r for i in range(len(q_from))]


def publish_arm_sdk(
    publisher: ChannelPublisher,
    low_state: LowState_,
    q_cmd: list,
    sdk_weight: float = 1.0,
) -> None:
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_

    low_cmd = unitree_hg_msg_dds__LowCmd_()
    low_cmd.mode_machine = low_state.mode_machine
    low_cmd.mode_pr = low_state.mode_pr

    en = low_cmd.motor_cmd[int(G1Joint.ArmSdkEnable)]
    en.mode = 1
    en.q = float(sdk_weight)
    en.dq = en.kp = en.kd = en.tau = 0.0

    for i, joint in enumerate(ARM_JOINTS):
        mc = low_cmd.motor_cmd[int(joint)]
        mc.mode = 1
        mc.tau = 0.0
        mc.q = float(q_cmd[i])
        mc.dq = 0.0
        if i >= 14:
            mc.kp = DualArmController.KP_WAIST
            mc.kd = DualArmController.KD_WAIST
        elif joint in (G1Joint.LeftShoulderPitch, G1Joint.RightShoulderPitch):
            mc.kp, mc.kd = DualArmController.KP_SHOULDER_PITCH, DualArmController.KD_SHOULDER_PITCH
        elif joint in (G1Joint.LeftShoulderRoll, G1Joint.RightShoulderRoll):
            mc.kp, mc.kd = DualArmController.KP_SHOULDER_ROLL, DualArmController.KD_SHOULDER_ROLL
        elif joint in (G1Joint.LeftShoulderYaw, G1Joint.RightShoulderYaw):
            mc.kp, mc.kd = DualArmController.KP_SHOULDER_YAW, DualArmController.KD_SHOULDER_YAW
        elif joint in (G1Joint.LeftElbow, G1Joint.RightElbow):
            mc.kp, mc.kd = DualArmController.KP_ELBOW, DualArmController.KD_ELBOW
        elif joint in (G1Joint.LeftWristRoll, G1Joint.RightWristRoll):
            mc.kp, mc.kd = DualArmController.KP_WRIST_ROLL, DualArmController.KD_WRIST_ROLL
        elif joint in (G1Joint.LeftWristPitch, G1Joint.RightWristPitch):
            mc.kp, mc.kd = DualArmController.KP_WRIST_PITCH, DualArmController.KD_WRIST_PITCH
        elif joint in (G1Joint.LeftWristYaw, G1Joint.RightWristYaw):
            mc.kp, mc.kd = DualArmController.KP_WRIST_YAW, DualArmController.KD_WRIST_YAW
        else:
            mc.kp, mc.kd = 60.0, 1.5

    low_cmd.crc = _crc.Crc(low_cmd)
    publisher.Write(low_cmd)


class ArmNaturalHangKeeper:
    """后台 50Hz 保持双臂自然下垂，直至 stop()。"""

    def __init__(self):
        self._pub: Optional[ChannelPublisher] = None
        self.low_state: Optional[LowState_] = None
        self._ready = threading.Event()
        self._waist_yaw = 0.0
        self._pass_waist_from_state = False
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def _ensure_dds(self) -> None:
        if self._pub is not None:
            return
        self._pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self._pub.Init()
        sub = ChannelSubscriber("rt/lowstate", LowState_)
        sub.Init(self._on_state, 10)
        self._ready.wait(timeout=2.0)

    def _on_state(self, msg: LowState_):
        self.low_state = msg
        if not self._ready.is_set():
            self._ready.set()

    def build_q(self) -> list[float]:
        q = list(RL_LOWER_HANDOFF_Q)
        if self._pass_waist_from_state and self.low_state is not None:
            q[14] = float(self.low_state.motor_state[int(G1Joint.WaistYaw)].q)
            q[15] = float(self.low_state.motor_state[int(G1Joint.WaistRoll)].q)
            q[16] = float(self.low_state.motor_state[int(G1Joint.WaistPitch)].q)
        else:
            q[14] = self._waist_yaw
        return q

    def set_waist_yaw(self, yaw: float) -> None:
        self._waist_yaw = float(yaw)

    def sync_waist_from_robot(self) -> None:
        self._ensure_dds()
        time.sleep(0.05)
        if self.low_state is not None:
            self._waist_yaw = float(self.low_state.motor_state[int(G1Joint.WaistYaw)].q)

    def set_pass_waist_from_state(self, enabled: bool) -> None:
        self._pass_waist_from_state = enabled

    def publish_once(self, q_cmd: Optional[list] = None) -> None:
        if self._pub is None or self.low_state is None:
            return
        publish_arm_sdk(self._pub, self.low_state, q_cmd or self.build_q(), sdk_weight=1.0)

    def _loop(self) -> None:
        while self._running:
            t0 = time.time()
            self.publish_once()
            sleep_s = DT - (time.time() - t0)
            if sleep_s > 0:
                time.sleep(sleep_s)

    def is_running(self) -> bool:
        return self._running and self._thread is not None and self._thread.is_alive()

    def _burst_publish(self, q_cmd: list, *, frames: int = ARM_SDK_BURST_FRAMES) -> None:
        if self._pub is None or self.low_state is None:
            return
        for _ in range(frames):
            publish_arm_sdk(self._pub, self.low_state, q_cmd, sdk_weight=1.0)
            time.sleep(DT)

    def start_keepalive(self) -> None:
        self._ensure_dds()
        if self.is_running():
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="arm_hang_keep")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

    def move_to_hang(self, duration: float = DEFAULT_MOVE_DURATION, start_keepalive: bool = True) -> None:
        self._ensure_dds()
        assert self.low_state is not None
        q_start = [float(self.low_state.motor_state[int(j)].q) for j in ARM_JOINTS]
        target_q = self.build_q()
        print("[arm_hang] 双臂 → 自然下垂")

        t = 0.0
        while t < duration:
            self.publish_once(lerp_q(q_start, target_q, t / duration))
            time.sleep(DT)
            t += DT
        self.publish_once(target_q)

        if start_keepalive:
            self.start_keepalive()
            print("[arm_hang] 已启动自然下垂保持（50Hz，直至阶段1前交权）")

    def move_and_start(self, duration: float = DEFAULT_MOVE_DURATION) -> None:
        self.sync_waist_from_robot()
        self.move_to_hang(duration=duration, start_keepalive=True)

    def ensure_keepalive(self, q_cmd: Optional[list] = None) -> None:
        """保证 50Hz arm_sdk 不断流。merge 超过 ~250ms 未收到 arm_sdk 会回退 RL 上半身。"""
        if self.is_running():
            return
        self.stop()
        self.sync_waist_from_robot()
        self.set_pass_waist_from_state(False)
        self._ensure_dds()
        if self.low_state is None:
            print("[arm_hang] 警告: ensure_keepalive 时 low_state 未就绪")
            return
        hold_q = list(q_cmd) if q_cmd is not None else self.build_q()
        self._burst_publish(hold_q)
        self.start_keepalive()
        print("[arm_hang] 已恢复自然下垂保持（50Hz）")

    def resume_keepalive(self) -> None:
        """已在自然下垂位姿时直接启动 50Hz 保持（无插值、不淡出）。"""
        self.ensure_keepalive()

    def handoff_from_grasp(self, q_cmd: Optional[list] = None) -> None:
        """抓取结束：与 DualArmController 重叠发布后单独接管 arm_sdk。"""
        self.stop()
        self.sync_waist_from_robot()
        self.set_pass_waist_from_state(False)
        self._ensure_dds()
        if self.low_state is None:
            print("[arm_hang] 警告: handoff 时 low_state 未就绪")
            return
        if q_cmd is None:
            hold_q = [
                float(self.low_state.motor_state[int(j)].q) for j in ARM_JOINTS
            ]
        else:
            hold_q = list(q_cmd)
        print("[arm_hang] 接管 arm_sdk → 自然下垂保持")
        self._burst_publish(hold_q)
        self.start_keepalive()
        print("[arm_hang] 已启动自然下垂保持（50Hz，直至阶段1前交权）")

    def resume_after_grasp(self, duration: float = 1.0) -> None:
        if self.is_running():
            return
        self.sync_waist_from_robot()
        self.set_pass_waist_from_state(False)
        self.move_to_hang(duration=duration, start_keepalive=True)

    def release_to_policy(
        self,
        duration: float = RELEASE_BLEND_DURATION,
        q_start: Optional[list] = None,
    ) -> None:
        """平滑插值到 policy 上半身目标后再释放 arm_sdk（避免 merge 切换时突变）。"""
        self.stop()
        self._ensure_dds()
        if self._pub is None or self.low_state is None:
            return

        if q_start is None:
            q_start = [
                float(self.low_state.motor_state[int(j)].q) for j in ARM_JOINTS
            ]
        else:
            q_start = list(q_start)

        for _ in range(25):
            publish_arm_sdk(self._pub, self.low_state, q_start, sdk_weight=1.0)
            time.sleep(DT)

        rl_cmd: Optional[LowCmd_] = None
        rl_ready = threading.Event()

        def _on_rl(msg: LowCmd_):
            nonlocal rl_cmd
            rl_cmd = msg
            if not rl_ready.is_set():
                rl_ready.set()

        rl_sub = ChannelSubscriber("rt/lowcmd_rl", LowCmd_)
        rl_sub.Init(_on_rl, 10)
        deadline = time.time() + 2.0
        while not rl_ready.is_set() and time.time() < deadline:
            publish_arm_sdk(self._pub, self.low_state, q_start, sdk_weight=1.0)
            time.sleep(DT)
        if not rl_ready.is_set():
            print("[arm_hang] 警告: 未收到 rt/lowcmd_rl，将仅淡出 arm_sdk 权重")

        print(f"[arm_hang] 平滑过渡到 policy 位姿（{duration:.1f}s）...")
        t = 0.0
        while t < duration:
            r = smooth_ratio(t / duration)
            if rl_cmd is not None:
                q_policy = read_policy_q(rl_cmd)
                q = lerp_q(q_start, q_policy, r)
            else:
                q = q_start
            publish_arm_sdk(self._pub, self.low_state, q, sdk_weight=1.0)
            time.sleep(DT)
            t += DT

        if rl_cmd is not None:
            q_final = read_policy_q(rl_cmd)
        else:
            q_final = q_start
        publish_arm_sdk(self._pub, self.low_state, q_final, sdk_weight=1.0)
        time.sleep(0.1)
        publish_arm_sdk(self._pub, self.low_state, q_final, sdk_weight=0.0)
        print("[arm_hang] 已平滑交还给 policy。")


def read_arm_q_from_state(keeper: ArmNaturalHangKeeper) -> list[float]:
    """从 lowstate 读取当前腰+双臂关节角（ARM_JOINTS 顺序）。"""
    keeper._ensure_dds()
    time.sleep(0.05)
    if keeper.low_state is not None:
        return [
            float(keeper.low_state.motor_state[int(j)].q) for j in ARM_JOINTS
        ]
    return keeper.build_q()


def make_release_callback(keeper: ArmNaturalHangKeeper) -> Callable[[], None]:
    return keeper.stop


def make_handoff_callback(
    keeper: ArmNaturalHangKeeper,
) -> Callable[[Optional[list]], None]:
    """抓取结束：keeper 接管 arm_sdk（位姿与 controller 末帧一致，避免空窗/双写冲突）。"""
    def _handoff(q_cmd: Optional[list] = None) -> None:
        keeper.handoff_from_grasp(q_cmd)
    return _handoff
