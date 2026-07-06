#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双臂三阶段关节位姿测试

参照 box_demo_2 / dual_arm_target_reach.py，通过 rt/arm_sdk 发送左右臂（及可选腰部）关节指令。
共分 3 个阶段，每阶段到达目标位姿后按 Enter 进入下一阶段；结束后可恢复初始位姿并释放控制权。

用法:
    python arm_three_pose_test.py [--iface enP8p1s0]
    python arm_three_pose_test.py --arm-only   # 仅 14 个手臂关节，不含腰

注意:
    - 需与 merge_lowcmd_arm_sdk + run_pipeline 配合（与 box_demo_main 相同）
    - 请在下方 POSE_1 / POSE_2 / POSE_3 中填入自己的目标关节角 (rad)
"""

from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time
from enum import IntEnum, auto
from pathlib import Path

import numpy as np

# 引用 box_demo_2 目录下的模块
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

_ddsc = Path(os.environ.get("CYCLONEDDS_HOME", Path.home() / "cyclonedds-0.10-install")) / "lib" / "libddsc.so.0"
if _ddsc.is_file():
    ctypes.CDLL(str(_ddsc), mode=ctypes.RTLD_GLOBAL)

from unitree_sdk2py.core.channel import (  # noqa: E402
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_  # noqa: E402
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_  # noqa: E402
from unitree_sdk2py.utils.crc import CRC  # noqa: E402
from unitree_sdk2py.utils.thread import RecurrentThread  # noqa: E402

from dual_arm_target_reach import ARM_JOINTS, G1Joint, DualArmController, ZERO_Q  # noqa: E402

_crc = CRC()
_ctrl = DualArmController  # 复用 KP/KD 常量

ARM_JOINTS_ONLY = ARM_JOINTS[:14]
N_ARM = 14
N_FULL = len(ARM_JOINTS)
DT = 0.02
MOVE_DURATION = 3.0
RESTORE_DURATION = 3.0
RELEASE_DURATION = 2.0


# ─────────────────────────────────────────────────────────────────
# 三阶段目标位姿 — 请在此处自行修改 (单位: rad)
# ─────────────────────────────────────────────────────────────────
# 完整 17 维顺序 (左7 + 右7 + 腰3):
#   左: ShoulderPitch, ShoulderRoll, ShoulderYaw, Elbow, WristRoll, WristPitch, WristYaw
#   右: 同上
#   腰: WaistYaw, WaistRoll, WaistPitch

# POSE_1：与 dual_arm_target_reach.py 零位（双手叉腰预备）一致
POSE_1 = list(ZERO_Q)

POSE_2 = [
    # ── 左臂 7 ──
    -0.8, 0.0, 0.1, 0.5, 0.0, 0.0, 0.0,
    # ── 右臂 7 ──
    -0.8, 0.0, -0.1, 0.5, 0.0, 0.0, 0.0,
    # ── 腰部 3 ──
    0.0, 0.0, 0.0,
]

POSE_3 = [
    # ── 左臂 7 ──
    -0.8, 0.0, 0.1, 0.5, 0.0, 0.0, 0.0,
    # ── 右臂 7 ──
    -0.8, 0.0, -0.1, 0.5, 0.0, 0.0, 0.0,
    # ── 腰部 3 ──
    0.0, 0.0, 0.0,
]

TARGET_POSES = [POSE_1, POSE_2, POSE_3]


class Stage(IntEnum):
    MOVING = auto()
    HOLD = auto()
    RESTORE = auto()
    RELEASE = auto()
    DONE = auto()


def _validate_pose(name: str, pose: list, n: int) -> None:
    if len(pose) != n:
        raise ValueError(f"{name} 长度应为 {n}，当前为 {len(pose)}")


def _slice_pose(full_pose: list, arm_only: bool) -> list:
    return full_pose[:N_ARM] if arm_only else full_pose


class ArmThreePoseTest:
    def __init__(self, arm_only: bool):
        self.arm_only = arm_only
        self.joints = ARM_JOINTS_ONLY if arm_only else ARM_JOINTS
        self.n = len(self.joints)

        for i, pose in enumerate(TARGET_POSES, start=1):
            _validate_pose(f"POSE_{i}", pose, N_FULL)

        self.poses = [_slice_pose(p, arm_only) for p in TARGET_POSES]

        self._publisher = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self._publisher.Init()
        self._low_state: LowState_ | None = None

        sub = ChannelSubscriber("rt/lowstate", LowState_)
        sub.Init(self._on_low_state, 10)

        self._stage = Stage.HOLD
        self._phase_idx = 0
        self._stage_t = 0.0
        self._q_original: list[float] = []
        self._q_stage_start: list[float] | None = None
        self._q_target: list[float] = []
        self._restore_on_exit = True

    def _on_low_state(self, msg: LowState_):
        self._low_state = msg

    def wait_state(self) -> None:
        print("等待机器人状态 (rt/lowstate)...")
        while self._low_state is None:
            time.sleep(0.05)
        print("已收到状态。\n")

    def _read_q(self) -> list[float]:
        assert self._low_state is not None
        return [float(self._low_state.motor_state[int(j)].q) for j in self.joints]

    @staticmethod
    def _lerp(q_from: list, q_to: list, ratio: float) -> list:
        r = float(np.clip(ratio, 0.0, 1.0))
        return [q_from[i] * (1.0 - r) + q_to[i] * r for i in range(len(q_from))]

    def _publish(self, q_cmd: list, sdk_weight: float = 1.0) -> None:
        low_cmd = unitree_hg_msg_dds__LowCmd_()
        if self._low_state is not None:
            low_cmd.mode_machine = self._low_state.mode_machine
            low_cmd.mode_pr = self._low_state.mode_pr

        en = low_cmd.motor_cmd[int(G1Joint.ArmSdkEnable)]
        en.mode = 1
        en.q = float(sdk_weight)
        en.dq = en.kp = en.kd = en.tau = 0.0

        for i, joint in enumerate(self.joints):
            mc = low_cmd.motor_cmd[int(joint)]
            mc.mode = 1
            mc.tau = 0.0
            mc.q = float(q_cmd[i])
            mc.dq = 0.0
            if not self.arm_only and i >= 14:
                mc.kp = _ctrl.KP_WAIST
                mc.kd = _ctrl.KD_WAIST
            elif joint in (G1Joint.LeftShoulderPitch, G1Joint.RightShoulderPitch):
                mc.kp, mc.kd = _ctrl.KP_SHOULDER_PITCH, _ctrl.KD_SHOULDER_PITCH
            elif joint in (G1Joint.LeftShoulderRoll, G1Joint.RightShoulderRoll):
                mc.kp, mc.kd = _ctrl.KP_SHOULDER_ROLL, _ctrl.KD_SHOULDER_ROLL
            elif joint in (G1Joint.LeftShoulderYaw, G1Joint.RightShoulderYaw):
                mc.kp, mc.kd = _ctrl.KP_SHOULDER_YAW, _ctrl.KD_SHOULDER_YAW
            elif joint in (G1Joint.LeftElbow, G1Joint.RightElbow):
                mc.kp, mc.kd = _ctrl.KP_ELBOW, _ctrl.KD_ELBOW
            elif joint in (G1Joint.LeftWristRoll, G1Joint.RightWristRoll):
                mc.kp, mc.kd = _ctrl.KP_WRIST_ROLL, _ctrl.KD_WRIST_ROLL
            elif joint in (G1Joint.LeftWristPitch, G1Joint.RightWristPitch):
                mc.kp, mc.kd = _ctrl.KP_WRIST_PITCH, _ctrl.KD_WRIST_PITCH
            elif joint in (G1Joint.LeftWristYaw, G1Joint.RightWristYaw):
                mc.kp, mc.kd = _ctrl.KP_WRIST_YAW, _ctrl.KD_WRIST_YAW
            else:
                mc.kp, mc.kd = 60.0, 1.5

        low_cmd.crc = _crc.Crc(low_cmd)
        self._publisher.Write(low_cmd)

    def _begin_move(self, target: list[float]) -> None:
        self._q_target = list(target)
        self._q_stage_start = self._read_q()
        self._stage_t = 0.0
        self._stage = Stage.MOVING

    def _control_loop(self) -> None:
        if self._stage == Stage.DONE:
            return

        if self._stage == Stage.HOLD:
            # 固定目标位姿 + 全权重，避免等待 Enter 期间跟随 read_q() 导致上半身慢慢软下去
            self._publish(self._q_target, sdk_weight=1.0)
            return

        if self._stage == Stage.MOVING:
            ratio = self._stage_t / MOVE_DURATION
            q_cmd = self._lerp(self._q_stage_start or self._read_q(), self._q_target, ratio)
            self._publish(q_cmd, sdk_weight=1.0)
            self._stage_t += DT
            if self._stage_t >= MOVE_DURATION:
                self._stage = Stage.HOLD
                self._stage_t = 0.0
            return

        if self._stage == Stage.RESTORE:
            ratio = self._stage_t / RESTORE_DURATION
            q_cmd = self._lerp(self._q_target, self._q_original, ratio)
            self._publish(q_cmd, sdk_weight=1.0)
            self._stage_t += DT
            if self._stage_t >= RESTORE_DURATION:
                self._stage = Stage.RELEASE
                self._stage_t = 0.0
                print("已恢复初始位姿，释放控制权...")
            return

        if self._stage == Stage.RELEASE:
            ratio = self._stage_t / RELEASE_DURATION
            self._publish(self._q_original, sdk_weight=1.0 - ratio)
            self._stage_t += DT
            if self._stage_t >= RELEASE_DURATION:
                self._publish(self._q_original, sdk_weight=0.0)
                self._stage = Stage.DONE
                print("释放完成。")

    def _wait_enter(self, prompt: str) -> None:
        print(prompt)
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            print()
            self._restore_on_exit = True
            self._phase_idx = 3
            self._stage = Stage.RESTORE
            self._stage_t = 0.0
            return

    def run(self) -> None:
        self.wait_state()
        self._q_original = self._read_q()
        self._q_target = list(self._q_original)
        print("已记录当前关节角作为初始位姿。")
        print(f"控制关节数: {self.n} ({'仅手臂' if self.arm_only else '手臂+腰'})")
        print("控制线程已启动，上半身将保持当前站立姿态，等待 Enter...\n")

        thread = RecurrentThread(interval=DT, target=self._control_loop, name="arm_three_pose")
        thread.Start()

        # 启动瞬间先连续发几帧，避免 merge 因 arm_sdk 断流切回 RL 导致上半身发软
        for _ in range(25):
            self._publish(self._q_target, sdk_weight=1.0)
            time.sleep(DT)

        try:
            for phase in range(3):
                self._phase_idx = phase

                self._wait_enter(f"\n>>> 按 Enter 开始阶段 {phase + 1}/3 <<<")
                if self._stage == Stage.RESTORE:
                    break

                print(f"[阶段 {phase + 1}/3] 运动中 ({MOVE_DURATION:.1f}s)...")
                self._begin_move(self.poses[phase])

                while self._stage == Stage.MOVING:
                    time.sleep(0.05)

                print(f"[阶段 {phase + 1}/3] 已到达目标位姿，保持中。")

                if phase < 2:
                    # 保持 HOLD，控制线程持续发目标位姿；主线程等待 Enter
                    self._wait_enter(f">>> 按 Enter 进入阶段 {phase + 2}/3 <<<")
                    if self._stage == Stage.RESTORE:
                        break
                else:
                    self._wait_enter(">>> 三阶段完成。按 Enter 恢复初始位姿并退出 <<<")
                    if self._stage != Stage.RESTORE:
                        self._stage = Stage.RESTORE
                        self._stage_t = 0.0

            while self._stage != Stage.DONE:
                time.sleep(0.05)

        except KeyboardInterrupt:
            print("\n中断，恢复初始位姿...")
            self._stage = Stage.RESTORE
            self._stage_t = 0.0
            while self._stage != Stage.DONE:
                time.sleep(0.05)

        print("\n测试结束。")


def main() -> int:
    parser = argparse.ArgumentParser(description="双臂三阶段关节位姿测试")
    parser.add_argument("--iface", default=None, help="DDS 网络接口，如 enP8p1s0")
    parser.add_argument(
        "--arm-only",
        action="store_true",
        help="仅控制左右臂 14 关节，不含腰部",
    )
    args = parser.parse_args()

    if args.iface:
        ChannelFactoryInitialize(0, args.iface)
    else:
        ChannelFactoryInitialize(0)

    ArmThreePoseTest(arm_only=args.arm_only).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
