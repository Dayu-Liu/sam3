#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
忠实复现官方 C++ 例程 g1_arm7_sdk_dds_example 的四阶段上肢控制流程。

阶段:
  1. 双臂复位 (5s) : weight 0→1 (平方律), 关节 → init_pos (零位)
  2. 双臂水平张开 (5s): weight=1, 关节 → target_pos (clamp 步进)
  3. 双臂放下 (5s) : weight=1, 关节 → init_pos (clamp 步进)
  4. 退出控制 (2s) : weight 1→0 (线性)

用法:
    python test_arm_sdk_official.py --iface eth0
"""

import argparse
import ctypes
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

# 须先于 unitree_sdk2py 加载 libddsc
_ddsc = Path(os.environ.get("CYCLONEDDS_HOME", Path.home() / "cyclonedds-0.10-install")) / "lib" / "libddsc.so.0"
if _ddsc.is_file():
    ctypes.CDLL(str(_ddsc), mode=ctypes.RTLD_GLOBAL)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_

# ─── 关节索引 (与官方 SDK G1JointIndex 一致) ───────────────────────
class J:
    WaistYaw          = 12
    WaistRoll         = 13
    WaistPitch        = 14
    LeftShoulderPitch = 15
    LeftShoulderRoll  = 16
    LeftShoulderYaw   = 17
    LeftElbow         = 18
    LeftWristRoll     = 19
    LeftWristPitch    = 20
    LeftWristYaw      = 21
    RightShoulderPitch = 22
    RightShoulderRoll  = 23
    RightShoulderYaw   = 24
    RightElbow         = 25
    RightWristRoll     = 26
    RightWristPitch    = 27
    RightWristYaw      = 28
    kNotUsedJoint      = 29

ARM_JOINTS = [
    J.LeftShoulderPitch, J.LeftShoulderRoll, J.LeftShoulderYaw,
    J.LeftElbow, J.LeftWristRoll, J.LeftWristPitch, J.LeftWristYaw,
    J.RightShoulderPitch, J.RightShoulderRoll, J.RightShoulderYaw,
    J.RightElbow, J.RightWristRoll, J.RightWristPitch, J.RightWristYaw,
    J.WaistYaw, J.WaistRoll, J.WaistPitch,
]

CTRL_DT = 0.02       # 50 Hz
KP = 60.0
KD = 1.5
MAX_JOINT_DELTA = 0.01   # rad/step, 与 C++ 一致

# init_pos: 零位 (双臂下垂)
INIT_POS = [0.0] * len(ARM_JOINTS)

# target_pos: 双臂水平张开 (来自 SDK Python 例程)
_kPi_2 = math.pi / 2
TARGET_POS = [
    0.,  _kPi_2, 0., _kPi_2, 0., 0., 0.,   # left arm
    0., -_kPi_2, 0., _kPi_2, 0., 0., 0.,   # right arm
    0., 0., 0.,                              # waist
]


def make_msg():
    msg = unitree_hg_msg_dds__LowCmd_()
    # 设置上肢关节的 dq/tau (全程不变)
    for j in ARM_JOINTS:
        msg.motor_cmd[j].dq = 0.0
        msg.motor_cmd[j].kp = KP
        msg.motor_cmd[j].kd = KD
        msg.motor_cmd[j].tau = 0.0
    # 使能槽位
    msg.motor_cmd[J.kNotUsedJoint].dq = 0.0
    msg.motor_cmd[J.kNotUsedJoint].kp = 0.0
    msg.motor_cmd[J.kNotUsedJoint].kd = 0.0
    msg.motor_cmd[J.kNotUsedJoint].tau = 0.0
    return msg


def main():
    parser = argparse.ArgumentParser(description="复现官方 arm_sdk 四阶段例程")
    parser.add_argument("--iface", default=None, help="DDS 网络接口")
    args = parser.parse_args()

    print("=" * 55)
    print("  G1 arm_sdk 官方例程复现 (Python)")
    print("  1. 双臂复位 5s  → weight 0→1")
    print("  2. 双臂张开 5s  → target_pos")
    print("  3. 双臂放下 5s  → init_pos")
    print("  4. 退出控制 2s  → weight 1→0")
    print("=" * 55)

    print("\nWARNING: 请确保机器人已悬挂/锁定站立，周围无障碍物。")
    input("按 Enter 继续...")

    # ── 初始化 DDS ────────────────────────────────────────────────
    print("\n[Init] DDS 初始化...")
    if args.iface:
        ChannelFactoryInitialize(0, args.iface)
    else:
        ChannelFactoryInitialize(0)

    publisher = ChannelPublisher("rt/arm_sdk", LowCmd_)
    publisher.Init()
    print("  rt/arm_sdk publisher 就绪")

    msg = make_msg()

    # ═══════════════════════════════════════════════════════════════
    # 阶段 1: 双臂复位 (5s)
    # ═══════════════════════════════════════════════════════════════
    input("\nPress ENTER to init arms (复位到零位, 5s)...")

    print("Initializing arms ...")
    init_duration = 5.0
    num_steps = int(init_duration / CTRL_DT)
    delta_weight = 1.0 / num_steps
    weight = 0.0

    for i in range(num_steps):
        weight += delta_weight
        weight = min(weight, 1.0)

        # C++ 用的是 weight * weight (平方律), 使过渡更平缓
        msg.motor_cmd[J.kNotUsedJoint].q = weight * weight

        for j_idx, motor_idx in enumerate(ARM_JOINTS):
            msg.motor_cmd[motor_idx].q = INIT_POS[j_idx]

        publisher.Write(msg)
        time.sleep(CTRL_DT)

    print("Done! (weight=%.2f)" % (weight * weight))

    # ═══════════════════════════════════════════════════════════════
    # 阶段 2: 双臂水平张开 (5s)
    # ═══════════════════════════════════════════════════════════════
    input("\nPress ENTER to lift arms (水平张开, 5s)...")

    print("Start arm ctrl!")
    period = 5.0
    num_steps = int(period / CTRL_DT)
    current_jpos = [0.0] * len(ARM_JOINTS)

    for i in range(num_steps):
        for j in range(len(ARM_JOINTS)):
            delta = TARGET_POS[j] - current_jpos[j]
            current_jpos[j] += max(-MAX_JOINT_DELTA, min(MAX_JOINT_DELTA, delta))
            msg.motor_cmd[ARM_JOINTS[j]].q = current_jpos[j]

        publisher.Write(msg)
        time.sleep(CTRL_DT)

    print("Done!")

    # ═══════════════════════════════════════════════════════════════
    # 阶段 3: 双臂放下 (5s)
    # ═══════════════════════════════════════════════════════════════
    input("\nPress ENTER to lower arms (放下双臂, 5s)...")

    print("Lowering arms ...")
    num_steps = int(period / CTRL_DT)

    for i in range(num_steps):
        for j in range(len(ARM_JOINTS)):
            delta = INIT_POS[j] - current_jpos[j]
            current_jpos[j] += max(-MAX_JOINT_DELTA, min(MAX_JOINT_DELTA, delta))
            msg.motor_cmd[ARM_JOINTS[j]].q = current_jpos[j]

        publisher.Write(msg)
        time.sleep(CTRL_DT)

    print("Done!")

    # ═══════════════════════════════════════════════════════════════
    # 阶段 4: 退出控制 (2s)
    # ═══════════════════════════════════════════════════════════════
    input("\nPress ENTER to release arm ctrl (退出控制, 2s)...")

    print("Stopping arm ctrl ...")
    stop_time = 2.0
    num_steps = int(stop_time / CTRL_DT)
    delta_weight = 1.0 / num_steps
    weight = 1.0  # 从 1 开始

    for i in range(num_steps):
        weight -= delta_weight
        weight = max(weight, 0.0)

        # C++ 退出阶段用线性 weight (不是平方)
        msg.motor_cmd[J.kNotUsedJoint].q = weight
        publisher.Write(msg)
        time.sleep(CTRL_DT)

    print("Done!")
    print("\n程序正常退出。")


if __name__ == "__main__":
    main()
