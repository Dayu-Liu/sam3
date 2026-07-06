#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
零位执行测试脚本

流程：
  1. 当前位姿 → 零位（与 dual_arm_target_reach.py 中 ZERO_Q 一致）
  2. 保持零位
  3. 按 Enter 后：零位 → 恢复为执行前的原始状态
  4. 逐步释放控制权

用法:
    python zero_position_test.py [--iface eth0]
    python zero_position_test.py --test-waist   # 测试腰部运动
    python zero_position_test.py --test-arm-only   # 仅手臂到零位，不发腰部指令
"""

import os
import sys
import time
import threading

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber,
)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread

_crc = CRC()

from dual_arm_target_reach import ARM_JOINTS, ZERO_Q, G1Joint, DualArmController

# 仅手臂关节（左7 + 右7），不含腰部
ARM_JOINTS_NO_WAIST = [
    G1Joint.LeftShoulderPitch,  G1Joint.LeftShoulderRoll,
    G1Joint.LeftShoulderYaw,    G1Joint.LeftElbow,
    G1Joint.LeftWristRoll,      G1Joint.LeftWristPitch,
    G1Joint.LeftWristYaw,
    G1Joint.RightShoulderPitch, G1Joint.RightShoulderRoll,
    G1Joint.RightShoulderYaw,   G1Joint.RightElbow,
    G1Joint.RightWristRoll,     G1Joint.RightWristPitch,
    G1Joint.RightWristYaw,
]
# 零位手臂姿态（与 dual_arm_target_reach ZERO_Q 的手臂部分一致）
ZERO_Q_ARM_ONLY = ZERO_Q[:14]  # 左肩横滚 30°，右肩横滚 -30°，其余 0

# 控制参数（与 dual_arm_target_reach 一致）
KP = 60.0
KD = 1.5
KP_WAIST = DualArmController.KP_WAIST
KD_WAIST = DualArmController.KD_WAIST
DT = 0.02  # 50 Hz
DUR_TO_ZERO = 3.0
DUR_TO_ORIGINAL = 3.0
DUR_RELEASE = 2.0

# 腰部测试目标：双臂保持站立姿态，腰部 waist_pitch 前倾约 5.7°
# ARM_JOINTS 顺序：左7 + 右7 + 腰3
WAIST_TEST_TARGET = [
    # 左臂 (站立姿态)
    0.2908, 0.2244, -0.0361, 0.9847, 0.1591, 0.0325, 0.0093,
    # 右臂
    0.2885, -0.2163, 0.0412, 0.9852, -0.9, 0.0302, -0.0034,
    # 腰部: waist_yaw, waist_roll, waist_pitch (前倾 0.1 rad ≈ 5.7°)
    0, 0, 0.1,
]

# WAIST_TEST_TARGET = [
#     # 左臂 (站立姿态)
#     0.2908, 0.2244, -0.0361, 0.9847, 0.1591, 0.0325, 0.0093,
#     # 右臂
#     0.2885, -0.2163, 0.0412, 0.9852, -0.1206, 0.0302, -0.0034,
#     # 腰部: waist_yaw, waist_roll, waist_pitch (前倾 0.1 rad ≈ 5.7°)
#     0, 0, 0.1,
# ]

def main():
    import argparse
    parser = argparse.ArgumentParser(description="零位执行测试")
    parser.add_argument("--iface", default=None, help="DDS 网络接口，如 eth0")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--test-waist", action="store_true",
                       help="测试腰部运动：双臂保持站立姿态，waist_pitch 前倾约 5.7°")
    group.add_argument("--test-arm-only", action="store_true",
                       help="仅手臂到零位（左/右肩横滚 30°），不发腰部指令到 rt/arm_sdk")
    args = parser.parse_args()

    if args.iface:
        ChannelFactoryInitialize(0, args.iface)
    else:
        ChannelFactoryInitialize(0)

    # DDS
    publisher = ChannelPublisher("rt/arm_sdk", LowCmd_)
    publisher.Init()
    low_state = [None]

    def on_state(msg: LowState_):
        low_state[0] = msg

    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(on_state, 10)

    print("等待机器人状态...")
    while low_state[0] is None:
        time.sleep(0.05)
    print("已收到状态。\n")

    arm_only = args.test_arm_only
    joints_to_use = ARM_JOINTS_NO_WAIST if arm_only else ARM_JOINTS

    def read_q():
        s = low_state[0]
        return [float(s.motor_state[int(j)].q) for j in joints_to_use]

    def lerp(q_from, q_to, ratio):
        r = float(np.clip(ratio, 0.0, 1.0))
        return [q_from[i] * (1.0 - r) + q_to[i] * r for i in range(len(q_from))]

    def publish(q_cmd, sdk_weight=1.0):
        low_cmd = unitree_hg_msg_dds__LowCmd_()
        low_cmd.motor_cmd[int(G1Joint.ArmSdkEnable)].q = float(sdk_weight)
        for i, joint in enumerate(joints_to_use):
            mc = low_cmd.motor_cmd[int(joint)]
            mc.tau = 0.0
            mc.q = float(q_cmd[i])
            mc.dq = 0.0
            if arm_only:
                mc.kp = KP
                mc.kd = KD
            else:
                mc.kp = KP_WAIST if i >= 14 else KP
                mc.kd = KD_WAIST if i >= 14 else KD
        low_cmd.crc = _crc.Crc(low_cmd)
        publisher.Write(low_cmd)

    # 保存原始状态
    q_original = read_q()
    print("已记录当前位姿（原始状态）")

    # 目标位姿：零位 或 腰部测试目标 或 仅手臂零位
    if args.test_waist:
        q_target = WAIST_TEST_TARGET
        print("模式：腰部运动测试（双臂站立姿态，waist_pitch 前倾 0.1 rad ≈ 5.7°）\n")
    elif arm_only:
        q_target = ZERO_Q_ARM_ONLY
        print("模式：仅手臂到零位（左/右肩横滚 30°，不发腰部指令）\n")
    else:
        q_target = ZERO_Q

    # 状态：0=到目标, 1=保持, 2=回原始, 3=释放, 4=完成
    stage = [0]
    stage_t = [0.0]
    return_requested = threading.Event()

    def control_loop():
        if stage[0] == 4:
            return

        if stage[0] == 0:  # 到目标
            ratio = stage_t[0] / DUR_TO_ZERO
            q_cmd = lerp(q_original, q_target, ratio)
            publish(q_cmd, 1.0)
            stage_t[0] += DT
            if stage_t[0] >= DUR_TO_ZERO:
                stage[0] = 1
                stage_t[0] = 0.0
                label = "腰部测试目标" if args.test_waist else ("手臂零位" if arm_only else "零位")
                print(f"[  3.0s] 已到达{label}，保持中。按 Enter 恢复原始状态...")

        elif stage[0] == 1:  # 保持目标，等待 Enter
            publish(q_target, 1.0)
            if return_requested.is_set():
                stage[0] = 2
                stage_t[0] = 0.0
                print("\n[  0.0s] 目标 → 原始状态")

        elif stage[0] == 2:  # 回原始
            ratio = stage_t[0] / DUR_TO_ORIGINAL
            q_cmd = lerp(q_target, q_original, ratio)
            publish(q_cmd, 1.0)
            stage_t[0] += DT
            if stage_t[0] >= DUR_TO_ORIGINAL:
                stage[0] = 3
                stage_t[0] = 0.0
                print("[  3.0s] 已恢复原始状态，释放控制权...")

        elif stage[0] == 3:  # 释放
            ratio = stage_t[0] / DUR_RELEASE
            publish(q_original, 1.0 - ratio)
            stage_t[0] += DT
            if stage_t[0] >= DUR_RELEASE:
                stage[0] = 4
                publish(q_original, 0.0)
                print("[  2.0s] 释放完成。")

    label = "腰部测试目标" if args.test_waist else ("手臂零位" if arm_only else "零位")
    print(f"[  0.0s] 原始状态 → {label}")
    ctrl_thread = RecurrentThread(interval=DT, target=control_loop, name="zero_test_ctrl")
    ctrl_thread.Start()

    # 主线程：等待 Enter
    try:
        input()
        return_requested.set()
    except (EOFError, KeyboardInterrupt):
        return_requested.set()

    # 等待控制线程完成
    while stage[0] < 4:
        time.sleep(0.2)

    print("\n测试完成。")


if __name__ == "__main__":
    main()
