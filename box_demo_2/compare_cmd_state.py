#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比 rt/lowcmd（指令关节角）与 rt/lowstate（实际关节角）的误差。

用法:
    python compare_cmd_state.py [--iface eth0]
"""

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

JOINT_NAMES = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee",
    "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee",
    "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
    "arm_sdk_enable",
]

# 只显示腰部 + 手臂关节
FOCUS_INDICES = list(range(12, 29))


def main():
    parser = argparse.ArgumentParser(description="对比指令关节角与实际关节角")
    parser.add_argument("--iface", default=None, help="DDS 网络接口，如 eth0")
    args = parser.parse_args()

    if args.iface:
        ChannelFactoryInitialize(0, args.iface)
    else:
        ChannelFactoryInitialize(0)

    last_cmd = [None]
    last_state = [None]

    def on_cmd(msg: LowCmd_):
        last_cmd[0] = msg

    def on_state(msg: LowState_):
        last_state[0] = msg

    sub_cmd = ChannelSubscriber("rt/lowcmd", LowCmd_)
    sub_cmd.Init(on_cmd, 10)

    sub_state = ChannelSubscriber("rt/lowstate", LowState_)
    sub_state.Init(on_state, 10)

    print("等待 rt/lowcmd 和 rt/lowstate ...")
    while last_cmd[0] is None or last_state[0] is None:
        time.sleep(0.05)
    print("已连接，开始对比。按 Ctrl+C 退出。\n")

    try:
        while True:
            cmd = last_cmd[0]
            state = last_state[0]
            if cmd is None or state is None:
                time.sleep(0.1)
                continue

            print(f"\n{'='*80}")
            print(f"  {'关节':25s}  {'指令(rad)':>10s}  {'实际(rad)':>10s}  {'误差(rad)':>10s}  {'误差(°)':>8s}")
            print(f"{'-'*80}")

            for i in FOCUS_INDICES:
                name = JOINT_NAMES[i] if i < len(JOINT_NAMES) else f"motor_{i}"
                q_cmd = float(cmd.motor_cmd[i].q)
                q_act = float(state.motor_state[i].q)
                err = q_cmd - q_act
                err_deg = math.degrees(err)
                flag = " <<<" if abs(err_deg) > 3.0 else ""
                print(
                    f"  [{i:2d}] {name:25s}  {q_cmd:+10.4f}  {q_act:+10.4f}  "
                    f"{err:+10.4f}  {err_deg:+7.2f}°{flag}"
                )

            # 汇总
            errors = [float(cmd.motor_cmd[i].q) - float(state.motor_state[i].q) for i in FOCUS_INDICES]
            max_err = max(errors, key=abs)
            max_idx = errors.index(max_err)
            print(f"{'-'*80}")
            print(
                f"  最大误差: {JOINT_NAMES[FOCUS_INDICES[max_idx]]}  "
                f"{max_err:+.4f} rad ({math.degrees(max_err):+.2f}°)"
            )
            print(f"{'='*80}")

            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n退出。")


if __name__ == "__main__":
    main()
