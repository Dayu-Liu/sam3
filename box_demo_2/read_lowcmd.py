#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
订阅 rt/lowcmd 和 rt/arm_sdk，打印收到的 LowCmd_ 消息。

用于观察谁在向这两个话题发送指令（如内置运动控制发 rt/lowcmd，你的脚本发 rt/arm_sdk）。

用法:
    python read_lowcmd.py [--iface eth0]
"""

import argparse
import math
import sys
import time

import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_

# 关节名称（与 read_state.py 一致）
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


def format_cmd(msg: LowCmd_, topic: str) -> str:
    """将 LowCmd_ 格式化为可读字符串"""
    lines = [
        f"\n{'='*60}",
        f"  {topic}  (mode_pr={msg.mode_pr}, mode_machine={msg.mode_machine})",
        f"{'='*60}",
    ]
    # 重点：腰部 12-14、手臂 15-28、arm_sdk_enable 29
    for i in [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]:
        if i < len(msg.motor_cmd):
            mc = msg.motor_cmd[i]
            name = JOINT_NAMES[i] if i < len(JOINT_NAMES) else f"motor_{i}"
            deg = math.degrees(mc.q) if abs(mc.q) < 100 else mc.q
            lines.append(
                f"  [{i:2d}] {name:25s}  q={mc.q:+.4f} ({deg:+.2f}°)  "
                f"kp={mc.kp:.1f} kd={mc.kd:.1f} tau={mc.tau:.2f}"
            )
    lines.append("=" * 60)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="订阅 rt/lowcmd 和 rt/arm_sdk，打印收到的 LowCmd_")
    parser.add_argument("--iface", default=None, help="DDS 网络接口，如 eth0")
    args = parser.parse_args()

    if args.iface:
        ChannelFactoryInitialize(0, args.iface)
    else:
        ChannelFactoryInitialize(0)

    last_lowcmd = [None]
    last_arm_sdk = [None]
    count_lowcmd = [0]
    count_arm_sdk = [0]
    last_print_lowcmd = [0.0]
    last_print_arm_sdk = [0.0]
    THROTTLE = 1.0  # 每个话题最多每秒打印一次

    def on_lowcmd(msg: LowCmd_):
        last_lowcmd[0] = msg
        count_lowcmd[0] += 1

    def on_arm_sdk(msg: LowCmd_):
        last_arm_sdk[0] = msg
        count_arm_sdk[0] += 1

    sub_lowcmd = ChannelSubscriber("rt/lowcmd", LowCmd_)
    sub_arm_sdk = ChannelSubscriber("rt/arm_sdk", LowCmd_)
    sub_lowcmd.Init(on_lowcmd, 10)
    sub_arm_sdk.Init(on_arm_sdk, 10)

    print("订阅 rt/lowcmd 和 rt/arm_sdk，等待消息...")
    print("  rt/lowcmd  : 通常由机器人内置运动控制发布")
    print("  rt/arm_sdk : 通常由你的脚本（zero_position_test 等）发布")
    print("每个话题最多每秒打印一次，按 Ctrl+C 退出。\n")

    try:
        while True:
            now = time.time()
            if last_lowcmd[0] is not None and now - last_print_lowcmd[0] >= THROTTLE:
                print(format_cmd(last_lowcmd[0], f"rt/lowcmd  (共收到 {count_lowcmd[0]} 条)"))
                last_print_lowcmd[0] = now
            if last_arm_sdk[0] is not None and now - last_print_arm_sdk[0] >= THROTTLE:
                print(format_cmd(last_arm_sdk[0], f"rt/arm_sdk (共收到 {count_arm_sdk[0]} 条)"))
                last_print_arm_sdk[0] = now
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n退出。")


if __name__ == "__main__":
    main()
