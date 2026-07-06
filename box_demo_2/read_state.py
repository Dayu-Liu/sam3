#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取宇树 G1 机器人全身关节角（站立状态 / 零位前）。

通过 DDS 订阅 rt/lowstate 获取 LowState_，motor_state[i].q 为第 i 个关节角度 (rad)。

用法:
    python read_state.py [--iface eth0] [--once] [--no-csv]
    --once: 只读一次后退出；默认持续打印
    --no-csv: 不写入 CSV；默认每次读取后追加到 state.csv
"""

import argparse
import csv
import math
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

from dual_arm_target_reach import ArmKinematics

# G1 关节索引与名称 (motor_state 索引 0-29 为主 body，共 30 个)
JOINT_NAMES = [
    # 左腿 0-5
    "left_hip_pitch",
    "left_hip_roll",
    "left_hip_yaw",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
    # 右腿 6-11
    "right_hip_pitch",
    "right_hip_roll",
    "right_hip_yaw",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
    # 腰部 12-14
    "waist_yaw",
    "waist_roll",
    "waist_pitch",
    # 左臂 15-21
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "left_wrist_roll",
    "left_wrist_pitch",
    "left_wrist_yaw",
    # 右臂 22-28
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
    # 29
    "arm_sdk_enable",
]

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state.csv")


def save_state_to_csv(s, need_header: list) -> None:
    """将当前 state 追加写入 CSV 文件（转置格式：第一列为变量名，后续列为各时刻数据）"""
    n = min(30, len(s.motor_state))
    # 正运动学：左右臂 7 关节角 → 末端位姿（motor_state 15-21 左臂，22-28 右臂）
    ik = ArmKinematics()
    left_q = [float(s.motor_state[i].q) for i in range(15, 22)]
    right_q = [float(s.motor_state[i].q) for i in range(22, 29)]
    left_fk, _ = ik.forward_kinematics(left_q, left=True)
    right_fk, _ = ik.forward_kinematics(right_q, left=False)

    names = ["timestamp"] + [
        f"{JOINT_NAMES[i] if i < len(JOINT_NAMES) else f'motor_{i}'}_q"
        for i in range(n)
    ] + [
        f"{JOINT_NAMES[i] if i < len(JOINT_NAMES) else f'motor_{i}'}_dq"
        for i in range(n)
    ] + [
        "left_fk_x", "left_fk_y", "left_fk_z",
        "right_fk_x", "right_fk_y", "right_fk_z",
    ]
    new_col = [
        datetime.now().isoformat(),
        *[float(s.motor_state[i].q) for i in range(n)],
        *[float(s.motor_state[i].dq) for i in range(n)],
        float(left_fk[0]), float(left_fk[1]), float(left_fk[2]),
        float(right_fk[0]), float(right_fk[1]), float(right_fk[2]),
    ]
    if need_header[0]:
        # 新文件：每行一个变量名，第一列是表头，第二列是第一个数据
        rows = [[names[i], new_col[i]] for i in range(len(names))]
        need_header[0] = False
    else:
        # 追加：读入已有数据，给每行加一列
        with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
        if len(rows) == len(names) and len(rows[0]) > 0:
            for i in range(len(names)):
                rows[i].append(str(new_col[i]))
        else:
            # 文件格式异常，按新文件重写
            rows = [[names[i], new_col[i]] for i in range(len(names))]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="读取 G1 全身关节角")
    parser.add_argument("--iface", default=None, help="DDS 网络接口，如 eth0")
    parser.add_argument("--once", action="store_true", help="只读一次后退出")
    parser.add_argument("--no-csv", action="store_true", help="不写入 CSV 文件")
    args = parser.parse_args()

    state_holder = [None]

    def on_state(msg: LowState_):
        state_holder[0] = msg

    if args.iface:
        ChannelFactoryInitialize(0, args.iface)
    else:
        ChannelFactoryInitialize(0)

    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(on_state, 10)

    print("等待机器人状态 (rt/lowstate)...")
    timeout = 10.0
    deadline = time.time() + timeout
    while state_holder[0] is None and time.time() < deadline:
        time.sleep(0.05)

    if state_holder[0] is None:
        print("超时：未收到状态，请确认机器人已上电且 DDS 网络正常。")
        sys.exit(1)

    csv_need_header = [not os.path.exists(CSV_PATH)]

    def print_state():
        s = state_holder[0]
        print("\n" + "=" * 60)
        print("  G1 全身关节角 (rad)")
        print("=" * 60)
        for i in range(min(30, len(s.motor_state))):
            q = float(s.motor_state[i].q)
            dq = float(s.motor_state[i].dq)
            deg = math.degrees(q)
            name = JOINT_NAMES[i] if i < len(JOINT_NAMES) else f"motor_{i}"
            print(f"  [{i:2d}] {name:25s}  q={q:+.4f} rad ({deg:+.2f}°)  dq={dq:+.4f}")
        print("=" * 60)
        # 输出可复制的列表
        q_list = [float(s.motor_state[i].q) for i in range(min(30, len(s.motor_state)))]
        print("\n关节角列表 (rad, 可复制):")
        print(q_list)
        if not args.no_csv:
            save_state_to_csv(s, csv_need_header)

    print_state()

    if not args.once:
        print("\n持续监听中，Ctrl+C 退出...")
        try:
            while True:
                time.sleep(0.5)
                print_state()
        except KeyboardInterrupt:
            print("\n退出。")


if __name__ == "__main__":
    main()
