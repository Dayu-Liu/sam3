#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
机器人移动模块：通过文件 IPC 与 RL Pipeline (MjlabLocoPolicy) 通信控制 G1 行走。

RL Pipeline 每步读取 /tmp/robojudo_ext_cmd.json 中的速度指令和 FSM 状态，
RL 策略据此生成行走电机指令，无需 LocoClient 或 Sport 模式切换。
"""

import json
import math
import os
import tempfile
import time

CMD_FILE = "/tmp/robojudo_ext_cmd.json"

# AGILE consumes an absolute pelvis/base-height command. Keep the values inside
# the range we have benchmarked most often; keyboard/manual control can adjust
# this field without changing the box_demo visual/IK flow.
STAND_HEIGHT = 0.72
MIN_HEIGHT = 0.40
MAX_HEIGHT = 0.72


def _write_cmd(fsm: str | None = None,
               forward: float = 0.0, lateral: float = 0.0, yaw: float = 0.0,
               height: float | None = None, units: str | None = None):
    """写入外部指令文件，RL Pipeline 读取后驱动行走。使用原子写入避免读到半截数据。"""
    cmd = {
        "fsm": fsm,
        "velocity": {"forward": forward, "lateral": lateral, "yaw": yaw},
        "timestamp": time.time(),
    }
    if height is not None:
        cmd["height"] = float(max(MIN_HEIGHT, min(MAX_HEIGHT, height)))
    if units is not None:
        cmd["units"] = str(units)
    try:
        dir_name = os.path.dirname(CMD_FILE) or "/tmp"
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".json")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(cmd, f)
            os.rename(tmp_path, CMD_FILE)
        except Exception:
            os.unlink(tmp_path)
            raise
    except Exception as e:
        print(f"  RobotMover: 写入指令文件失败: {e}")


def _clear_cmd():
    """清除指令文件。"""
    try:
        if os.path.exists(CMD_FILE):
            os.remove(CMD_FILE)
    except Exception:
        pass


class RobotMover:
    """通过文件 IPC 控制 G1 机器人行走（RL Pipeline 驱动）。"""

    def __init__(self, velocity: float = 0.2):
        """
        Args:
            velocity: 默认行走速度 (m/s)，建议 0.1~0.3
        """
        self._velocity = velocity
        self._height = STAND_HEIGHT

    def initialize(self):
        """切换到 RL_FULL 模式准备行走。RL Pipeline 的 prepare() 已完成站立。"""
        _write_cmd(fsm="RL_FULL", forward=0.0, lateral=0.0, yaw=0.0, height=self._height)
        time.sleep(1.5)  # 等待 FSM 转换完成
        print("  RobotMover: RL_FULL 模式就绪，可进行行走控制。")

    def _walk_with_refresh(self, vx: float, vy: float, vyaw: float,
                           duration_s: float):
        """高频刷新 IPC 文件，确保 pipeline 能读到命令。"""
        chunk = 0.1  # 每 100ms 刷新
        deadline = time.time() + duration_s
        while time.time() < deadline:
            _write_cmd(fsm="RL_FULL", forward=vx, lateral=vy, yaw=vyaw, height=self._height)
            time.sleep(chunk)
        # 停住
        _write_cmd(fsm="RL_FULL", forward=0.0, lateral=0.0, yaw=0.0, height=self._height)
        time.sleep(0.5)

    # _map_http_axis 映射常量：forward=[-0.5, 1.0], lateral=[-0.5, 0.5], yaw=[-0.5, 0.5]
    # 文件 IPC 值需在 [-1,1] 范围，由 pipeline 映射到实际速度
    _FWD_MAX = 1.0   # 前进 max=1.0 m/s
    _LAT_MAX = 0.5   # 侧移 max=0.5 m/s
    _YAW_MAX = 0.5   # 偏航 max=0.5 rad/s
    _MIN_WALK_DURATION = 0.5  # RL 策略至少需要 0.5s 才能从静止启动行走
    _MIN_FWD_NORM = 0.5  # 前进归一化速度下限（低于此值 RL 策略只会抖动不行走）
    _MIN_LAT_NORM = 1.0  # 侧移归一化速度下限（与键盘 a/d 按键力度一致）

    def move_forward(self, distance_m: float):
        """
        机器人前进/后退。

        Args:
            distance_m: 行走距离 (m)，正=前进，负=后退
        """
        if abs(distance_m) < 0.01:
            return

        vx = max(-1.0, min(1.0, self._velocity / self._FWD_MAX))
        # 短距离时保证最小归一化速度
        if abs(vx) < self._MIN_FWD_NORM:
            vx = self._MIN_FWD_NORM * (1.0 if distance_m > 0 else -1.0)
        # 用实际归一化速度反算实际速度和行走时长
        vx_raw = vx * self._FWD_MAX
        duration_s = abs(distance_m) / abs(vx_raw)
        duration_s = max(duration_s, self._MIN_WALK_DURATION)

        direction = "前进" if distance_m > 0 else "后退"
        print(f"  RobotMover: {direction} {abs(distance_m)*100:.0f}cm (v={vx_raw:+.2f}m/s, {duration_s:.2f}s)")
        self._walk_with_refresh(vx, 0.0, 0.0, duration_s)

    def move_left(self, distance_m: float):
        """
        机器人左移/右移。

        Args:
            distance_m: 行走距离 (m)，正=左移，负=右移
        """
        if abs(distance_m) < 0.01:
            return

        # 用最小归一化速度（与键盘 a/d 一致），按距离反算行走时长
        vy = self._MIN_LAT_NORM * (1.0 if distance_m > 0 else -1.0)
        vy_raw = vy * self._LAT_MAX
        duration_s = abs(distance_m) / abs(vy_raw)
        # RL 策略启动需要约 0.3-0.5s 加速，行走时长太短实际移不动
        duration_s = max(duration_s, 0.6)

        direction = "左移" if distance_m > 0 else "右移"
        print(f"  RobotMover: {direction} {abs(distance_m)*100:.0f}cm (v={vy_raw:+.2f}m/s, norm={vy:+.2f}, {duration_s:.2f}s)")
        self._walk_with_refresh(0.0, vy, 0.0, duration_s)

    def rotate(self, angle_rad: float, yaw_speed: float = 0.35):
        """
        原地转向。

        Args:
            angle_rad: 旋转角度 (rad)，正=左转，负=右转
            yaw_speed: 偏航速度 (rad/s)
        """
        if abs(angle_rad) < 0.04:
            return
        vyaw_raw = yaw_speed if angle_rad > 0 else -yaw_speed
        vyaw = vyaw_raw / self._YAW_MAX  # 归一化到 [-1,1]
        duration_s = abs(angle_rad) / yaw_speed
        direction = "左" if angle_rad > 0 else "右"
        print(f"  RobotMover: 原地往{direction}转 {abs(math.degrees(angle_rad)):.0f}°  ({duration_s:.2f}s, raw={vyaw_raw:+.2f}rad/s)")
        self._walk_with_refresh(0.0, 0.0, vyaw, duration_s)

    def stop(self):
        """停止移动（速度置零）。"""
        _write_cmd(fsm=None, forward=0.0, lateral=0.0, yaw=0.0, height=self._height)

    def set_height(self, height: float):
        """Set AGILE absolute base-height command and refresh IPC once."""
        self._height = float(max(MIN_HEIGHT, min(MAX_HEIGHT, height)))
        _write_cmd(fsm="RL_FULL", forward=0.0, lateral=0.0, yaw=0.0, height=self._height)
        print(f"  RobotMover: base height command = {self._height:.2f}m")

    def adjust_height(self, delta_m: float):
        """Increment AGILE height command by delta_m."""
        self.set_height(self._height + float(delta_m))

    def shutdown(self):
        """切换到 RL_LOWER 模式（下半身 RL 保持平衡，上半身留给手臂控制）。"""
        _write_cmd(fsm="RL_LOWER", forward=0.0, lateral=0.0, yaw=0.0, height=self._height)
        time.sleep(0.5)
        # 不删除文件：run_pipeline.py 需要持续读到 RL_LOWER，否则会切回默认模式
        # 导致 RL 策略重新控制上肢，与 arm_sdk 冲突引发全身抽搐
        print("  RobotMover: 已切换到 RL_LOWER 模式（IPC 文件保持）。")

    def release(self):
        """清除 IPC 文件，恢复键盘/遥控器手动控制。"""
        _clear_cmd()
        print("  RobotMover: IPC 文件已清除，可恢复手动控制。")
