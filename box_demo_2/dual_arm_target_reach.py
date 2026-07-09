#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
宇树G1双臂末端位置控制脚本

将左右手末端执行器分别平滑移动到指定的笛卡尔坐标点（躯干坐标系）。
运动学参数直接来自 G1 URDF，IK精度 < 0.1mm。

用法:
    python dual_arm_target_reach.py <lx> <ly> <lz> <rx> <ry> <rz> [网络接口]

参数 (单位：米, 躯干坐标系 — X朝前, Y朝左, Z朝上):
    lx ly lz    左手末端目标位置
    rx ry rz    右手末端目标位置
    网络接口     可选，DDS网络接口名，如 eth0

示例:
    python dual_arm_target_reach.py 0.35 0.25 0.45 0.35 -0.25 0.45
    python dual_arm_target_reach.py 0.35 0.25 0.45 0.35 -0.25 0.45 eth0

典型可达范围参考 (零位末端在 [0.25, ±0.15, 0.04]):
    左手: x∈[0.10,0.55]  y∈[0.00, 0.50]  z∈[-0.10,0.55]
    右手: x∈[0.10,0.55]  y∈[-0.50,0.00]  z∈[-0.10,0.55]

控制流程:
    阶段1 (3s)  : 当前位姿 → 零位
    阶段2 (5s)  : 零位 → 点A
    阶段3 (2s)  : 点A → 点B（y向内0.11）
    阶段4 (2s)  : 点B → 点C（z向上0.2）
    阶段5 (2s)  : 点C → 点B
    阶段6 (2s)  : 点B → 点A
    阶段7 (3s)  : 点A → 零位
    阶段8 (2s)  : 逐渐释放 arm_sdk 控制权（motor_cmd[29].q 淡出）
    腰部全程保持零位。

box_demo_2 说明:
    发布 **rt/arm_sdk**（与 merge_lowcmd_arm_sdk.py + run_pipeline -c g1_mjlab_loco_real_merge 配合；勿再发 rt/lowcmd）。
    每帧同步 **mode_machine / mode_pr**（与 rt/lowstate 一致）。
"""

import ctypes
import fcntl
import os
import select
import sys
import threading
import time
import math
import argparse
from enum import IntEnum
from pathlib import Path

import numpy as np

from grip_metrics_log import GripMetricsLog

_ddsc = Path(os.environ.get("CYCLONEDDS_HOME", Path.home() / "cyclonedds-0.10-install")) / "lib" / "libddsc.so.0"
if _ddsc.is_file():
    ctypes.CDLL(str(_ddsc), mode=ctypes.RTLD_GLOBAL)

from unitree_sdk2py.core.channel import (
    ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread


# ─────────────────────────────────────────────────────────────────
# 关节索引
# ─────────────────────────────────────────────────────────────────

class G1Joint(IntEnum):
    WaistYaw   = 12
    WaistRoll  = 13
    WaistPitch = 14
    LeftShoulderPitch  = 15
    LeftShoulderRoll   = 16
    LeftShoulderYaw    = 17
    LeftElbow          = 18
    LeftWristRoll      = 19
    LeftWristPitch     = 20   # 仅29DOF有效
    LeftWristYaw       = 21   # 仅29DOF有效
    RightShoulderPitch = 22
    RightShoulderRoll  = 23
    RightShoulderYaw   = 24
    RightElbow         = 25
    RightWristRoll     = 26
    RightWristPitch    = 27  # 仅29DOF有效
    RightWristYaw      = 28  # 仅29DOF有效
    ArmSdkEnable       = 29  # arm_sdk 权重：motor_cmd[29].q ∈ [0,1]


# 控制的关节列表：左臂7 + 右臂7 + 腰部3，顺序与 target_q 一一对应
ARM_JOINTS = [
    G1Joint.LeftShoulderPitch,  G1Joint.LeftShoulderRoll,
    G1Joint.LeftShoulderYaw,    G1Joint.LeftElbow,
    G1Joint.LeftWristRoll,      G1Joint.LeftWristPitch,
    G1Joint.LeftWristYaw,
    G1Joint.RightShoulderPitch, G1Joint.RightShoulderRoll,
    G1Joint.RightShoulderYaw,   G1Joint.RightElbow,
    G1Joint.RightWristRoll,     G1Joint.RightWristPitch,
    G1Joint.RightWristYaw,
    G1Joint.WaistYaw, G1Joint.WaistRoll, G1Joint.WaistPitch,
]
N_JOINTS = len(ARM_JOINTS)

# 零位姿态：两臂向外张开约 30°，避免过度贴近躯干
# ARM_JOINTS 顺序：左7 + 右7 + 腰3
# 索引1 = LeftShoulderRoll  (正值=向外)
# 索引8 = RightShoulderRoll (负值=向外)
# 索引14-16 = WaistYaw, WaistRoll, WaistPitch（站立状态实测值，单位 rad）
# _ZERO_ShoulderROLL = math.radians(0)   # 30° ≈ 0.524 rad
# _ZERO_ShoulderPITCH = math.radians(0)
# _ZERO_ShoulderYAW = math.radians(30)
# _ZERO_ELBOW = math.radians(0)
# _ZERO_WristPitch = math.radians(0)
# _ZERO_WristYAW = math.radians(0)

_ZERO_ShoulderPITCH = math.radians(84.9)
_ZERO_ShoulderROLL  = math.radians(6.8)
_ZERO_ShoulderYAW   = math.radians(13.8)
_ZERO_ELBOW         = math.radians(42.4)
_ZERO_WristPitch    = math.radians(38.7)
_ZERO_WristYAW      = math.radians(7.7)

# ## 张开双臂预备
# ZERO_Q = [0.0] * N_JOINTS
# ZERO_Q[1] =  _ZERO_ShoulderROLL   # 左肩横滚：向外 30°
# ZERO_Q[8] = -_ZERO_ShoulderROLL   # 右肩横滚：向外 30°
# ZERO_Q[0] =  _ZERO_ShoulderPITCH   # 左肩横滚：向外 30°
# ZERO_Q[7] =  _ZERO_ShoulderPITCH   # 右肩横滚：向外 30°
# ZERO_Q[3] =  -_ZERO_ELBOW
# ZERO_Q[10] =  -_ZERO_ELBOW 
# ZERO_Q[5] =  -_ZERO_WristPitch
# ZERO_Q[12] =  -_ZERO_WristPitch
# ZERO_Q[14] = 0.0      # waist_yaw   (站立状态)
# ZERO_Q[15] = 0.0      # waist_roll  (站立状态)
# ZERO_Q[16] = 0.0      # waist_pitch (站立状态)

##双手叉腰预备
ZERO_Q = [0.0] * N_JOINTS
ZERO_Q[1] =  _ZERO_ShoulderROLL   # 左肩横滚：向外 30°
ZERO_Q[8] = -_ZERO_ShoulderROLL   # 右肩横滚：向外 30°
ZERO_Q[0] =  _ZERO_ShoulderPITCH   
ZERO_Q[7] =  _ZERO_ShoulderPITCH   
ZERO_Q[2] =  _ZERO_ShoulderYAW   
ZERO_Q[9] =  -_ZERO_ShoulderYAW   
ZERO_Q[3] =  -_ZERO_ELBOW
ZERO_Q[10] =  -_ZERO_ELBOW 
ZERO_Q[5] =  -_ZERO_WristPitch
ZERO_Q[12] =  -_ZERO_WristPitch
ZERO_Q[6] =  _ZERO_WristYAW
ZERO_Q[13] =  -_ZERO_WristYAW
ZERO_Q[14] = 0.0                     # waist_yaw   (站立状态)
ZERO_Q[15] = 0.0                     # waist_roll  (站立状态)
ZERO_Q[16] = 0.0                     # waist_pitch (站立状态)

# 交还给 GR00T-WBC 前手臂+腰要收到的「中性姿势」。必须等于 GR00T adapter 释放
# arm_sdk 后保持的上半身姿势，否则释放瞬间手臂会从这里跳到 GR00T 的姿势 → 整机失稳
# （"散掉"）。GR00T 的 DEFAULT_MOTOR_ANGLES 上半身（腰12-14 + 臂15-28）全是 0，所以
# 这里也全 0（手臂自然下垂、腰回正）。旧值是照 RoboJuDo 的抱姿调的——RoboJuDo 的
# RL_LOWER 会硬撑那个姿势，GR00T 不会（解耦下策略不控手臂、释放后停在 0 位）。
RL_LOWER_HANDOFF_Q = [
    0.29, 0.22, 0.0, 0.98, 0.2, 0.03, -0.03,   # 左臂 7
    0.29, -0.22, 0.0, 0.98, -0.2, 0.03, 0.03,  # 右臂 7
    0.0, 0.0, 0.0,                              # 腰 yaw / roll / pitch
]

# ─────────────────────────────────────────────────────────────────
# 运动学求解器（直接基于 G1 URDF 关节参数，IK精度 < 0.1mm）
# ─────────────────────────────────────────────────────────────────

class ArmKinematics:
    """
    基于G1 URDF实测参数的7自由度机械臂运动学求解器。

    坐标系：躯干坐标系（torso_link），X朝前、Y朝左、Z朝上。
    末端定义：手掌中心 (left/right_rubber_hand)。

    关节参数格式（来自URDF joint origin）:
        (xyz, rpy, rotation_axis)
        其中 xyz/rpy 描述父链接到关节坐标系的固定变换，
        rotation_axis 为关节转动轴（在关节坐标系内）。
    """

    # 左臂关节链：肩俯仰 → 肩横滚 → 肩偏航 → 肘 → 腕横滚 → 腕俯仰 → 腕偏航
    _LEFT_JOINTS = [
        ([0.0039563,  0.10022,  0.24778], [ 0.27931,  5.4949e-5, -1.9159e-4], [0, 1, 0]),
        ([0.0,        0.038,   -0.013831], [-0.27925,  0.0,        0.0      ], [1, 0, 0]),
        ([0.0,        0.00624, -0.1032  ], [ 0.0,      0.0,        0.0      ], [0, 0, 1]),
        ([0.015783,   0.0,     -0.080518], [ 0.0,      0.0,        0.0      ], [0, 1, 0]),
        ([0.100,      0.00188791, -0.010], [ 0.0,      0.0,        0.0      ], [1, 0, 0]),
        ([0.038,      0.0,      0.0     ], [ 0.0,      0.0,        0.0      ], [0, 1, 0]),
        ([0.046,      0.0,      0.0     ], [ 0.0,      0.0,        0.0      ], [0, 0, 1]),
    ]
    _LEFT_EE  = ([0.12, 0.0, 0.0], [0.0, 0.0, 0.0])   # 手掌中心，从 wrist_yaw 沿 X 轴 0.12m

    # 右臂关节链（Y/Z 轴方向镜像）
    _RIGHT_JOINTS = [
        ([0.0039563, -0.10021,  0.24778], [-0.27931,  5.4949e-5,  1.9159e-4], [0, 1, 0]),
        ([0.0,       -0.038,   -0.013831], [ 0.27925,  0.0,        0.0      ], [1, 0, 0]),
        ([0.0,       -0.00624, -0.1032  ], [ 0.0,      0.0,        0.0      ], [0, 0, 1]),
        ([0.015783,   0.0,     -0.080518], [ 0.0,      0.0,        0.0      ], [0, 1, 0]),
        ([0.100,     -0.00188791, -0.010], [ 0.0,      0.0,        0.0      ], [1, 0, 0]),
        ([0.038,      0.0,      0.0     ], [ 0.0,      0.0,        0.0      ], [0, 1, 0]),
        ([0.046,      0.0,      0.0     ], [ 0.0,      0.0,        0.0      ], [0, 0, 1]),
    ]
    _RIGHT_EE = ([0.12, 0.0, 0.0], [0.0, 0.0, 0.0])    # 手掌中心，从 wrist_yaw 沿 X 轴 0.12m

    # 关节角度限制 [min, max] (rad)
    _LIMITS_LEFT = [
        (-3.0892, 2.6704), (-1.5882, 2.2515), (-2.618, 2.618),
        (-1.0472, 2.0944), (-1.9722, 1.9722), (-1.6144, 1.6144), (-1.6144, 1.6144),
    ]
    _LIMITS_RIGHT = [
        (-3.0892, 2.6704), (-2.2515, 1.5882), (-2.618, 2.618),
        (-1.0472, 2.0944), (-1.9722, 1.9722), (-1.6144, 1.6144), (-1.6144, 1.6144),
    ]

    # ── 矩阵工具 ─────────────────────────────────────────────────

    @staticmethod
    def _rpy_mat(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """URDF RPY → 旋转矩阵（外旋 ZYX 顺序）"""
        cr, sr = math.cos(roll),  math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw),   math.sin(yaw)
        Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
        Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
        Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
        return Rz @ Ry @ Rx

    @staticmethod
    def _axis_rot(axis: list, q: float) -> np.ndarray:
        """绕任意轴 axis 旋转角度 q 的旋转矩阵"""
        ax = np.asarray(axis, dtype=float)
        c, s = math.cos(q), math.sin(q)
        x, y, z = ax
        return np.array([
            [c + x*x*(1-c),   x*y*(1-c) - z*s, x*z*(1-c) + y*s],
            [y*x*(1-c) + z*s, c + y*y*(1-c),   y*z*(1-c) - x*s],
            [z*x*(1-c) - y*s, z*y*(1-c) + x*s, c + z*z*(1-c)  ],
        ])

    def _joint_T(self, xyz, rpy, axis, q) -> np.ndarray:
        """单个关节的 4×4 齐次变换矩阵"""
        T = np.eye(4)
        T[:3, :3] = self._rpy_mat(*rpy) @ self._axis_rot(axis, q)
        T[:3,  3] = xyz
        return T

    def _fixed_T(self, xyz, rpy) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = self._rpy_mat(*rpy)
        T[:3,  3] = xyz
        return T

    # ── 正运动学 ─────────────────────────────────────────────────

    def forward_kinematics(self, q: list, left: bool = True) -> tuple:
        """
        正运动学：7个关节角 → 手掌末端位置（躯干坐标系）

        Returns:
            position (np.ndarray, shape 3): 末端位置 [x, y, z]
            rotation (np.ndarray, shape 3×3): 末端姿态旋转矩阵
        """
        joints = self._LEFT_JOINTS if left else self._RIGHT_JOINTS
        ee     = self._LEFT_EE     if left else self._RIGHT_EE
        T = np.eye(4)
        for i, (xyz, rpy, axis) in enumerate(joints):
            T = T @ self._joint_T(xyz, rpy, axis, q[i])
        T = T @ self._fixed_T(*ee)
        return T[:3, 3].copy(), T[:3, :3].copy()

    # ── 数值雅可比 ───────────────────────────────────────────────

    def _jacobian(self, q: np.ndarray, left: bool) -> np.ndarray:
        J   = np.zeros((3, 7))
        p0, _ = self.forward_kinematics(q, left)
        eps = 1e-6
        for i in range(7):
            q2 = q.copy(); q2[i] += eps
            p1, _ = self.forward_kinematics(q2, left)
            J[:, i] = (p1 - p0) / eps
        return J

    # ── 逆运动学（阻尼最小二乘迭代法）──────────────────────────

    def _ik_single(
        self,
        target: np.ndarray,
        left: bool,
        q: np.ndarray,
        max_iter: int,
        tol: float,
        damping: float,
        limits: list,
    ) -> tuple:
        """单起点 IK 迭代，返回 (q_list, err_m)。"""
        for _ in range(max_iter):
            pos, _ = self.forward_kinematics(q, left)
            err = target - pos
            if np.linalg.norm(err) < tol:
                break
            J  = self._jacobian(q, left)
            JT = J.T
            dq = JT @ np.linalg.solve(J @ JT + damping**2 * np.eye(3), err)
            q += dq
            for i in range(7):
                q[i] = float(np.clip(q[i], limits[i][0], limits[i][1]))
        final_pos, _ = self.forward_kinematics(q, left)
        return q.tolist(), float(np.linalg.norm(target - final_pos))

    def inverse_kinematics(
        self,
        target_pos: list,
        left: bool = True,
        q0: list = None,
        max_iter: int = 1000,
        tol: float = 1e-4,
        damping: float = 0.02,
        num_restarts: int = 10,
    ) -> tuple:
        """
        逆运动学：目标末端位置 → 7个关节角（多起点，取最优解）

        Args:
            target_pos:    目标位置 [x, y, z]，躯干坐标系，单位米
            left:          True=左臂，False=右臂
            q0:            优先使用的初始关节角（可选）
            max_iter:      每次起点最大迭代次数
            tol:           收敛阈值（米）
            damping:       阻尼系数
            num_restarts:  随机起点数（在启发式起点之外额外追加）

        Returns:
            q     (list[float]): 7个关节角，单位弧度
            err_m (float):       最终末端误差，单位米
        """
        limits = self._LIMITS_LEFT if left else self._LIMITS_RIGHT
        target = np.array(target_pos, dtype=float)

        # ── 启发式初始构型（前伸/举手/侧伸等典型姿态）────────────
        # q 顺序: shoulder_pitch, shoulder_roll, shoulder_yaw,
        #         elbow, wrist_roll, wrist_pitch, wrist_yaw
        heuristic_starts: list[np.ndarray] = [
            np.zeros(7),                                         # 零位
            np.array([-0.5, 0.0,  0.0, 1.0, 0.0, 0.0, 0.0]),  # 前伸（低）
            np.array([-1.0, 0.0,  0.0, 1.5, 0.0, 0.0, 0.0]),  # 前伸（深）
            np.array([-0.3, 0.3,  0.0, 0.8, 0.0, 0.0, 0.0]),  # 略外展
            np.array([-0.8, 0.0,  0.3, 1.2, 0.0, 0.0, 0.0]),  # 偏转
            np.array([-0.6, 0.0, -0.3, 1.0, 0.0, 0.0, 0.0]),  # 内旋
        ]
        # 对右臂，肩横滚方向取反
        if not left:
            for s in heuristic_starts[1:]:
                s[1] = -s[1]

        starts: list[np.ndarray] = []
        if q0 is not None:
            starts.append(np.array(q0, dtype=float))
        starts.extend(heuristic_starts)

        # 随机起点
        rng = np.random.default_rng(0)
        for _ in range(num_restarts):
            starts.append(np.array(
                [rng.uniform(lo, hi) for lo, hi in limits], dtype=float
            ))

        best_q, best_err = None, float("inf")
        for q_init in starts:
            q_clipped = np.array(
                [np.clip(q_init[i], limits[i][0], limits[i][1]) for i in range(7)],
                dtype=float,
            )
            q_sol, err = self._ik_single(target, left, q_clipped, max_iter, tol, damping, limits)
            if err < best_err:
                best_err = err
                best_q = q_sol
            if best_err < tol:
                break

        return best_q, best_err


# ─────────────────────────────────────────────────────────────────
# 控制阶段
# ─────────────────────────────────────────────────────────────────

class _LegacyStageUnused(IntEnum):
    ZERO     = 0   # 当前位姿 → 零位
    ZERO_TO_A = 1   # 零位 → 点A
    A_TO_B   = 2   # 点A → 点B（y向内0.11）
    B_TO_C   = 3   # 点B → 点C（z向上0.2）
    C_TO_B   = 4   # 点C → 点B
    B_TO_A   = 5   # 点B → 点A
    A_TO_ZERO = 6   # 点A → 零位
    RELEASE  = 7   # 逐渐释放 arm_sdk 控制权
    DONE     = 8


# ─────────────────────────────────────────────────────────────────
# 双臂控制器
# ─────────────────────────────────────────────────────────────────

class Stage(IntEnum):
    ZERO = 0
    ZERO_TO_A = 1
    A_TO_B = 2
    B_TO_C = 3
    C_TO_B = 4
    B_TO_A = 5
    A_TO_ZERO = 6
    WAIT_RL_LOWER = 7
    BLEND_TO_HANDOFF = 8
    RELEASE = 9
    DONE = 10
    RECOVER_TO_ZERO = 11


class RecoverStep(IntEnum):
    NONE = 0
    ACK_DROP = 1
    WAIT_CAPTURE = 2
    WAIT_GO_A = 3


# 夹持检测：阶段3/4/5 结束后、按 Enter 前（仅电机力矩 tau_est）
GRIP_CHECK_ARM_INDICES = (1, 3, 8, 10)   # 左右肩 roll、左右肘
GRIP_DROP_RATIO = 0.70
GRIP_BASELINE_WINDOW = 5       # 各阶段用第 5 个 250ms 窗口的 tau_mean 作基线

# 等 Enter 期间持续夹持监测（阶段3/4/5 结束后直至按 Enter 或判脱落）
GRIP_WATCH_INTERVAL_S = 0.25   # 每 250ms 汇总判定一次
GRIP_WATCH_FAIL_STREAK = 2     # 连续 N 次判定脱落才触发（降低误报）

_GRIP_CHECK_AFTER_STAGES = frozenset({Stage.A_TO_B, Stage.B_TO_C, Stage.C_TO_B})
_GRIP_STAGE_LABELS = {
    Stage.A_TO_B: "阶段3 A→B",
    Stage.B_TO_C: "阶段4 B→C",
    Stage.C_TO_B: "阶段5 C→B",
}


def solve_all_ik_targets(
    left_target: list,
    right_target: list,
    *,
    left_q0: list | None = None,
    right_q0: list | None = None,
    ik_warn_limit: float = 0.020,
    ik_err_limit: float = 0.100,
) -> dict:
    """由左右手 torso 目标点求解点 A/B/C 关节角与 FK。"""
    ik = ArmKinematics()
    left_a = np.array(left_target, dtype=float)
    right_a = np.array(right_target, dtype=float)

    a_restarts = 2 if left_q0 is not None else 8
    a_max_iter = 300 if left_q0 is not None else 600
    left_q_a, left_err = ik.inverse_kinematics(
        left_a.tolist(), left=True, q0=left_q0,
        num_restarts=a_restarts, max_iter=a_max_iter,
    )
    right_q_a, right_err = ik.inverse_kinematics(
        right_a.tolist(), left=False, q0=right_q0,
        num_restarts=a_restarts, max_iter=a_max_iter,
    )
    left_fk, _ = ik.forward_kinematics(left_q_a, left=True)
    right_fk, _ = ik.forward_kinematics(right_q_a, left=False)

    print(f"  左臂 IK 误差: {left_err*1000:.2f} mm  FK验证: {left_fk.round(3).tolist()}")
    print(f"  右臂 IK 误差: {right_err*1000:.2f} mm  FK验证: {right_fk.round(3).tolist()}")

    if left_err > ik_err_limit or right_err > ik_err_limit:
        raise ValueError(
            f"IK误差过大 (左:{left_err*1000:.1f}mm, 右:{right_err*1000:.1f}mm)，"
            "目标点超出机械臂工作空间，请将机器人移近箱子后重试。"
        )
    if left_err > ik_warn_limit or right_err > ik_warn_limit:
        print(f"  [警告] IK误差偏大 (左:{left_err*1000:.1f}mm, 右:{right_err*1000:.1f}mm)")

    left_b = left_a.copy()
    right_b = right_a.copy()
    left_b[1] -= 0.10
    right_b[1] += 0.10
    left_q_b, _ = ik.inverse_kinematics(
        left_b.tolist(), left=True, q0=left_q_a, num_restarts=2, max_iter=200,
    )
    right_q_b, _ = ik.inverse_kinematics(
        right_b.tolist(), left=False, q0=right_q_a, num_restarts=2, max_iter=200,
    )

    left_c = left_b.copy()
    right_c = right_b.copy()
    left_c[0] -= 0.05
    right_c[0] -= 0.05
    left_c[2] += 0.1
    right_c[2] += 0.1
    left_q_c, _ = ik.inverse_kinematics(
        left_c.tolist(), left=True, q0=left_q_b, num_restarts=2, max_iter=200,
    )
    right_q_c, _ = ik.inverse_kinematics(
        right_c.tolist(), left=False, q0=right_q_b, num_restarts=2, max_iter=200,
    )

    waist = [ZERO_Q[14], ZERO_Q[15], ZERO_Q[16]]
    return {
        "left_a": left_a,
        "right_a": right_a,
        "left_fk": left_fk,
        "right_fk": right_fk,
        "target_q_a": left_q_a + right_q_a + waist,
        "target_q_b": left_q_b + right_q_b + waist,
        "target_q_c": left_q_c + right_q_c + waist,
    }


class DualArmController:

    # 各阶段时长（秒）
    DURATION = {
        Stage.ZERO: 3.0,
        Stage.ZERO_TO_A:  3.0,   # 手臂前伸，重心前移，放慢让RL策略补偿
        Stage.A_TO_B:  1.0,
        Stage.B_TO_C:  1.0,
        Stage.C_TO_B:  1.0,   # 手臂前伸+下放箱子，重心前移最大，放慢
        Stage.B_TO_A:  1.0,   # 手臂外伸，重心微前移
        Stage.A_TO_ZERO: 2.0,
        Stage.RELEASE: 2.0,
        Stage.RECOVER_TO_ZERO: 2.0,
    }
    DURATION[Stage.BLEND_TO_HANDOFF] = 2.5
    DURATION[Stage.RELEASE] = 2.5

    KP = 80.0
    KD = 10.0
    KP_WAIST = 250.0
    KD_WAIST = 5.0
    KP_SHOULDER_PITCH = 150.0
    KD_SHOULDER_PITCH = 10.0
    KP_SHOULDER_ROLL = 120.0
    KD_SHOULDER_ROLL = 10.0
    KP_SHOULDER_YAW = 130.0
    KD_SHOULDER_YAW = 10.0
    KP_ELBOW = 130.0
    KD_ELBOW = 10.0
    KP_WRIST_ROLL = 180.0
    KD_WRIST_ROLL = 10.0
    KP_WRIST_PITCH = 180.0
    KD_WRIST_PITCH = 10.0
    KP_WRIST_YAW = 180.0
    KD_WRIST_YAW = 10.0

    # IK误差阈值（米）
    IK_WARN_LIMIT  = 0.020   # 20mm：超出则打印警告
    IK_ERR_LIMIT   = 0.100   # 100mm：超出则拒绝执行（几何上无法到达）

    def __init__(self, left_target: list, right_target: list,
                 stop_after_zero: bool = False, start_from_reach: bool = False,
                 end_behavior: str = "handoff_rl_lower",
                 left_q0: list = None, right_q0: list = None,
                 enable_grip_check: bool = True,
                 grip_log_dir: str | None = None,
                 grip_log_tag: str | None = None):
        """
        Args:
            left_target, right_target: 左右手目标位置 [x,y,z] (torso 系)
            stop_after_zero: 若 True，完成零位后即停止（用于「先到零位再采图」流程）
            start_from_reach: 若 True，跳过零位阶段，直接从零位→目标开始（假定已在零位）
            enable_grip_check: 阶段3/4/5 结束后检测箱子是否仍在手中
            grip_log_dir: 夹持指标 JSON/PNG 输出目录（与 vision_log 同 run 目录即可）
            grip_log_tag: 输出文件名后缀，如 attempt_01
        """
        if end_behavior not in ("handoff_rl_lower", "release", "hold_handoff"):
            raise ValueError(f"Unsupported end_behavior: {end_behavior}")

        print("正在求解逆运动学...")
        solved = solve_all_ik_targets(
            left_target, right_target,
            left_q0=left_q0, right_q0=right_q0,
            ik_warn_limit=self.IK_WARN_LIMIT,
            ik_err_limit=self.IK_ERR_LIMIT,
        )
        self.left_a = solved["left_a"]
        self.right_a = solved["right_a"]
        self.left_fk = solved["left_fk"]
        self.right_fk = solved["right_fk"]
        self.target_q_a = solved["target_q_a"]
        self.target_q_b = solved["target_q_b"]
        self.target_q_c = solved["target_q_c"]
        self.target_q: list = self.target_q_a

        self._stop_after_zero = stop_after_zero
        self._start_from_reach = start_from_reach
        self.end_behavior = end_behavior
        self.handoff_q = list(RL_LOWER_HANDOFF_Q)
        self._enable_grip_check = enable_grip_check
        self._grip_log_dir = grip_log_dir
        self._grip_log_tag = grip_log_tag
        self._grip_metrics_log = (
            GripMetricsLog(grip_log_dir) if enable_grip_check else None
        )
        self._grip_drop_count = 0
        self._grip_regrasp_count = 0
        self._regrasp_callback = None

        # 夹持检测 / 重抓恢复
        self._grip_tau_baseline: float | None = None
        self._grip_check_next_stage = Stage.A_TO_B
        self._grip_check_source_stage = Stage.A_TO_B
        self._recovery_step = RecoverStep.NONE
        self._grip_watch_active = False
        self._grip_watch_t = 0.0
        self._grip_watch_samples: list[float] = []
        self._grip_watch_window_idx = 0
        self._grip_watch_consecutive_fails = 0
        self._grip_drop_detected = False

        # 运行时状态（start_from_reach 时跳过零位，直接从 ZERO_TO_A 开始）
        self._stage          = Stage.ZERO_TO_A if start_from_reach else Stage.ZERO
        self._stage_t        = 0.0
        self._q_stage_start  = None   # 每阶段初始时快照一次
        self._control_dt     = 0.02   # 50 Hz
        self.done            = False
        self._abort          = False
        # 勿在 RecurrentThread 里调 input()：会卡住 50Hz，arm_sdk 停发 → merge 认为过期而用 RL 上半身
        self._waiting_enter = False
        self._hold_q_cmd: list = list(ZERO_Q)
        self._pending_next_stage = Stage.ZERO
        self._release_q_cmd: list = list(ZERO_Q)
        self._enter_prompt = "按 Enter 继续..."
        self._stage_end_dispatched = False

        # DDS
        self._low_cmd    = unitree_hg_msg_dds__LowCmd_()
        self._low_state  = None
        self._state_ready = False
        self._crc        = CRC()
        self._ctrl_thread = None
        self._ctrl_done = False

    def _stop_ctrl_thread(self) -> None:
        """停止 50Hz 控制线程。run() 返回后必须调用，否则下一轮会与 keeper/新 controller 双写 arm_sdk。"""
        if self._ctrl_thread is None:
            return
        self._ctrl_done = True
        try:
            self._ctrl_thread.Wait(timeout=1.0)
        except Exception:
            pass
        self._ctrl_thread = None

    # ── DDS 回调 ─────────────────────────────────────────────────

    def _on_low_state(self, msg: LowState_):
        self._low_state = msg
        if not self._state_ready:
            self._state_ready = True

    # ── 工具函数 ─────────────────────────────────────────────────

    def _read_q(self) -> list:
        return [self._low_state.motor_state[int(j)].q for j in ARM_JOINTS]

    def _kp_for_arm_index(self, arm_idx: int) -> float:
        joint = ARM_JOINTS[arm_idx]
        if arm_idx >= 14:
            return self.KP_WAIST
        if joint in (G1Joint.LeftShoulderPitch, G1Joint.RightShoulderPitch):
            return self.KP_SHOULDER_PITCH
        if joint in (G1Joint.LeftShoulderRoll, G1Joint.RightShoulderRoll):
            return self.KP_SHOULDER_ROLL
        if joint in (G1Joint.LeftShoulderYaw, G1Joint.RightShoulderYaw):
            return self.KP_SHOULDER_YAW
        if joint in (G1Joint.LeftElbow, G1Joint.RightElbow):
            return self.KP_ELBOW
        if joint in (G1Joint.LeftWristRoll, G1Joint.RightWristRoll):
            return self.KP_WRIST_ROLL
        if joint in (G1Joint.LeftWristPitch, G1Joint.RightWristPitch):
            return self.KP_WRIST_PITCH
        if joint in (G1Joint.LeftWristYaw, G1Joint.RightWristYaw):
            return self.KP_WRIST_YAW
        return self.KP

    def _sample_grip_tau(self) -> float | None:
        """返回 4 监测关节估计力矩绝对值的平均 (Nm)。无 state 时返回 None。"""
        if self._low_state is None:
            return None
        tau_vals: list[float] = []
        for arm_idx in GRIP_CHECK_ARM_INDICES:
            joint = ARM_JOINTS[arm_idx]
            ms = self._low_state.motor_state[int(joint)]
            tau_vals.append(abs(float(getattr(ms, "tau_est", 0.0))))
        return sum(tau_vals) / max(len(tau_vals), 1)

    def _evaluate_grip_tau_list(self, samples: list[float]) -> tuple[float, dict]:
        if not samples:
            return 0.0, {"tau_mean": 0.0, "n": 0}
        tau_m = float(np.mean(samples))
        return tau_m, {"tau_mean": tau_m, "n": len(samples)}

    def _clear_grip_watch(self) -> None:
        self._grip_watch_active = False
        self._grip_watch_t = 0.0
        self._grip_watch_samples = []
        self._grip_watch_consecutive_fails = 0

    def _start_grip_watch(self) -> None:
        self._grip_watch_active = True
        self._grip_watch_t = 0.0
        self._grip_watch_samples = []
        self._grip_watch_window_idx = 0
        self._grip_watch_consecutive_fails = 0
        self._grip_drop_detected = False

    def _grip_watch_tick(self) -> None:
        sample = self._sample_grip_tau()
        if sample is not None:
            self._grip_watch_samples.append(sample)
        self._grip_watch_t += self._control_dt
        if self._grip_watch_t < GRIP_WATCH_INTERVAL_S:
            return
        self._grip_watch_t = 0.0
        if not self._grip_watch_samples:
            return
        tau_mean, details = self._evaluate_grip_tau_list(self._grip_watch_samples)
        self._grip_watch_samples = []
        self._grip_watch_window_idx += 1
        window_idx = self._grip_watch_window_idx
        label = _GRIP_STAGE_LABELS.get(self._grip_check_source_stage, "夹持")

        # 前 4 个 250ms 窗口仅记录，不判脱落
        if window_idx < GRIP_BASELINE_WINDOW:
            self._record_grip_metrics(
                label, details, is_baseline=False, gripped=None,
                window_idx=window_idx,
            )
            return

        # 第 5 个 250ms 窗口：记录本阶段基线，不判脱落
        if window_idx == GRIP_BASELINE_WINDOW:
            self._grip_tau_baseline = tau_mean
            thr = self._grip_tau_baseline * GRIP_DROP_RATIO
            print(
                f"  [{label}] 力矩基线 tau_mean={self._grip_tau_baseline:.2f}Nm "
                f"(第{GRIP_BASELINE_WINDOW}个250ms窗口, 脱落阈值={thr:.2f}Nm)"
            )
            self._record_grip_metrics(
                label, details, is_baseline=True, gripped=True,
                window_idx=window_idx,
            )
            self._grip_watch_consecutive_fails = 0
            return

        gripped = self._is_box_gripped(tau_mean)
        self._record_grip_metrics(
            label, details, is_baseline=False, gripped=gripped,
            window_idx=window_idx,
        )
        if gripped:
            self._grip_watch_consecutive_fails = 0
            return
        self._grip_watch_consecutive_fails += 1
        if self._grip_watch_consecutive_fails < GRIP_WATCH_FAIL_STREAK:
            return
        self._on_grip_watch_fail(tau_mean, details)

    def _record_grip_metrics(
        self,
        label: str,
        details: dict,
        *,
        is_baseline: bool,
        gripped: bool | None,
        drop_detected: bool = False,
        window_idx: int | None = None,
    ) -> None:
        if self._grip_metrics_log is None:
            return
        self._grip_metrics_log.append(
            stage_label=label,
            tau_mean=float(details["tau_mean"]),
            is_baseline=is_baseline,
            gripped=gripped,
            baseline_tau=self._grip_tau_baseline,
            drop_detected=drop_detected,
            window_idx=window_idx,
        )

    def _schedule_grip_metrics_save(self, *, event: str | None = None) -> None:
        """后台落盘，避免 matplotlib 阻塞主线程导致 arm_sdk 空窗。"""
        if self._grip_metrics_log is None or not self._grip_metrics_log.records:
            return
        log = self._grip_metrics_log
        out_dir = self._grip_log_dir
        base = self._grip_log_tag or "grip"
        tag = f"{base}_{event}" if event else base

        def _worker() -> None:
            try:
                log.save(out_dir, tag=tag)
            except Exception as exc:
                print(f"[grip_log] 后台保存失败: {exc}")

        threading.Thread(
            target=_worker, daemon=True, name=f"grip_log_{tag}",
        ).start()

    def _flush_grip_metrics(self, event: str) -> None:
        """脱落/重抓等关键节点立即落盘（文件名带 event 后缀）。"""
        self._schedule_grip_metrics_save(event=event)

    def _on_grip_watch_fail(self, tau_mean: float, details: dict) -> None:
        if self._recovery_step != RecoverStep.NONE:
            return
        label = _GRIP_STAGE_LABELS.get(
            self._grip_check_source_stage, "夹持等待",
        )
        print(
            f"\n⚠ [{label}] 等待 Enter 期间检测到箱子脱落 "
            f"(tau_mean={tau_mean:.2f}Nm)"
        )
        self._record_grip_metrics(
            label, details, is_baseline=False, gripped=False,
            drop_detected=True,
        )
        self._grip_drop_count += 1
        self._flush_grip_metrics(f"drop{self._grip_drop_count:02d}")
        self._clear_grip_watch()
        self._grip_drop_detected = True
        self._begin_drop_recovery(label, tau_mean, details)

    def _is_box_gripped(self, tau_mean: float) -> bool:
        """基线建立后：力矩低于基线的 70% 则视为未夹住。"""
        if self._grip_tau_baseline is None:
            return True
        return tau_mean >= self._grip_tau_baseline * GRIP_DROP_RATIO

    def _update_targets_from_regrasp(self, left_target: list, right_target: list) -> None:
        """重拍后更新点 A/B/C。IK 初值与首次抓取对齐，避免从零位收敛到错误构型。"""
        prev_left_q = list(self.target_q_a[:7])
        prev_right_q = list(self.target_q_a[7:14])

        def _solve(lq0, rq0, label: str) -> dict:
            print(f"  [重抓 IK] {label}")
            return solve_all_ik_targets(
                left_target, right_target,
                left_q0=lq0, right_q0=rq0,
                ik_warn_limit=self.IK_WARN_LIMIT,
                ik_err_limit=self.IK_ERR_LIMIT,
            )

        print("[重抓] 求解逆运动学...")
        solved: dict | None
        try:
            solved = _solve(prev_left_q, prev_right_q, "沿用上一轮点A构型作初值")
        except ValueError as exc:
            print(f"  [重抓 IK] 初值求解失败: {exc}")
            solved = None

        if solved is not None:
            ik = ArmKinematics()
            lf, _ = ik.forward_kinematics(solved["target_q_a"][:7], left=True)
            rf, _ = ik.forward_kinematics(solved["target_q_a"][7:14], left=False)
            left_err = float(np.linalg.norm(lf - solved["left_a"]))
            right_err = float(np.linalg.norm(rf - solved["right_a"]))
            if max(left_err, right_err) > self.IK_WARN_LIMIT:
                print(
                    f"  [重抓 IK] 初值解误差偏大 "
                    f"(左:{left_err*1000:.1f}mm 右:{right_err*1000:.1f}mm)，"
                    "改为全起点搜索..."
                )
                solved = None

        if solved is None:
            solved = _solve(None, None, "全起点搜索（与首次抓取一致）")

        self.left_a = solved["left_a"]
        self.right_a = solved["right_a"]
        self.left_fk = solved["left_fk"]
        self.right_fk = solved["right_fk"]
        self.target_q_a = solved["target_q_a"]
        self.target_q_b = solved["target_q_b"]
        self.target_q_c = solved["target_q_c"]
        self.target_q = self.target_q_a
        self._grip_tau_baseline = None
        print(f"  新点A 左 torso: {self.left_a.round(3).tolist()}  FK: {self.left_fk.round(3).tolist()}")
        print(f"  新点A 右 torso: {self.right_a.round(3).tolist()}  FK: {self.right_fk.round(3).tolist()}")
        self._grip_regrasp_count += 1
        self._flush_grip_metrics(f"regrasp{self._grip_regrasp_count:02d}")

    def _begin_grip_wait_enter(self, completed_stage: Stage, next_stage: Stage) -> None:
        """阶段3/4/5 结束后：保持夹持姿态，等 Enter，全程持续监测。"""
        self._clear_grip_watch()
        self._grip_tau_baseline = None
        self._grip_check_source_stage = completed_stage
        self._grip_check_next_stage = next_stage
        self._hold_q_cmd = self._terminal_q_for_stage(completed_stage)
        self._pending_next_stage = next_stage
        label = _GRIP_STAGE_LABELS.get(completed_stage, str(completed_stage))
        print(f"\n[{label}] 保持夹持，等待 Enter（期间持续监测）")
        self._waiting_enter = True
        self._enter_prompt = f"[{label}] 按 Enter 继续（持续监测夹持）...\n"
        self._start_grip_watch()

    def _begin_drop_recovery(self, label: str, tau_mean: float, details: dict) -> None:
        self._clear_grip_watch()
        baseline_s = (
            f"{self._grip_tau_baseline:.2f}Nm"
            if self._grip_tau_baseline is not None else "无"
        )
        print(f"\n⚠ [{label}] 检测到箱子可能脱落（电机力矩相对基线明显下降）")
        print(f"  当前 tau_mean={tau_mean:.2f}Nm  基线={baseline_s}")
        self._recovery_step = RecoverStep.ACK_DROP
        self._waiting_enter = True
        self._enter_prompt = (
            f"⚠ [{label}] 箱子可能已脱落。按 Enter 确认，双手将回到零位...\n"
        )

    def _print_joint_tracking_error(self, target_q: list, label: str) -> None:
        """对比目标关节角与 lowstate 实际值，打印逐关节误差。"""
        if self._low_state is None:
            print(f"[{label}] 无法读取 lowstate，跳过关节误差对比")
            return
        actual_q = self._read_q()
        print(f"\n{'=' * 62}")
        print(f"  [{label}] 目标关节角 vs 实际 state")
        print(f"{'=' * 62}")
        print(f"  {'关节':22s}  {'目标(rad)':>10s}  {'实际(rad)':>10s}  "
              f"{'误差(rad)':>10s}  {'误差(°)':>8s}")
        print(f"  {'-' * 60}")
        max_err = 0.0
        max_joint = ""
        for i, joint in enumerate(ARM_JOINTS):
            tgt = float(target_q[i])
            act = float(actual_q[i])
            err = act - tgt
            if abs(err) > max_err:
                max_err = abs(err)
                max_joint = joint.name
            print(f"  {joint.name:22s}  {tgt:+10.4f}  {act:+10.4f}  "
                  f"{err:+10.4f}  {math.degrees(err):+8.2f}°")
        left_max = max(abs(actual_q[i] - target_q[i]) for i in range(7))
        right_max = max(abs(actual_q[i] - target_q[i]) for i in range(7, 14))
        waist_max = max(abs(actual_q[i] - target_q[i]) for i in range(14, 17))
        print(f"  {'-' * 60}")
        print(f"  最大误差: {math.degrees(max_err):.2f}° ({max_joint})")
        print(f"  左臂 max={math.degrees(left_max):.2f}°  "
              f"右臂 max={math.degrees(right_max):.2f}°  "
              f"腰部 max={math.degrees(waist_max):.2f}°")
        print()

    @staticmethod
    def _lerp(q_from: list, q_to: list, ratio: float) -> list:
        r = float(np.clip(ratio, 0.0, 1.0))
        return [q_from[i] * (1.0 - r) + q_to[i] * r for i in range(len(q_from))]

    def _terminal_q_for_stage(self, stage: Stage) -> list:
        """当前阶段结束时关节目标（与 lerp ratio=1 一致），用于按 Enter 等待期间保持位姿。"""
        if stage == Stage.ZERO:
            return list(ZERO_Q)
        if stage == Stage.ZERO_TO_A:
            return list(self.target_q_a)
        if stage == Stage.A_TO_B:
            return list(self.target_q_b)
        if stage == Stage.B_TO_C:
            return list(self.target_q_c)
        if stage == Stage.C_TO_B:
            return list(self.target_q_b)
        if stage == Stage.B_TO_A:
            return list(self.target_q_a)
        if stage == Stage.A_TO_ZERO:
            return list(ZERO_Q)
        if stage == Stage.WAIT_RL_LOWER:
            return list(ZERO_Q)
        if stage == Stage.BLEND_TO_HANDOFF:
            return list(self.handoff_q)
        if stage == Stage.RELEASE:
            return list(self._release_q_cmd)
        if stage == Stage.RECOVER_TO_ZERO:
            return list(self._hold_q_cmd)
        return list(ZERO_Q)

    def _publish(self, q_cmd: list, sdk_weight: float = 1.0):
        st = self._low_state
        if st is not None:
            self._low_cmd.mode_machine = st.mode_machine
            self._low_cmd.mode_pr = st.mode_pr
        en = self._low_cmd.motor_cmd[int(G1Joint.ArmSdkEnable)]
        en.mode = 1
        en.q = float(sdk_weight)
        en.dq = 0.0
        en.kp = 0.0
        en.kd = 0.0
        en.tau = 0.0
        for i, joint in enumerate(ARM_JOINTS):
            mc = self._low_cmd.motor_cmd[int(joint)]
            mc.mode = 1
            mc.tau = 0.0
            mc.q   = float(q_cmd[i])
            mc.dq  = 0.0
            if i >= 14:
                mc.kp = self.KP_WAIST
                mc.kd = self.KD_WAIST
            elif joint in (G1Joint.LeftShoulderPitch, G1Joint.RightShoulderPitch):
                mc.kp = self.KP_SHOULDER_PITCH
                mc.kd = self.KD_SHOULDER_PITCH
            elif joint in (G1Joint.LeftShoulderRoll, G1Joint.RightShoulderRoll):
                mc.kp = self.KP_SHOULDER_ROLL
                mc.kd = self.KD_SHOULDER_ROLL
            elif joint in (G1Joint.LeftShoulderYaw, G1Joint.RightShoulderYaw):
                mc.kp = self.KP_SHOULDER_YAW
                mc.kd = self.KD_SHOULDER_YAW
            elif joint in (G1Joint.LeftElbow, G1Joint.RightElbow):
                mc.kp = self.KP_ELBOW
                mc.kd = self.KD_ELBOW
            elif joint in (G1Joint.LeftWristRoll, G1Joint.RightWristRoll):
                mc.kp = self.KP_WRIST_ROLL
                mc.kd = self.KD_WRIST_ROLL
            elif joint in (G1Joint.LeftWristPitch, G1Joint.RightWristPitch):
                mc.kp = self.KP_WRIST_PITCH
                mc.kd = self.KD_WRIST_PITCH
            elif joint in (G1Joint.LeftWristYaw, G1Joint.RightWristYaw):
                mc.kp = self.KP_WRIST_YAW
                mc.kd = self.KD_WRIST_YAW
            else:
                mc.kp = self.KP
                mc.kd = self.KD
        self._low_cmd.crc = self._crc.Crc(self._low_cmd)
        self._publisher.Write(self._low_cmd)

    # ── 控制主循环（50Hz 定时回调）────────────────────────────────

    def _control_loop(self):
        if self._abort or self._ctrl_done:
            return

        if self._stage == Stage.DONE:
            if self.end_behavior == "hold_handoff":
                self._publish(self._release_q_cmd, sdk_weight=1.0)
            return

        # 主线程按 Enter 期间：持续发本阶段末端目标，避免 merge 因 arm_sdk 断流切回 RL 上半身
        if self._waiting_enter:
            self._publish(self._hold_q_cmd, sdk_weight=1.0)
            if (
                self._grip_watch_active
                and self._enable_grip_check
                and self._recovery_step == RecoverStep.NONE
            ):
                self._grip_watch_tick()
            return

        dur = self.DURATION.get(self._stage, 0.0)

        # 首次进入本阶段：记录起始角度
        if self._q_stage_start is None:
            self._q_stage_start = self._read_q()

        ratio = self._stage_t / dur if dur > 0 else 1.0

        if self._stage == Stage.ZERO:
            q_cmd = self._lerp(self._q_stage_start, ZERO_Q, ratio)
            self._publish(q_cmd, sdk_weight=1.0)

        elif self._stage == Stage.ZERO_TO_A:
            q_cmd = self._lerp(ZERO_Q, self.target_q_a, ratio)
            self._publish(q_cmd, sdk_weight=1.0)

        elif self._stage == Stage.A_TO_B:
            q_cmd = self._lerp(self.target_q_a, self.target_q_b, ratio)
            self._publish(q_cmd, sdk_weight=1.0)

        elif self._stage == Stage.B_TO_C:
            q_cmd = self._lerp(self.target_q_b, self.target_q_c, ratio)
            self._publish(q_cmd, sdk_weight=1.0)

        elif self._stage == Stage.C_TO_B:
            q_cmd = self._lerp(self.target_q_c, self.target_q_b, ratio)
            self._publish(q_cmd, sdk_weight=1.0)

        elif self._stage == Stage.B_TO_A:
            q_cmd = self._lerp(self.target_q_b, self.target_q_a, ratio)
            self._publish(q_cmd, sdk_weight=1.0)

        elif self._stage == Stage.A_TO_ZERO:
            q_cmd = self._lerp(self.target_q_a, ZERO_Q, ratio)
            self._publish(q_cmd, sdk_weight=1.0)

        elif self._stage == Stage.RECOVER_TO_ZERO:
            q_cmd = self._lerp(self._q_stage_start, ZERO_Q, ratio)
            self._publish(q_cmd, sdk_weight=1.0)

        elif self._stage == Stage.BLEND_TO_HANDOFF:
            q_cmd = self._lerp(self._q_stage_start, self.handoff_q, ratio)
            self._publish(q_cmd, sdk_weight=1.0)

        elif self._stage == Stage.RELEASE:
            self._publish(self._release_q_cmd, sdk_weight=1.0 - ratio)

        # 阶段计时推进
        self._stage_t += self._control_dt
        if self._stage_t >= dur and not self._stage_end_dispatched:
            self._stage_end_dispatched = True
            self._handle_stage_end()

    def _apply_stage_transition(self, next_stage: Stage) -> None:
        self._stage = next_stage
        self._stage_t = 0.0
        self._q_stage_start = None
        self._stage_end_dispatched = False
        if self._stage == Stage.DONE:
            w = 1.0 if self.end_behavior == "hold_handoff" else 0.0
            self._publish(self._release_q_cmd, sdk_weight=w)
            if self.end_behavior != "hold_handoff":
                self.done = True

    def _release_enter_and_advance(self) -> None:
        """由主线程在 input() 返回后调用，与 50Hz 控制线程分离。"""
        self._clear_grip_watch()
        self._grip_drop_detected = False

        if self._recovery_step == RecoverStep.ACK_DROP:
            self._recovery_step = RecoverStep.NONE
            self._hold_q_cmd = list(ZERO_Q)
            print("[重抓] 双手回到零位...")
            self._apply_stage_transition(Stage.RECOVER_TO_ZERO)
            self._waiting_enter = False
            return

        if self._recovery_step == RecoverStep.WAIT_CAPTURE:
            if self._regrasp_callback is None:
                raise RuntimeError("箱子脱落但未提供 regrasp_callback")
            print("[重抓] 拍照并重新计算点A...")
            left_t, right_t = self._regrasp_callback()
            self._update_targets_from_regrasp(left_t, right_t)
            self._recovery_step = RecoverStep.WAIT_GO_A
            self._enter_prompt = (
                "新点A 已计算（见上方坐标）。按 Enter 从零位运动到点A...\n"
            )
            return

        if self._recovery_step == RecoverStep.WAIT_GO_A:
            self._recovery_step = RecoverStep.NONE
            self._waiting_enter = False
            self._apply_stage_transition(Stage.ZERO_TO_A)
            return

        if self._stage == Stage.ZERO_TO_A:
            self._print_joint_tracking_error(self.target_q_a, "点A")
        if self._stage == Stage.WAIT_RL_LOWER:
            if self.end_behavior == "hold_handoff":
                print("开始收至自然下垂...")
            else:
                print("开始对齐 RL_LOWER 接管姿态...")
        self._waiting_enter = False
        self._enter_prompt = "按 Enter 继续..."
        self._apply_stage_transition(self._pending_next_stage)

    def _handle_stage_end(self) -> None:
        if self._stage == Stage.RECOVER_TO_ZERO:
            self._recovery_step = RecoverStep.WAIT_CAPTURE
            self._waiting_enter = True
            self._enter_prompt = "已回到零位。按 Enter 重新拍照计算点A...\n"
            print("[重抓] 已到达零位")
            return

        if self._stage == Stage.A_TO_ZERO:
            if self.end_behavior == "hold_handoff":
                print("[阶段8] 按 Enter 收至自然下垂并保持（不释放 arm_sdk）...")
                self._hold_q_cmd = list(ZERO_Q)
                self._pending_next_stage = Stage.BLEND_TO_HANDOFF
                self._release_q_cmd = list(self.handoff_q)
                self._enter_prompt = "按 Enter 收至自然下垂并保持...\n"
                self._apply_stage_transition(Stage.WAIT_RL_LOWER)
                self._waiting_enter = True
                return
            if self.end_behavior == "handoff_rl_lower":
                print("[阶段8] GR00T-WBC 已在 RL_LOWER（mover.shutdown 已切）。"
                      "按 Enter 把上半身收到中性姿势(全0)并释放交还给 GR00T...")
                self._hold_q_cmd = list(ZERO_Q)
                self._pending_next_stage = Stage.BLEND_TO_HANDOFF
                self._release_q_cmd = list(self.handoff_q)
                self._enter_prompt = "按 Enter 开始收中性并释放交还给 GR00T...\n"
                self._apply_stage_transition(Stage.WAIT_RL_LOWER)
                self._waiting_enter = True
                return

            print("[阶段8] 直接释放 arm_sdk 控制权")
            self._release_q_cmd = list(ZERO_Q)
            self._apply_stage_transition(Stage.RELEASE)
            return

        if self._stage == Stage.BLEND_TO_HANDOFF:
            if self.end_behavior == "hold_handoff":
                print("[完成] 已收至自然下垂，保持 arm_sdk 直至外部接管")
                self._release_q_cmd = list(self.handoff_q)
                self._apply_stage_transition(Stage.DONE)
                return
            print("[阶段9] 释放 arm_sdk 控制权")
            self._release_q_cmd = list(self.handoff_q)
            self._apply_stage_transition(Stage.RELEASE)
            return

        next_stage = Stage(int(self._stage) + 1)

        # stop_after_zero：完成零位后即停止，不进入 ZERO_TO_A
        if self._stop_after_zero and self._stage == Stage.ZERO and next_stage == Stage.ZERO_TO_A:
            print("[  3.0s] 已到达零位，停止。")
            self._publish(ZERO_Q, sdk_weight=1.0)
            self.done = True
            return

        elapsed = sum(
            self.DURATION.get(Stage(s), 0.0) for s in range(int(self._stage) + 1)
        )
        _labels = {
            Stage.ZERO_TO_A:  "阶段2：零位 → 点A",
            Stage.A_TO_B:  "阶段3：点A → 点B",
            Stage.B_TO_C:  "阶段4：点B → 点C",
            Stage.C_TO_B:  "阶段5：点C → 点B",
            Stage.B_TO_A:  "阶段6：点B → 点A",
            Stage.A_TO_ZERO: "阶段7：点A → 收到胸面后 → 零位",
            Stage.WAIT_RL_LOWER: "阶段8：等待切换到 RL_LOWER",
            Stage.BLEND_TO_HANDOFF: "阶段8：对齐 RL_LOWER 接管姿态",
            Stage.RELEASE: "阶段9：释放 arm_sdk 控制权",
            Stage.DONE:    "全部完成",
        }
        print(f"[{elapsed:5.1f}s] {_labels.get(next_stage, '')}")

        if (
            self._enable_grip_check
            and self._stage in _GRIP_CHECK_AFTER_STAGES
            and next_stage in (Stage.B_TO_C, Stage.C_TO_B, Stage.B_TO_A)
        ):
            self._begin_grip_wait_enter(self._stage, next_stage)
            return

        # 这些过渡前需按 Enter：等待在主线程做，此处只挂起并指定保持的关节目标
        if next_stage in (Stage.ZERO_TO_A, Stage.A_TO_B, Stage.B_TO_C, Stage.C_TO_B,
                          Stage.B_TO_A, Stage.A_TO_ZERO):
            self._hold_q_cmd = self._terminal_q_for_stage(self._stage)
            self._pending_next_stage = next_stage
            self._waiting_enter = True
            if self._stage == Stage.ZERO_TO_A and next_stage == Stage.A_TO_B:
                self._enter_prompt = (
                    "点A已到位，按 Enter 读取关节 state 并与点A目标对比后继续...\n"
                )
            else:
                self._enter_prompt = "按 Enter 继续..."
            return

        self._apply_stage_transition(next_stage)

    def abort(self) -> None:
        """请求停止 50Hz 发布（Ctrl+C 等外部中断时调用）。"""
        self._abort = True
        self._waiting_enter = False
        self.done = True
        self._stop_ctrl_thread()
        self._schedule_grip_metrics_save(event="abort")
        if self._publisher is not None and self._state_ready:
            try:
                hold_q = self._read_q()
                for _ in range(25):
                    self._publish(hold_q, sdk_weight=1.0)
                    time.sleep(self._control_dt)
            except Exception:
                pass

    # ── 公开接口 ─────────────────────────────────────────────────

    def _wait_for_enter(self) -> None:
        """等待 Enter；夹持等待期间用非阻塞读，以便脱落时切换提示。"""
        prompt = self._enter_prompt
        if not (self._grip_watch_active and self._enable_grip_check):
            input(prompt)
            return

        fd = sys.stdin.fileno()
        fl = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
        try:
            print(prompt, end="", flush=True)
            while True:
                if self._grip_drop_detected:
                    self._grip_drop_detected = False
                    print(f"\n{self._enter_prompt}", end="", flush=True)
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                if ready:
                    sys.stdin.readline()
                    return
        finally:
            fcntl.fcntl(fd, fcntl.F_SETFL, fl)

    def run(self, release_external_hold=None, handoff_external_hold=None,
            regrasp_callback=None):
        """初始化DDS通信，等待状态就绪，启动50Hz控制线程，阻塞至完成。

        release_external_hold: 可选回调。必须在任何本控制器 arm_sdk 发布之前调用，
        先停止外部自然下垂 keeper，避免 50Hz 双写冲突；随后连发若干帧 hold 位姿
        （与 stage 间 _waiting_enter 保持位姿同理），再启动 50Hz 控制线程。

        handoff_external_hold: end_behavior=hold_handoff 时，在停止本控制器 50Hz 线程之前
        先调用该回调让 keeper 接管 arm_sdk（与 controller 末帧位姿一致），避免 arm_sdk 空窗。

        regrasp_callback: 箱子脱落重抓时调用，应返回新的 (left_target, right_target)
        torso 系点A；由 box_demo 拍照 + SAM3 提供。
        """
        self._regrasp_callback = regrasp_callback
        self._publisher = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self._publisher.Init()
        self._subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self._subscriber.Init(self._on_low_state, 10)

        print("\n等待机器人状态消息...")
        while not self._state_ready:
            time.sleep(0.05)
        print("已收到机器人状态。\n")

        if release_external_hold is not None:
            release_external_hold()

        hold_q = self._read_q()
        for _ in range(25):
            self._publish(hold_q, sdk_weight=1.0)
            time.sleep(self._control_dt)

        if self._start_from_reach:
            print("[  0.0s] 阶段2：零位 → 点A")
        else:
            print("[  0.0s] 阶段1：当前位姿 → 零位")

        self._ctrl_done = False
        self._ctrl_thread = RecurrentThread(
            interval=self._control_dt,
            target=self._control_loop,
            name="dual_arm_ctrl",
        )
        self._ctrl_thread.Start()

        while not self.done:
            if self._waiting_enter:
                self._wait_for_enter()
                self._release_enter_and_advance()
            if self.end_behavior == "hold_handoff" and self._stage == Stage.DONE:
                break
            time.sleep(0.05)

        if self._abort:
            self._stop_ctrl_thread()
            return

        if self.end_behavior == "hold_handoff":
            handoff_q = list(self._release_q_cmd)
            # keeper 先接管（controller 50Hz 仍在发相同 handoff_q），再停 controller 线程
            if handoff_external_hold is not None:
                handoff_external_hold(handoff_q)
                time.sleep(0.12)
            self._stop_ctrl_thread()
        else:
            self._stop_ctrl_thread()

        self._schedule_grip_metrics_save(event="final")


# ─────────────────────────────────────────────────────────────────
# 命令行入口
# ─────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="宇树G1双臂末端位置控制 — 将左右手末端移动到指定坐标点",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python dual_arm_target_reach.py 0.35 0.25 0.45 0.35 -0.25 0.45
  python dual_arm_target_reach.py 0.35 0.25 0.45 0.35 -0.25 0.45 eth0

坐标系说明 (躯干坐标系 torso_link):
  X 轴朝前，Y 轴朝左，Z 轴朝上，单位：米
  零位时末端大约在 [+0.25, ±0.15, +0.04]

典型可达范围:
  左手: x∈[0.10,0.55]  y∈[0.00, 0.50]  z∈[-0.10,0.55]
  右手: x∈[0.10,0.55]  y∈[-0.50,0.00]  z∈[-0.10,0.55]
        """,
    )
    parser.add_argument("lx", type=float, help="左手目标 X (m)")
    parser.add_argument("ly", type=float, help="左手目标 Y (m)")
    parser.add_argument("lz", type=float, help="左手目标 Z (m)")
    parser.add_argument("rx", type=float, help="右手目标 X (m)")
    parser.add_argument("ry", type=float, help="右手目标 Y (m)")
    parser.add_argument("rz", type=float, help="右手目标 Z (m)")
    parser.add_argument(
        "--end-behavior",
        choices=("handoff_rl_lower", "release", "hold_handoff"),
        default="handoff_rl_lower",
        help="结束行为：对齐 RL_LOWER 后释放 / 直接 release / 保持 handoff 不释放",
    )
    parser.add_argument(
        "iface", nargs="?", default=None,
        help="DDS 网络接口（可选，如 eth0）",
    )
    return parser


def main():
    parser = build_parser()
    args   = parser.parse_args()

    left_target  = [args.lx, args.ly, args.lz]
    right_target = [args.rx, args.ry, args.rz]

    print("=" * 60)
    print("  宇树G1 — 双臂末端位置控制")
    print("=" * 60)
    print(f"  左手目标: x={args.lx:+.3f}  y={args.ly:+.3f}  z={args.lz:+.3f}  (m)")
    print(f"  右手目标: x={args.rx:+.3f}  y={args.ry:+.3f}  z={args.rz:+.3f}  (m)")
    print()

    try:
        controller = DualArmController(
            left_target,
            right_target,
            end_behavior=args.end_behavior,
        )
    except ValueError as e:
        print(f"\n[错误] {e}")
        sys.exit(1)

    print("\n警告：运行前请确保机器人周围无障碍物，手臂可以自由活动！")
    input("按 Enter 开始执行，Ctrl+C 取消...\n")

    if args.iface:
        ChannelFactoryInitialize(0, args.iface)
    else:
        ChannelFactoryInitialize(0)

    try:
        controller.run()
    except KeyboardInterrupt:
        print("\n用户中断。")
        sys.exit(0)

    print("程序正常退出。")
    sys.exit(0)


if __name__ == "__main__":
    main()
