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
import os
import sys
import time
import math
import argparse
from enum import IntEnum
from pathlib import Path

import numpy as np

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
                 left_q0: list = None, right_q0: list = None):
        """
        Args:
            left_target, right_target: 左右手目标位置 [x,y,z] (torso 系)
            stop_after_zero: 若 True，完成零位后即停止（用于「先到零位再采图」流程）
            start_from_reach: 若 True，跳过零位阶段，直接从零位→目标开始（假定已在零位）
        """
        if end_behavior not in ("handoff_rl_lower", "release", "hold_handoff"):
            raise ValueError(f"Unsupported end_behavior: {end_behavior}")

        print("正在求解逆运动学...")
        ik = ArmKinematics()
        left_a = np.array(left_target, dtype=float)
        right_a = np.array(right_target, dtype=float)

        # 点A — 目标抓取点（有热启动时减少搜索量）
        a_restarts = 2 if left_q0 is not None else 8
        a_max_iter = 300 if left_q0 is not None else 600
        left_q_a,  left_err  = ik.inverse_kinematics(left_a.tolist(),  left=True,
                                                       q0=left_q0, num_restarts=a_restarts, max_iter=a_max_iter)
        right_q_a, right_err = ik.inverse_kinematics(right_a.tolist(), left=False,
                                                       q0=right_q0, num_restarts=a_restarts, max_iter=a_max_iter)

        left_fk,  _ = ik.forward_kinematics(left_q_a,  left=True)
        right_fk, _ = ik.forward_kinematics(right_q_a, left=False)

        self.left_fk = left_fk   # FK 末端位置 [x,y,z]，供 CSV 等使用
        self.right_fk = right_fk

        print(f"  左臂 IK 误差: {left_err*1000:.2f} mm  "
              f"FK验证: {left_fk.round(3).tolist()}")
        print(f"  右臂 IK 误差: {right_err*1000:.2f} mm  "
              f"FK验证: {right_fk.round(3).tolist()}")

        if left_err > self.IK_ERR_LIMIT or right_err > self.IK_ERR_LIMIT:
            raise ValueError(
                f"IK误差过大 (左:{left_err*1000:.1f}mm, 右:{right_err*1000:.1f}mm)，"
                "目标点超出机械臂工作空间，请将机器人移近箱子后重试。"
            )
        if left_err > self.IK_WARN_LIMIT or right_err > self.IK_WARN_LIMIT:
            print(f"  [警告] IK误差偏大 (左:{left_err*1000:.1f}mm, 右:{right_err*1000:.1f}mm)")
            print(f"  [警告] 目标点在工作空间边界附近，手臂将尽力靠近但可能无法完全到达。")
            print(f"  [建议] 将机器人向前移约 {max(left_err,right_err)*100:.0f}cm 可获得更好效果。")

        # 点B：A基础上 y向内5cm — 热启动，A解已知
        left_b = left_a.copy()
        right_b = right_a.copy()
        left_b[1] -= 0.10
        right_b[1] += 0.10

        left_q_b,  _ = ik.inverse_kinematics(left_b.tolist(),  left=True,
                                               q0=left_q_a, num_restarts=2, max_iter=200)
        right_q_b, _ = ik.inverse_kinematics(right_b.tolist(), left=False,
                                               q0=right_q_a, num_restarts=2, max_iter=200)

        # 点C：B基础上 x-15cm, z+10cm — 热启动，B解已知
        left_c = left_b.copy()
        right_c = right_b.copy()
        # left_c[0] -= 0.15
        # right_c[0] -= 0.15
        left_c[0] -= 0.05
        right_c[0] -= 0.05
        left_c[2] += 0.1
        right_c[2] += 0.1
        left_q_c,  _ = ik.inverse_kinematics(left_c.tolist(),  left=True,
                                               q0=left_q_b, num_restarts=2, max_iter=200)
        right_q_c, _ = ik.inverse_kinematics(right_c.tolist(), left=False,
                                               q0=right_q_b, num_restarts=2, max_iter=200)

        waist = [ZERO_Q[14], ZERO_Q[15], ZERO_Q[16]]
        self.target_q_a = left_q_a + right_q_a + waist

        self.target_q_b = left_q_b + right_q_b + waist
        self.target_q_c = left_q_c + right_q_c + waist
        self.target_q: list = self.target_q_a  # 默认/CSV 用点A

        self._stop_after_zero = stop_after_zero
        self._start_from_reach = start_from_reach
        self.end_behavior = end_behavior
        self.handoff_q = list(RL_LOWER_HANDOFF_Q)

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

    # ── DDS 回调 ─────────────────────────────────────────────────

    def _on_low_state(self, msg: LowState_):
        self._low_state = msg
        if not self._state_ready:
            self._state_ready = True

    # ── 工具函数 ─────────────────────────────────────────────────

    def _read_q(self) -> list:
        return [self._low_state.motor_state[int(j)].q for j in ARM_JOINTS]

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
        if self._abort or self._stage == Stage.DONE:
            return

        # 主线程按 Enter 期间：持续发本阶段末端目标，避免 merge 因 arm_sdk 断流切回 RL 上半身
        if self._waiting_enter:
            self._publish(self._hold_q_cmd, sdk_weight=1.0)
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

        elif self._stage == Stage.BLEND_TO_HANDOFF:
            q_cmd = self._lerp(ZERO_Q, self.handoff_q, ratio)
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
            self.done = True

    def _release_enter_and_advance(self) -> None:
        """由主线程在 input() 返回后调用，与 50Hz 控制线程分离。"""
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
        if self._publisher is not None and self._state_ready:
            try:
                hold_q = self._read_q()
                for _ in range(25):
                    self._publish(hold_q, sdk_weight=1.0)
                    time.sleep(self._control_dt)
            except Exception:
                pass

    # ── 公开接口 ─────────────────────────────────────────────────

    def run(self, release_external_hold=None, handoff_external_hold=None):
        """初始化DDS通信，等待状态就绪，启动50Hz控制线程，阻塞至完成。

        release_external_hold: 可选回调。在 50Hz 线程启动前，本控制器会先连发
        若干帧 arm_sdk（与 stage 间 _waiting_enter 保持位姿同理），再调用该回调
        停止外部自然下垂 keeper，避免 arm_sdk 空窗导致 merge 回退到 policy。

        handoff_external_hold: end_behavior=hold_handoff 时，BLEND 完成后先连发
        若干帧 handoff 位姿，再调用该回调启动外部 keeper，避免 RELEASE 淡出导致
        policy 抢回上半身。
        """
        self._publisher = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self._publisher.Init()
        self._subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self._subscriber.Init(self._on_low_state, 10)

        print("\n等待机器人状态消息...")
        while not self._state_ready:
            time.sleep(0.05)
        print("已收到机器人状态。\n")

        hold_q = self._read_q()
        for _ in range(25):
            self._publish(hold_q, sdk_weight=1.0)
            time.sleep(self._control_dt)
        if release_external_hold is not None:
            release_external_hold()

        if self._start_from_reach:
            print("[  0.0s] 阶段2：零位 → 点A")
        else:
            print("[  0.0s] 阶段1：当前位姿 → 零位")

        ctrl_thread = RecurrentThread(
            interval=self._control_dt,
            target=self._control_loop,
            name="dual_arm_ctrl",
        )
        ctrl_thread.Start()

        while not self.done:
            if self._waiting_enter:
                input(self._enter_prompt)
                self._release_enter_and_advance()
            time.sleep(0.05)

        if self._abort:
            return

        if self.end_behavior == "hold_handoff":
            handoff_q = list(self._release_q_cmd)
            for _ in range(25):
                self._publish(handoff_q, sdk_weight=1.0)
                time.sleep(self._control_dt)
            if handoff_external_hold is not None:
                handoff_external_hold()


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
