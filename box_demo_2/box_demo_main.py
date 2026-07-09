#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
box_demo_2 主入口 — 货架前搜索 + 抓取

流程：
  0. 释放相机（停 videohub / RealSense 预热），再双臂自然下垂（3s 插值 + 50Hz 保持），按 Enter 继续
  1. SAM3 mask 边界对齐 → Y 中心对齐 → 点A X 对齐 → IK 检查 → 双臂抓取（任一步底盘移动后进入下一次 attempt）
  2. 抓取后（或本轮 attempt 用尽）保持自然下垂，按 Enter 开始新一轮抓取；Ctrl+C 退出
"""

# `import ssl` MUST come first: cv2 / unitree_sdk2py(cyclonedds) pull in the
# SYSTEM libcrypto, after which _ssl binds to the wrong OpenSSL and import openai
# /httpx fails (ImportError: libcrypto.so.3: version OPENSSL_3.3.0 not found).
# Loading ssl first binds _ssl to the conda env's libcrypto and caches it.
import ssl  # noqa: E402,F401  (must precede cv2 / unitree_sdk2py)
import argparse
import csv
import ctypes
import math
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

# unitree_sdk2 第三方 native 库；须在 cv2 / unitree_sdk2py 之前写入 LD_LIBRARY_PATH
_UNITREE_SDK2_LIB = Path.home() / "unitree_sdk2-main/thirdparty/lib/aarch64"
if _UNITREE_SDK2_LIB.is_dir():
    _lib_path = str(_UNITREE_SDK2_LIB)
    _existing = os.environ.get("LD_LIBRARY_PATH", "")
    if _lib_path not in _existing.split(":"):
        os.environ["LD_LIBRARY_PATH"] = (
            f"{_lib_path}:{_existing}" if _existing else _lib_path
        )

import cv2
import numpy as np

# ── 代理隔离：本机服务 (SAM3 / 相机) 绝不走系统 socks/http 代理 ──────────────
# ~/.bashrc 导出了 ALL_PROXY=socks5://127.0.0.1:7897 (clash)。tmux 交互 shell 会
# source .bashrc，导致 requests (SAM3/相机) 把 localhost 流量也塞进代理。
# 这里强制 localhost 直连。
_no_proxy = set(filter(None, os.environ.get("no_proxy", "").split(",")))
_no_proxy.update(["127.0.0.1", "localhost", "0.0.0.0", "::1"])
os.environ["no_proxy"] = os.environ["NO_PROXY"] = ",".join(sorted(_no_proxy))

# ─────────────────────────────────────────────────────────────────
# 本地模块导入（同目录）
# ─────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
import fcntl
import termios


def _flush_stdin():
    """清空 stdin 缓冲区，避免残留按键（如行走期间误触 Enter）被后续 input() 吃掉。"""
    fd = sys.stdin.fileno()
    fl = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
    try:
        while sys.stdin.read(1):
            pass
    except (TypeError, OSError):
        pass
    finally:
        fcntl.fcntl(fd, fcntl.F_SETFL, fl)

# 须先于 unitree_sdk2py / cyclonedds：避免 LD_LIBRARY_PATH 先加载错误 libddsc（与 RoboJuDo run_pipeline 一致）
_ddsc = Path(os.environ.get("CYCLONEDDS_HOME", Path.home() / "cyclonedds-0.10-install")) / "lib" / "libddsc.so.0"
if _ddsc.is_file():
    ctypes.CDLL(str(_ddsc), mode=ctypes.RTLD_GLOBAL)

from capture_and_predict import CameraSession, predict_from_arrays
from vision_session_log import VisionSessionLog
from dual_arm_target_reach import DualArmController, ARM_JOINTS, ArmKinematics
from arm_natural_hang import (
    ArmNaturalHangKeeper,
    make_handoff_callback,
    make_release_callback,
    read_arm_q_from_state,
)
from robot_move import RobotMover  # AGILE base (default / fallback)
try:
    from groot_mover import GrootMover  # GR00T-WBC base (writes units=agile)
except Exception as _groot_import_err:  # pragma: no cover
    GrootMover = None
try:
    from remote_mover import RemoteMover  # GR00T-WBC base over HTTP (box_demo on robot)
except Exception as _remote_import_err:  # pragma: no cover
    RemoteMover = None

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_

GOAL_CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "goal.csv")


# ─────────────────────────────────────────────────────────────────
# 相机外参（来自 URDF d435_joint，固定不变）
# ─────────────────────────────────────────────────────────────────

# d435_joint origin: xyz (m), 相对于 torso_link
_T_TRANS = np.array([0.0576235, 0.01753, 0.42987])

# d435_joint origin: rpy = [0, pitch, 0]，纯 Y 轴旋转
_PITCH = 0.8307767239493009   # rad ≈ 47.6°
_cp, _sp = math.cos(_PITCH), math.sin(_PITCH)

# R_torso_d435_body = Ry(pitch)
_R_TORSO_BODY = np.array([
    [ _cp,  0,  _sp],
    [ 0,    1,  0  ],
    [-_sp,  0,  _cp],
])

# 光学坐标系 → d435 body 坐标系的固定旋转
#   optical: Z=深度/前, X=右, Y=下
#   body:    X=深度/前, Y=左, Z=上
# 对应关系: X_body = Z_optical, Y_body = -X_optical, Z_body = -Y_optical
_R_BODY_OPTICAL = np.array([
    [ 0,  0,  1],
    [-1,  0,  0],
    [ 0, -1,  0],
])

# 总旋转矩阵：从光学坐标系直接到 torso_link
_R_TORSO_OPT = _R_TORSO_BODY @ _R_BODY_OPTICAL


def cam_to_torso(p_cam: list) -> np.ndarray:
    """将相机光学坐标系下的三维点转换到 torso_link 坐标系。"""
    p = np.asarray(p_cam, dtype=float)
    return _R_TORSO_OPT @ p + _T_TRANS


def grasp_to_point_a_torso(grasp_left_cam, grasp_right_cam) -> tuple[np.ndarray, np.ndarray]:
    """SAM3 抓取点 + 手长补偿 → 左右点A（torso 系）。"""
    left_torso = cam_to_torso(grasp_left_cam)
    right_torso = cam_to_torso(grasp_right_cam)
    left_torso[0] -= 0.07
    left_torso[1] += 0.09
    left_torso[2] += 0.04
    right_torso[0] -= 0.07
    right_torso[1] -= 0.09
    right_torso[2] += 0.04
    return left_torso, right_torso


def save_goal_to_csv(left_torso, right_torso, target_q, left_fk, right_fk, need_header: list) -> None:
    """将目标值追加写入 goal.csv（转置格式：第一列为变量名，后续列为各次数据）"""
    names = [
        "timestamp",
        "left_target_x", "left_target_y", "left_target_z",
        "right_target_x", "right_target_y", "right_target_z",
        "left_fk_x", "left_fk_y", "left_fk_z",
        "right_fk_x", "right_fk_y", "right_fk_z",
    ] + [f"{joint.name}_q" for joint in ARM_JOINTS]
    new_col = [
        datetime.now().isoformat(),
        float(left_torso[0]), float(left_torso[1]), float(left_torso[2]),
        float(right_torso[0]), float(right_torso[1]), float(right_torso[2]),
        float(left_fk[0]), float(left_fk[1]), float(left_fk[2]),
        float(right_fk[0]), float(right_fk[1]), float(right_fk[2]),
    ] + [float(q) for q in target_q]
    if need_header[0]:
        rows = [[names[i], new_col[i]] for i in range(len(names))]
        need_header[0] = False
    else:
        with open(GOAL_CSV_PATH, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
        if len(rows) == len(names) and len(rows[0]) > 0:
            for i in range(len(names)):
                rows[i].append(str(new_col[i]))
        else:
            rows = [[names[i], new_col[i]] for i in range(len(names))]
    with open(GOAL_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)


# ─────────────────────────────────────────────────────────────────
# WaistRotator: 上半身旋转控制（通过 rt/arm_sdk，下半身由 RL_LOWER 接管）
# ─────────────────────────────────────────────────────────────────

class WaistRotator:
    """转腰 / 行走前锁腰。有 arm_keeper 时由 keeper 发 arm_sdk（双臂自然下垂）。"""

    KP_WAIST = 20.0
    KD_WAIST = 1.5
    KP_ARM = 40.0
    KD_ARM = 2.0
    ROTATE_DURATION = 3.5
    ROTATE_INTERVAL = 0.05
    KEEPALIVE_INTERVAL = 0.05
    SETTLE_TOLERANCE = 0.02
    SETTLE_TIMEOUT = 1.0

    def __init__(self, arm_keeper: ArmNaturalHangKeeper | None = None):
        self._keeper = arm_keeper
        self._pub = None
        self._sub = None
        self._low_state = None
        self._arm_q = None
        self._ready = threading.Event()
        self.yaw = 0.0
        self._hold = False
        self._hold_target = 0.0
        self._thread = None

    def initialize(self):
        if self._keeper is not None:
            self._keeper.sync_waist_from_robot()
            self.yaw = self._keeper._waist_yaw
            print("  腰部旋转器已就绪（双臂自然下垂 + rt/arm_sdk）")
            return
        if self._pub is not None:
            return
        self._pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self._pub.Init()
        self._sub = ChannelSubscriber("rt/lowstate", LowState_)
        self._sub.Init(self._on_state, 10)
        self._ready.wait(timeout=1.0)
        print("  腰部旋转器已就绪 (RL_LOWER + rt/arm_sdk)")

    def rotate(self, angle_rad: float):
        if abs(angle_rad) < 0.04:
            return
        start_yaw = self.yaw
        target = start_yaw + angle_rad
        direction = "左" if angle_rad > 0 else "右"
        print(f"  腰部旋转: 往{direction}转 {abs(math.degrees(angle_rad)):.0f}° "
              f"(累计{math.degrees(target):.0f}°)")
        if self._keeper is not None:
            self._keeper.set_pass_waist_from_state(False)
            self._keeper._ensure_dds()
            t0 = time.time()
            while True:
                elapsed = time.time() - t0
                if elapsed >= self.ROTATE_DURATION:
                    break
                progress = 0.5 * (1 - math.cos(math.pi * elapsed / self.ROTATE_DURATION))
                self._keeper.set_waist_yaw(start_yaw + angle_rad * progress)
                self._keeper.publish_once()
                time.sleep(self.ROTATE_INTERVAL)
            self._keeper.set_waist_yaw(target)
            self._keeper.publish_once()
            self.yaw = target
            if not self._keeper.is_running():
                self._keeper.start_keepalive()
            self._wait_settle_keeper(target)
            return
        self.initialize()
        t0 = time.time()
        while True:
            elapsed = time.time() - t0
            if elapsed >= self.ROTATE_DURATION:
                break
            progress = 0.5 * (1 - math.cos(math.pi * elapsed / self.ROTATE_DURATION))
            self._publish_once(start_yaw + angle_rad * progress)
            time.sleep(self.ROTATE_INTERVAL)
        self._publish_once(target)
        self._start_hold(target)
        self.yaw = target
        self._wait_settle(target)

    def stop_hold(self):
        """行走前锁腰：roll/pitch=0，yaw 取实测快照；keeper 模式不停自然下垂 keepalive。"""
        if self._keeper is not None:
            self._keeper.set_pass_waist_from_state(False)
            self._keeper.sync_waist_from_robot()
            self.yaw = self._keeper._waist_yaw
            return
        self._hold = False

    def sync_yaw(self):
        if self._keeper is not None:
            self._keeper.sync_waist_from_robot()
            self.yaw = self._keeper._waist_yaw
            return
        self.initialize()
        time.sleep(0.1)
        if self._low_state is not None:
            self.yaw = float(self._low_state.motor_state[12].q)
        else:
            self.yaw = 0.0

    # ── private ──────────────────────────────────────────────────

    def _on_state(self, msg):
        self._low_state = msg
        if self._arm_q is None:
            self._arm_q = [float(msg.motor_state[i].q) for i in range(12, 29)]
        if not self._ready.is_set():
            self._ready.set()

    def _wait_settle(self, target_yaw):
        """Wait until actual waist yaw converges to target within tolerance."""
        t0 = time.time()
        while time.time() - t0 < self.SETTLE_TIMEOUT:
            if self._low_state is not None:
                actual = float(self._low_state.motor_state[12].q)
                if abs(actual - target_yaw) < self.SETTLE_TOLERANCE:
                    return
            time.sleep(0.02)

    def _wait_settle_keeper(self, target_yaw: float) -> None:
        t0 = time.time()
        while time.time() - t0 < self.SETTLE_TIMEOUT:
            if self._keeper is not None and self._keeper.low_state is not None:
                actual = float(self._keeper.low_state.motor_state[12].q)
                if abs(actual - target_yaw) < self.SETTLE_TOLERANCE:
                    return
            time.sleep(0.02)

    def _publish_once(self, target_yaw: float):
        """Publish one arm_sdk frame: waist at target, arms hold position."""
        if self._pub is None or self._arm_q is None:
            return
        cmd = unitree_hg_msg_dds__LowCmd_()
        if self._low_state is not None:
            cmd.mode_machine = self._low_state.mode_machine
            cmd.mode_pr = self._low_state.mode_pr
        # arm_sdk enable flag
        en = cmd.motor_cmd[29]
        en.mode = 1; en.q = 1.0; en.kp = 0.0; en.kd = 0.0; en.tau = 0.0
        # waist joints (12=yaw, 13=roll, 14=pitch)
        for mi, tgt in zip((12, 13, 14), (target_yaw, 0.0, 0.0)):
            mc = cmd.motor_cmd[mi]
            mc.mode = 1; mc.tau = 0.0; mc.dq = 0.0
            mc.q = float(tgt); mc.kp = self.KP_WAIST; mc.kd = self.KD_WAIST
        # arm joints (15-28): hold current position
        for mi in range(15, 29):
            mc = cmd.motor_cmd[mi]
            mc.mode = 1; mc.tau = 0.0; mc.dq = 0.0
            mc.q = self._arm_q[mi - 12]; mc.kp = self.KP_ARM; mc.kd = self.KD_ARM
        self._pub.Write(cmd)

    def _start_hold(self, target_yaw: float):
        """Start or update background keepalive thread."""
        self._hold_target = target_yaw
        if self._thread is not None and self._thread.is_alive():
            # thread already running: just update target, no gap
            return
        self._hold = True
        self._thread = threading.Thread(target=self._keepalive_loop, daemon=True)
        self._thread.start()

    def _keepalive_loop(self):
        """Background loop: publish arm_sdk to prevent waist drift-back."""
        while self._hold:
            self._publish_once(self._hold_target)
            time.sleep(self.KEEPALIVE_INTERVAL)


# ─────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="box_demo: 采集→预测→坐标转换→双臂抓取")
    parser.add_argument("--prompt",     default="box",             help="SAM3 提示词，默认 box")
    parser.add_argument("--point-cloud-stage", required=True,
                        choices=("sor", "sor_dbscan"),
                        help="SAM3 OBB 点云阶段：sor=仅 SOR，sor_dbscan=SOR+DBSCAN")
    parser.add_argument("--host",       default="192.168.112.198", help="SAM3 服务端 IP")
    parser.add_argument("--port",       default=5300, type=int,    help="SAM3 服务端端口")
    parser.add_argument("--iface",      default=None,              help="DDS 网络接口，如 eth0")
    parser.add_argument("--no-confirm", action="store_true",       help="跳过执行前确认提示")
    parser.add_argument("--locomotion", choices=("agile", "groot", "remote"), default="agile",
                        help="运控底座: agile=AGILE pipeline(rt/lowcmd_rl 归一化), "
                             "groot=GR00T-WBC adapter 本机文件 IPC(units=agile 物理 m/s), "
                             "remote=box_demo 跑在机器人上, 经 HTTP 把命令转给 5080 的 GR00T 底座。"
                             "必须与启动的 adapter 一致。")
    parser.add_argument("--ipc-url", default="http://127.0.0.1:5001",
                        help="--locomotion remote 时, 5080 上 agile_http_ipc_server 的地址")
    # 调小步前进的旋钮(groot/remote 底座)。见 FORWARD_STEP_TUNING.md 的 MuJoCo 实测。
    parser.add_argument("--min-duration", type=float, default=1.5,
                        help="选项A: 一次前进/横移/转向命令的最短执行时间(s)。实测 T<=1.2s 最差摆动相位仍会后退, T>=1.5s(vx>=0.15) 才可靠净前进; 大移动不受影响。")
    parser.add_argument("--min-distance", type=float, default=0.08,
                        help="选项B: 最小线性增量(m)。把过小的前进/横移顶到此值。实测可靠最小增量~8cm。0=关。")
    parser.add_argument("--warmup-time", type=float, default=0.6,
                        help="选项C(治后退的主手段): 移动前预热步态的时长(s)。sim2sim 实测 0.6s 让跨所有摆动相位的最差净位移严格>0(+3.9cm, 绝不后退)。0=关。会让每步变成~10cm+。")
    parser.add_argument("--warmup-speed", type=float, default=0.15,
                        help="预热速度(m/s), 配合 --warmup-time。")
    parser.add_argument("--settle-before", type=float, default=0.0,
                        help="移动前原地 Balance 这么久(s)。反直觉: 摆动是周期的, 固定 settle 反而可能停在坏相位, sim2sim 里 settle=0.5 每次都比 0 差。默认关, 用 warmup 代替。")
    parser.add_argument("--walk-min-height", type=float, default=0.72,
                        help="走路最低高度(m)。实测低于~0.70m Walk 崩掉(0.68 只 3cm), 0.72 留一档余量。低于此发走路命令会告警。")
    parser.add_argument("--auto-raise-walk", action="store_true",
                        help="高度低于 --walk-min-height 时, 移动前自动抬高到该高度再走。")
    parser.add_argument("--dist-gain", type=float, default=1.0,
                        help="选项D: 开环距离补偿(实际≈命令×55~60%%)。设~1.7 让单次开环更准; 1.0=关(靠闭环重感知纠)。")
    parser.add_argument(
        "--end-behavior",
        choices=("handoff_rl_lower", "release", "hold_handoff"),
        default="handoff_rl_lower",
        help="抓取结束行为；启用自然下垂时 box_demo 自动用 hold_handoff",
    )
    parser.add_argument("--skip-arm-init", action="store_true",
                        help="跳过启动时双臂自然下垂与平滑交还 policy")
    parser.add_argument("--max-attempts", default=7, type=int,
                        help="单次抓取的最大尝试次数（每次底盘移动后计为新的一次）")
    parser.add_argument("--walk-velocity", default=0.7, type=float,
                        help="行走速度 m/s (默认 0.7，匹配键盘 w 的速度，仅 AGILE 底座)")
    parser.add_argument("--walk-scale",    default=1.0, type=float,
                        help="行走距离补偿系数 (>1 走更远，补偿 RL 欠追踪)")
    parser.add_argument("--camera-url",    default=os.environ.get("CAMERA_URL", ""),
                        help="网络 RGBD 相机 /capture URL；留空则使用本机 USB 相机")
    parser.add_argument("--camera-serial", default=os.environ.get("HEAD_CAMERA_SERIAL", "406122070550"),
                        help="RealSense 序列号")
    parser.add_argument("--no-grip-check", action="store_true",
                        help="关闭阶段3/4/5结束后的箱子夹持检测与自动重抓")
    parser.add_argument("--vision-log-dir", default=None,
                        help="视觉记录根目录，默认 img/runs")
    parser.add_argument("--no-vision-log", action="store_true",
                        help="关闭视觉记录（不保存 img/runs）")
    args = parser.parse_args()

    print("=" * 65)
    print("  box_demo: SAM3 感知 + 双臂抓取")
    print(f"  SAM3 点云阶段: {args.point_cloud_stage}")
    print("=" * 65)

    arm_keeper = None if args.skip_arm_init else ArmNaturalHangKeeper()
    waist = WaistRotator(arm_keeper)

    shutdown_done = False
    active_controller: DualArmController | None = None

    def _graceful_shutdown(controller=None, interrupted: bool = False) -> None:
        nonlocal shutdown_done
        if shutdown_done:
            return
        shutdown_done = True
        if interrupted:
            print("\n[退出] Ctrl+C — 平滑交还 policy...")
        else:
            print("\n[退出] 平滑交还 policy...")
        if controller is not None:
            controller.abort()
        q_start = None
        if arm_keeper is not None:
            q_start = read_arm_q_from_state(arm_keeper)
        try:
            mover.stop()
        except Exception:
            pass
        print("[退出] 切换 RL_FULL...")
        try:
            mover.initialize()
            time.sleep(0.3)
        except Exception as exc:
            print(f"  警告: mover.initialize 失败: {exc}")
        if arm_keeper is None:
            waist._arm_q = None
        else:
            arm_keeper.release_to_policy(q_start=q_start)
        print("box_demo 结束。")

    # ── 辅助函数 ────────────────────────────────────────────────
    def _walk(dist_x: float, dist_y: float):
        """控制机器人移动（前后 + 横向，转向交给腰部）。"""
        if abs(dist_x) < 0.03 and abs(dist_y) < 0.03:
            return
        waist.stop_hold()  # 行走前锁腰；自然下垂 keeper 保持运行
        moved = False
        if abs(dist_x) >= 0.03:
            scaled = dist_x * args.walk_scale
            tag = f" (补偿×{args.walk_scale})" if args.walk_scale != 1.0 else ""
            direction = "后退" if dist_x < 0 else "直走"
            print(f"  {direction}: {abs(dist_x)*100:.0f}cm{tag}")
            mover.move_forward(scaled)
            moved = True
        if abs(dist_y) >= 0.03:
            side = "左移" if dist_y > 0 else "右移"
            print(f"  {side}: {abs(dist_y)*100:.0f}cm")
            mover.move_left(dist_y)
            moved = True
        if moved:
            time.sleep(1.0)  # 等 RL 策略执行完毕、机器人稳定后再进行下一步

    # ── 位置调整 / IK 参数 ─────────────────────────────────────────
    ADJUST_STEP = 0.05
    Y_CENTER_LIMIT = 0.10
    X_POINT_A_CLOSE = 0.10   # 点A 任意 x 低于此 → 后退
    X_POINT_A_FAR = 0.43     # 点A 任意 x 高于此 → 前进
    MAX_SAM3_RETRY = 30
    _ik_solver = ArmKinematics()
    IK_SOLVE_RESTARTS = 8
    IK_SOLVE_MAX_ITER = 600
    IK_WARN_LIMIT = DualArmController.IK_WARN_LIMIT       # 20mm
    IK_SEVERE_LIMIT = 0.060                               # 60mm 严重警告
    IK_UNSOLVABLE_LIMIT = DualArmController.IK_ERR_LIMIT  # 100mm 算不出来

    def _hang_here(reason: str) -> None:
        print(f"\n[暂停] {reason}")
        print("  进程停在此步（不退出）。按 Ctrl+C 可中断。")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            _graceful_shutdown(interrupted=True)
            raise SystemExit(0)

    def _solve_ik_pair(left_torso, right_torso, left_q0=None, right_q0=None):
        restarts = 2 if left_q0 is not None else IK_SOLVE_RESTARTS
        max_iter = 300 if left_q0 is not None else IK_SOLVE_MAX_ITER
        left_q, left_err = _ik_solver.inverse_kinematics(
            left_torso.tolist(), left=True, q0=left_q0,
            num_restarts=restarts, max_iter=max_iter)
        right_q, right_err = _ik_solver.inverse_kinematics(
            right_torso.tolist(), left=False, q0=right_q0,
            num_restarts=restarts, max_iter=max_iter)
        return left_q, left_err, right_q, right_err

    def _eval_ik(left_torso, right_torso):
        """IK 求解 + 警告/严重警告复检。返回 (ok, left_q, right_q, action, left_err, right_err)。"""
        left_q, left_err, right_q, right_err = _solve_ik_pair(left_torso, right_torso)
        max_err = max(left_err, right_err)
        print(f"  IK 误差: 左={left_err*1000:.1f}mm  右={right_err*1000:.1f}mm")

        if max_err >= IK_UNSOLVABLE_LIMIT:
            print(f"  [IK] 超出工作空间 (≥{IK_UNSOLVABLE_LIMIT*1000:.0f}mm)，需前进")
            return False, left_q, right_q, "forward", left_err, right_err

        if max_err > IK_SEVERE_LIMIT:
            print(f"  [严重警告] 误差 > {IK_SEVERE_LIMIT*1000:.0f}mm，重算一次...")
            left_q, left_err, right_q, right_err = _solve_ik_pair(
                left_torso, right_torso, left_q0=left_q, right_q0=right_q)
            max_err = max(left_err, right_err)
            print(f"  [严重警告] 重算后: 左={left_err*1000:.1f}mm  右={right_err*1000:.1f}mm")
            if max_err > IK_SEVERE_LIMIT:
                return False, left_q, right_q, "hang", left_err, right_err

        if max_err > IK_WARN_LIMIT:
            print(f"  [警告] 误差 > {IK_WARN_LIMIT*1000:.0f}mm，重算一次...")
            left_q, left_err, right_q, right_err = _solve_ik_pair(
                left_torso, right_torso, left_q0=left_q, right_q0=right_q)
            max_err = max(left_err, right_err)
            print(f"  [警告] 重算后: 左={left_err*1000:.1f}mm  右={right_err*1000:.1f}mm")
            if max_err > IK_WARN_LIMIT:
                print(f"  [警告] 仍超出 {IK_WARN_LIMIT*1000:.0f}mm，继续执行")

        return True, left_q, right_q, None, left_err, right_err

    def _log_ik_forward_retry(stage, color, depth, meta, attempt,
                                left_torso, right_torso, left_err, right_err, reason):
        """IK 不可达、需前进重试时，保存当前帧与分析上下文。"""
        vision_log.record(
            stage, color, depth, sam3=meta,
            extra={
                "attempt": attempt,
                "reason": reason,
                "left_torso": [float(v) for v in left_torso],
                "right_torso": [float(v) for v in right_torso],
                "left_ik_err_mm": round(float(left_err) * 1000, 1),
                "right_ik_err_mm": round(float(right_err) * 1000, 1),
                "next_move": "forward_5cm",
            },
        )

    def _log_y_align_adjust(stage, color, depth, meta, attempt,
                            box_center_torso, lateral, reason, next_move):
        """箱子中心 Y 超范围需横移时，保存当前帧与分析上下文。"""
        vision_log.record(
            stage, color, depth, sam3=meta,
            extra={
                "attempt": attempt,
                "reason": reason,
                "box_center_torso": [float(v) for v in box_center_torso],
                "box_center_y": round(float(lateral), 4),
                "y_center_limit": Y_CENTER_LIMIT,
                "next_move": next_move,
            },
        )

    def _log_x_align_adjust(stage, color, depth, meta, attempt,
                            left_torso, right_torso, reason, next_move):
        """点A X 超范围需底盘调整时，保存当前帧与分析上下文。"""
        vision_log.record(
            stage, color, depth, sam3=meta,
            extra={
                "attempt": attempt,
                "reason": reason,
                "left_torso": [float(v) for v in left_torso],
                "right_torso": [float(v) for v in right_torso],
                "left_x": round(float(left_torso[0]), 4),
                "right_x": round(float(right_torso[0]), 4),
                "x_close_limit": X_POINT_A_CLOSE,
                "x_far_limit": X_POINT_A_FAR,
                "next_move": next_move,
            },
        )

    # ── DDS 初始化 ──────────────────────────
    print("\n[感知] SAM3 检测")
    print("  初始化 DDS 通信 (用于 RL Pipeline 行走控制)...")
    if args.iface:
        ChannelFactoryInitialize(0, args.iface)
    else:
        ChannelFactoryInitialize(0)
    # 小步前进旋钮(只对 groot/remote 底座生效)。见 FORWARD_STEP_TUNING.md
    _mover_kw = dict(min_duration=args.min_duration, min_distance=args.min_distance,
                     warmup_time=args.warmup_time, warmup_speed=args.warmup_speed,
                     settle_before_s=args.settle_before,
                     walk_min_height=args.walk_min_height,
                     auto_raise_for_walk=args.auto_raise_walk, dist_gain=args.dist_gain)
    _knob_str = (f"min_dur={args.min_duration}s min_dist={args.min_distance}m "
                 f"warmup={args.warmup_time}s settle={args.settle_before}s "
                 f"walk_h>={args.walk_min_height}m dist_gain={args.dist_gain}")
    if args.locomotion == "remote":
        if RemoteMover is None:
            parser.error("--locomotion remote 但 remote_mover 不可用（导入失败）")
        # box_demo runs on the robot; locomotion commands relay over HTTP to the
        # GR00T base on the 5080 (agile_http_ipc_server --backend legacy-ipc).
        mover = RemoteMover(args.ipc_url, **_mover_kw)
        print(f"[locomotion] GR00T-WBC base over HTTP -> {args.ipc_url} ({_knob_str})")
    elif args.locomotion == "groot":
        if GrootMover is None:
            parser.error("--locomotion groot 但 groot_mover 不可用（导入失败）")
        # GR00T uses physical m/s with a velocity floor above the 0.05 Walk
        # threshold; its own conservative cruise is used (the AGILE-tuned
        # --walk-velocity 0.7 is a normalized value and does not apply here).
        mover = GrootMover(**_mover_kw)
        print(f"[locomotion] GR00T-WBC base (groot_mover, units=agile, physical m/s) ({_knob_str})")
    else:
        mover = RobotMover(velocity=args.walk_velocity)
        print("[locomotion] AGILE base (robot_move, normalized)")
    mover.initialize()
    time.sleep(0.5)

    camera = CameraSession(camera_url=args.camera_url, serial=args.camera_serial)
    try:
        # 先完成 videohub 释放 + RealSense 冷启动，再启动 arm_sdk 自然下垂，
        # 避免 stop_videohub 的重负载打断 50Hz rt/arm_sdk → merger 回退 policy 上半身。
        print("\n[初始化] 释放相机 (videohub / RealSense)...")
        camera.__enter__()

        if arm_keeper is not None:
            try:
                print("\n[初始化] 等待机器人状态...")
                arm_keeper.move_and_start()
                _flush_stdin()
                input("双臂已进入自然下垂，按 Enter 继续...\n")
            except KeyboardInterrupt:
                _graceful_shutdown(interrupted=True)
                raise SystemExit(0)

        # ── 图像保存目录 ─────────────────────────────────────────────
        img_dir = os.path.join(os.path.dirname(__file__), "img")
        os.makedirs(img_dir, exist_ok=True)
        vision_log = VisionSessionLog(
            base_dir=args.vision_log_dir or os.path.join(img_dir, "runs"),
            enabled=not args.no_vision_log,
        )

        def _resolve_grip_log_dir(log: VisionSessionLog) -> str | None:
            if args.no_grip_check:
                return None
            if log.root:
                return log.root
            stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
            d = os.path.join(img_dir, "grip_logs", stamp)
            os.makedirs(d, exist_ok=True)
            print(f"[grip_log] 记录目录 → {d}")
            return d

        grip_log_dir = _resolve_grip_log_dir(vision_log)

        def _sam3_predict(stage: str, color, depth, extra=None):
            """SAM3 预测并写入 vision_log。返回 (result, meta)。"""
            result, meta = predict_from_arrays(
                color, depth, args.prompt, args.host, args.port,
                args.point_cloud_stage, return_meta=True,
            )
            vision_log.record(stage, color, depth, sam3=meta, extra=extra)
            return result, meta

        def _mask_touches_active(meta: dict) -> bool:
            touches = meta.get("mask_touches")
            if not touches:
                return False
            return any(touches.get(k) for k in ("top", "bottom", "left", "right"))

        def _move_for_mask_touches(touches: dict) -> tuple[float, float]:
            move_x, move_y = 0.0, 0.0
            if touches.get("top"):
                move_x += ADJUST_STEP
            if touches.get("bottom"):
                move_x -= ADJUST_STEP
            if touches.get("left"):
                move_y += ADJUST_STEP
            if touches.get("right"):
                move_y -= ADJUST_STEP
            return move_x, move_y

        # ═══════════════════════════════════════════════════════════════
        # 主循环：搜索 + 抓取
        # ═══════════════════════════════════════════════════════════════
        max_attempts = args.max_attempts

        def _restart_grasp_round(succeeded: bool) -> None:
            """一轮抓取结束（成功或 attempt 用尽）后，按 Enter 从头开始新一轮。"""
            nonlocal vision_log, grip_log_dir
            if arm_keeper is not None:
                arm_keeper.ensure_keepalive()
            if succeeded:
                print("\n抓取完成!")
            else:
                print(f"\n{max_attempts} 次尝试均未成功抓取。")
            print("双臂保持自然下垂。")
            _flush_stdin()
            input("按 Enter 开始新一轮抓取...\n")

            # 仅切换腿部行走模式 (RL_FULL)，便于后续对齐时底盘移动。
            # 双臂仍由 arm_keeper 经 rt/arm_sdk 保持自然下垂，不会回到 policy 初始位姿。
            if arm_keeper is not None:
                arm_keeper.ensure_keepalive()
            print("[新一轮] 腿部切换为行走模式，双臂保持自然下垂...")
            try:
                mover.initialize()
                time.sleep(0.3)
            except Exception as exc:
                print(f"  警告: mover.initialize 失败: {exc}")

            if abs(waist.yaw) > 0.01:
                waist.rotate(-waist.yaw)
            waist.stop_hold()
            time.sleep(0.3)

            vision_log = VisionSessionLog(
                base_dir=args.vision_log_dir or os.path.join(img_dir, "runs"),
                enabled=not args.no_vision_log,
            )
            grip_log_dir = _resolve_grip_log_dir(vision_log)

        try:

            # ── 预走：先向货架方向前进 30cm ──
            # print("\n  预走: 向货架前进 30cm...")
            # _walk(0.30, 0.0)
            # time.sleep(0.5)

            # ── 腰部回正 ──
            if abs(waist.yaw) > 0.01:
                waist.rotate(-waist.yaw)
                waist.stop_hold()
                time.sleep(0.3)

            # ═══════════════════════════════════════════════════════════════
            # 搜索+抓取：仅正前方
            # ═══════════════════════════════════════════════════════════════

            while True:  # 每轮抓取结束后可按 Enter 重新开始

                print(f"\n{'=' * 55}")
                print(f"  开始抓取流程...")
                print(f"{'=' * 55}")

                grasp_succeeded = False
                for attempt in range(1, max_attempts + 1):
                    print(f"\n{'─' * 55}")
                    print(f"  第 {attempt}/{max_attempts} 次尝试")
                    print(f"{'─' * 55}")
                    ik_fwd_idx = 0

                    # ── 对齐-1: SAM3 mask 边界（需移动则本 attempt 结束）──
                    print("\n[对齐-1] SAM3 分割 → mask 边界检测")
                    color, depth = camera.capture()
                    result, meta = _sam3_predict("mask_align", color, depth,
                                                 extra={"attempt": attempt})
                    touches = meta.get("mask_touches")
                    if touches is not None and _mask_touches_active(meta):
                        move_x, move_y = _move_for_mask_touches(touches)
                        touch_names = [k for k, v in touches.items() if v]
                        print(f"  [对齐-1] mask 触及 {touch_names} → "
                              f"前/后 {move_x*100:+.0f}cm  左/右 {move_y*100:+.0f}cm")
                        _walk(move_x, move_y)
                        print("  [对齐-1] 已移动，进入下一次 attempt")
                        continue
                    print("  [对齐-1] mask 未触及四边（或无 mask 数据），继续")

                    # ── 确保 SAM3 完整预测成功（OBB + 抓取点）────────────
                    grasp_idx = 0
                    while result is None:
                        grasp_idx += 1
                        if grasp_idx > 1:
                            color, depth = camera.capture()
                        stage = "sam3_grasp" if grasp_idx == 1 else f"sam3_grasp_{grasp_idx}"
                        result, meta = _sam3_predict(stage, color, depth,
                                                     extra={"attempt": attempt})
                        if result is None:
                            print("  [SAM3] OBB/抓取点预测失败，重拍...")
                            if grasp_idx >= MAX_SAM3_RETRY:
                                print("  [SAM3] 多次失败，跳过本次 attempt")
                                break
                    if result is None:
                        continue

                    box_center_cam = np.array(result["center"], dtype=float)
                    box_center_torso = cam_to_torso(box_center_cam.tolist())

                    # ── 对齐-2: 箱子中心 Y（需移动则本 attempt 结束）────────
                    print("\n[对齐-2] 箱子中心 Y 检测 (torso 系)")
                    lateral = float(box_center_torso[1])
                    if abs(lateral) > Y_CENTER_LIMIT:
                        side = "左" if lateral > 0 else "右"
                        move_y = ADJUST_STEP if lateral > 0 else -ADJUST_STEP
                        print(f"  [对齐-2] 中心偏{side} {abs(lateral)*100:.0f}cm "
                              f"(>|{Y_CENTER_LIMIT*100:.0f}cm|) → 横移 {ADJUST_STEP*100:.0f}cm")
                        y_stage = "y_align_left" if lateral > 0 else "y_align_right"
                        y_move = "strafe_left_5cm" if move_y > 0 else "strafe_right_5cm"
                        _log_y_align_adjust(
                            y_stage, color, depth, meta, attempt,
                            box_center_torso, lateral,
                            reason="box_center_y_exceeds_limit",
                            next_move=y_move,
                        )
                        _walk(0.0, move_y)
                        print("  [对齐-2] 已移动，进入下一次 attempt")
                        continue
                    print(f"  [对齐-2] 通过 (y={lateral*100:+.1f}cm)")

                    # ── 坐标转换 + 手长补偿 ─────────────────────────────
                    print("\n[坐标转换] 相机系 → torso 系")
                    left_torso, right_torso = grasp_to_point_a_torso(
                        result["grasp_left"], result["grasp_right"],
                    )

                    print(f"  torso系 left  = [{left_torso[0]:.4f}, {left_torso[1]:.4f}, {left_torso[2]:.4f}]")
                    print(f"  torso系 right = [{right_torso[0]:.4f}, {right_torso[1]:.4f}, {right_torso[2]:.4f}]")
                    print(f"  箱子中心 (torso): x={box_center_torso[0]:.3f}  "
                          f"y={box_center_torso[1]:.3f}  z={box_center_torso[2]:.3f}")

                    # ── 对齐-3: 点A X 工作空间（需移动则本 attempt 结束）────
                    print("\n[对齐-3] 点A X 检测 (torso 系)")
                    left_x = float(left_torso[0])
                    right_x = float(right_torso[0])
                    if left_x < X_POINT_A_CLOSE or right_x < X_POINT_A_CLOSE:
                        print(f"  [对齐-3] 点A过近 (left={left_x:.3f}, right={right_x:.3f}, "
                              f"任一<{X_POINT_A_CLOSE:.2f}) → 后退 {ADJUST_STEP*100:.0f}cm")
                        _log_x_align_adjust(
                            "x_align_back", color, depth, meta, attempt,
                            left_torso, right_torso,
                            reason="point_a_x_below_close_limit",
                            next_move="backward_5cm",
                        )
                        _walk(-ADJUST_STEP, 0.0)
                        print("  [对齐-3] 已移动，进入下一次 attempt")
                        continue
                    if left_x > X_POINT_A_FAR or right_x > X_POINT_A_FAR:
                        print(f"  [对齐-3] 点A过远 (left={left_x:.3f}, right={right_x:.3f}, "
                              f"任一>{X_POINT_A_FAR:.2f}) → 前进 {ADJUST_STEP*100:.0f}cm")
                        _log_x_align_adjust(
                            "x_align_forward", color, depth, meta, attempt,
                            left_torso, right_torso,
                            reason="point_a_x_above_far_limit",
                            next_move="forward_5cm",
                        )
                        _walk(ADJUST_STEP, 0.0)
                        print("  [对齐-3] 已移动，进入下一次 attempt")
                        continue
                    print(f"  [对齐-3] 通过 (left_x={left_x:.3f}, right_x={right_x:.3f})")

                    # ── IK: 工作空间 + 误差复检 ─────────────────────────
                    print("\n[IK] 工作空间与误差检测")
                    ik_ok, left_q0, right_q0, ik_action, left_ik_err, right_ik_err = _eval_ik(
                        left_torso, right_torso)
                    if not ik_ok:
                        if ik_action == "forward":
                            ik_fwd_idx += 1
                            ik_stage = ("ik_unreachable" if ik_fwd_idx == 1
                                        else f"ik_unreachable_{ik_fwd_idx}")
                            _log_ik_forward_retry(
                                ik_stage, color, depth, meta, attempt,
                                left_torso, right_torso, left_ik_err, right_ik_err,
                                reason="max_err_ge_unsolvable_limit",
                            )
                            print("  [IK] 算不出解，前进 5cm 后进入下一次 attempt")
                            _walk(ADJUST_STEP, 0.0)
                            continue
                        if ik_action == "hang":
                            _hang_here(
                                f"IK 严重警告：重算后仍超出 {IK_SEVERE_LIMIT*1000:.0f}mm"
                            )

                    # ── 完整 IK + 初始化控制器 ──────────────────────────
                    print("\n[执行准备] 求解逆运动学并初始化控制器")
                    ik_start = time.time()
                    _grasp_end = "hold_handoff" if arm_keeper is not None else args.end_behavior
                    try:
                        controller = DualArmController(
                            left_target=left_torso.tolist(),
                            right_target=right_torso.tolist(),
                            end_behavior=_grasp_end,
                            left_q0=left_q0,
                            right_q0=right_q0,
                            enable_grip_check=not args.no_grip_check,
                            grip_log_dir=grip_log_dir,
                            grip_log_tag=f"attempt_{attempt:02d}",
                        )
                    except ValueError as e:
                        ik_elapsed = time.time() - ik_start
                        print(f"\n[IK 失败] {e} (耗时 {ik_elapsed:.1f}s)")
                        ik_fwd_idx += 1
                        ik_stage = ("ik_unreachable" if ik_fwd_idx == 1
                                    else f"ik_unreachable_{ik_fwd_idx}")
                        _log_ik_forward_retry(
                            ik_stage, color, depth, meta, attempt,
                            left_torso, right_torso, left_ik_err, right_ik_err,
                            reason=f"dual_arm_controller: {e}",
                        )
                        print("  完整 IK 无法收敛，前进 5cm 后进入下一次 attempt")
                        _walk(ADJUST_STEP, 0.0)
                        continue

                    ik_elapsed = time.time() - ik_start
                    print(f"  IK 求解完成 (耗时 {ik_elapsed:.1f}s)")

                    print(f"\n  左手目标 (torso): x={left_torso[0]:+.3f}  y={left_torso[1]:+.3f}  z={left_torso[2]:+.3f} m")
                    print(f"  右手目标 (torso): x={right_torso[0]:+.3f}  y={right_torso[1]:+.3f}  z={right_torso[2]:+.3f} m")

                    print("\n  目标关节角 (ARM_JOINTS 顺序：左臂7 + 右臂7 + 腰3)：")
                    for i, joint in enumerate(ARM_JOINTS):
                        q = controller.target_q[i]
                        deg = math.degrees(q)
                        print(f"    {joint.name:20s}  q={q:+.4f} rad ({deg:+.2f}°)")

                    csv_need_header = [not os.path.exists(GOAL_CSV_PATH)]
                    save_goal_to_csv(
                        left_torso, right_torso, controller.target_q,
                        controller.left_fk, controller.right_fk,
                        csv_need_header,
                    )
                    print(f"\n  已写入目标值 → {GOAL_CSV_PATH}")

                    if not args.no_confirm:
                        print("\n⚠  执行前请确认：")
                        print("   1. 机器人周围无障碍物，手臂可以自由活动")
                        print("   2. 抓取目标点位置正确（见上方坐标）")
                        _flush_stdin(); input("按 Enter 开始执行，Ctrl+C 取消...\n")

                    # ── 交权给 DualArmController ─────────────────────────
                    _handoff_cb = make_handoff_callback(arm_keeper) if arm_keeper is not None else None
                    _release_cb = make_release_callback(arm_keeper) if arm_keeper is not None else None
                    if arm_keeper is not None:
                        print("[执行] 切换到 RL_LOWER，DualArmController 无缝接管 arm_sdk...")
                    else:
                        print("[执行] 停掉腰部 keepalive，切换到 RL_LOWER，交权给手臂...")
                    waist.stop_hold()
                    time.sleep(0.15)
                    mover.shutdown()
                    time.sleep(0.5)

                    print("[执行] 开始双臂运动")
                    active_controller = controller

                    def _regrasp_capture_point_a():
                        """箱子脱落后：重拍并重新计算点A。"""
                        c, d = camera.capture()
                        res, m = predict_from_arrays(
                            c, d, args.prompt, args.host, args.port,
                            args.point_cloud_stage, return_meta=True,
                        )
                        vision_log.record(
                            "regrasp_sam3", c, d, sam3=m,
                            extra={"reason": "box_dropped_regrasp"},
                        )
                        if res is None:
                            raise RuntimeError("SAM3 重拍失败，请检查箱子是否在视野内")
                        lt, rt = grasp_to_point_a_torso(res["grasp_left"], res["grasp_right"])
                        print(f"  [重抓] 新点A 左 torso={lt.round(4).tolist()}  右 torso={rt.round(4).tolist()}")
                        return lt.tolist(), rt.tolist()

                    controller.run(
                        release_external_hold=_release_cb,
                        handoff_external_hold=_handoff_cb,
                        regrasp_callback=_regrasp_capture_point_a,
                    )
                    if arm_keeper is not None:
                        arm_keeper.ensure_keepalive()

                    grasp_succeeded = True
                    break

                if arm_keeper is not None:
                    _restart_grasp_round(grasp_succeeded)
                    continue

                # ── 无自然下垂：交还给 GR00T-WBC(RL_FULL)，可继续检测下一箱 ──
                waist.stop_hold()
                waist.sync_yaw()
                waist._arm_q = None
                mover.initialize()
                time.sleep(0.5)

        except KeyboardInterrupt:
            _graceful_shutdown(controller=active_controller, interrupted=True)
            raise SystemExit(0)
        except Exception as exc:
            print(f"\n[错误] {type(exc).__name__}: {exc}")
            if arm_keeper is not None and arm_keeper.is_running():
                _graceful_shutdown(controller=active_controller)
            raise

        # ── 抓取结束 ──────────────────────────────────────────
        if not shutdown_done:
            if arm_keeper is not None:
                print("\n抓取结束。双臂保持自然下垂。")
                _flush_stdin()
                input("按 Enter 解除自然下垂，交还给 policy 并退出...\n")
                _graceful_shutdown()
            else:
                print("\n抓取结束。")
                if abs(waist.yaw) > 0.01:
                    waist.rotate(-waist.yaw)
                mover.release()
    finally:
        camera.__exit__(None, None, None)


if __name__ == "__main__":
    main()
