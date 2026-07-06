#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 RoboJuDo 的 rt/lowcmd_rl 与 box_demo / dual_arm 的 rt/arm_sdk 合成为最终 rt/lowcmd。

- 默认：motor 0–11 及未启用 arm_sdk 时 12–34 均来自 RL（下半身 + 其余）。
- 当 rt/arm_sdk 中 motor_cmd[29].q > 阈值 且报文未过期：motor 12–29（腰 + 双臂 + arm_sdk 权重槽）
  整段来自 arm_sdk（kp/kd/q 等与 box_demo 一致）。

启动顺序（同一 DDS 域、同一网卡）：
  1) python merge_lowcmd_arm_sdk.py --iface enP8p1s0
  2) python run_pipeline.py -c g1_mjlab_loco_real_merge
  3) python box_demo_main.py --iface enP8p1s0

须先于 unitree_sdk2py 加载与 wheel 匹配的 libddsc（与 run_pipeline / box_demo 一致）。
"""

from __future__ import annotations

import argparse
import ctypes
import os
import sys
import threading
import time
from pathlib import Path

_ddsc = Path(os.environ.get("CYCLONEDDS_HOME", Path.home() / "cyclonedds-0.10-install")) / "lib" / "libddsc.so.0"
if _ddsc.is_file():
    ctypes.CDLL(str(_ddsc), mode=ctypes.RTLD_GLOBAL)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread

NUM_MOTORS = 35
RESERVE_LEN = 4
# 与 dual_arm_target_reach.G1Joint 一致：腰(12–14) + 双臂(15–28) + ArmSdkEnable(29)
OVERLAY_LO = 12
OVERLAY_HI = 30  # exclusive → 12..29


def _copy_motor(dst: LowCmd_, src: LowCmd_, idx: int) -> None:
    dm = dst.motor_cmd[idx]
    sm = src.motor_cmd[idx]
    dm.mode = sm.mode
    dm.q = sm.q
    dm.dq = sm.dq
    dm.tau = sm.tau
    dm.kp = sm.kp
    dm.kd = sm.kd
    dm.reserve = sm.reserve


def _copy_full(dst: LowCmd_, src: LowCmd_) -> None:
    dst.mode_pr = src.mode_pr
    dst.mode_machine = src.mode_machine
    for i in range(NUM_MOTORS):
        _copy_motor(dst, src, i)
    for i in range(RESERVE_LEN):
        dst.reserve[i] = src.reserve[i]


def _snapshot(msg: LowCmd_) -> LowCmd_:
    out = unitree_hg_msg_dds__LowCmd_()
    _copy_full(out, msg)
    out.crc = msg.crc
    return out


class Merger:
    def __init__(
        self,
        iface: str,
        hz: float,
        weight_threshold: float,
        arm_stale_s: float,
        rl_stale_s: float,
    ):
        self._iface = iface
        self._period = 1.0 / hz
        self._weight_threshold = weight_threshold
        self._arm_stale_s = arm_stale_s
        self._rl_stale_s = rl_stale_s
        self._lock = threading.Lock()
        self._rl: LowCmd_ | None = None
        self._rl_t = 0.0
        self._arm: LowCmd_ | None = None
        self._arm_t = 0.0
        self._crc = CRC()
        # 须在 ChannelFactoryInitialize 之后再创建 Publisher/Subscriber，否则 participant 为 None
        self._pub = None
        self._sub_rl = None
        self._sub_arm = None
        self._thread: RecurrentThread | None = None

    def _on_rl(self, msg: LowCmd_):
        with self._lock:
            self._rl = _snapshot(msg)
            self._rl_t = time.monotonic()

    def _on_arm(self, msg: LowCmd_):
        with self._lock:
            self._arm = _snapshot(msg)
            self._arm_t = time.monotonic()

    def _tick(self):
        now = time.monotonic()
        with self._lock:
            rl = self._rl
            rl_age = now - self._rl_t
            arm = self._arm
            arm_age = now - self._arm_t if arm is not None else 1e9

        if rl is None or rl_age > self._rl_stale_s:
            return

        out = unitree_hg_msg_dds__LowCmd_()
        _copy_full(out, rl)

        use_arm = (
            arm is not None
            and arm_age <= self._arm_stale_s
            and float(arm.motor_cmd[29].q) > self._weight_threshold
        )
        if use_arm:
            for i in range(OVERLAY_LO, OVERLAY_HI):
                _copy_motor(out, arm, i)

        out.crc = self._crc.Crc(out)
        if self._pub is not None:
            self._pub.Write(out)

    def run(self):
        ChannelFactoryInitialize(0, self._iface)
        self._pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self._pub.Init()
        self._sub_rl = ChannelSubscriber("rt/lowcmd_rl", LowCmd_)
        self._sub_rl.Init(self._on_rl, 10)
        self._sub_arm = ChannelSubscriber("rt/arm_sdk", LowCmd_)
        self._sub_arm.Init(self._on_arm, 10)
        self._thread = RecurrentThread(
            interval=self._period,
            target=self._tick,
            name="merge_lowcmd",
        )
        self._thread.Start()
        print(
            f"merge_lowcmd_arm_sdk: iface={self._iface} 合成→rt/lowcmd  "
            f"订阅 rt/lowcmd_rl + rt/arm_sdk  周期={self._period*1000:.1f} ms"
        )
        print("按 Ctrl+C 退出。")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\n退出。")
        finally:
            if self._thread is not None:
                self._thread.Wait(timeout=1.0)


def main():
    p = argparse.ArgumentParser(description="Merge rt/lowcmd_rl + rt/arm_sdk → rt/lowcmd")
    p.add_argument("--iface", type=str, default=os.environ.get("UNITREE_DDS_INTERFACE", "enP8p1s0"))
    p.add_argument("--hz", type=float, default=500.0, help="发布 rt/lowcmd 频率")
    p.add_argument("--weight-threshold", type=float, default=1e-3, help="arm_sdk motor[29].q 超过此值则覆盖腰+臂")
    p.add_argument("--arm-stale-s", type=float, default=0.25, help="超过此时间未收到 arm_sdk 则只用 RL 上半身")
    p.add_argument("--rl-stale-s", type=float, default=0.5, help="超过此时间未收到 lowcmd_rl 则停止写 rt/lowcmd")
    args = p.parse_args()
    Merger(
        iface=args.iface,
        hz=args.hz,
        weight_threshold=args.weight_threshold,
        arm_stale_s=args.arm_stale_s,
        rl_stale_s=args.rl_stale_s,
    ).run()


if __name__ == "__main__":
    main()
