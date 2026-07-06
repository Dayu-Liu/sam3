#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BrainCo Revo2 灵巧手独立控制脚本（仅手，不依赖 SONIC）

通过 DDS topic 向 brainco_hand_server 发送张开/闭合命令：
  rt/brainco/left/cmd
  rt/brainco/right/cmd

6 轴顺序: [thumb, thumb_aux, index, middle, ring, pinky]
  0.0 = 张开, 1.0 = 闭合

前置条件:
  1. G1 上已启动 brainco_hand_server（且未同时运行 SONIC deploy）
  2. 身体其余关节处于松软/阻尼状态（本脚本不控制全身）

用法:
  python brainco_hand_control.py
  python brainco_hand_control.py --iface enP8p1s0

运行后在终端输入:
  1 = 左手闭合    2 = 左手张开
  3 = 右手闭合    4 = 右手张开
  q = 退出
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from typing import List, Optional, Sequence

# unitree_sdk2py 可能未 pip 安装，尝试常见路径
_SDK2_CANDIDATES = [
    "/home/unitree/unitree_sdk2_python",
    os.path.expanduser("~/unitree_sdk2_python"),
]
for _p in _SDK2_CANDIDATES:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from unitree_sdk2py.core.channel import (  # noqa: E402
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import (  # noqa: E402
    MotorCmd_,
    MotorCmds_,
    MotorStates_,
)

OPEN_Q = [0.0] * 6
CLOSE_Q = [1.0] * 6
DEFAULT_IFACE = "enP8p1s0"
DEFAULT_HZ = 10.0

MENU = """
控制菜单:
  1 = 左手闭合    2 = 左手张开
  3 = 右手闭合    4 = 右手张开
  q = 退出
> """


def _make_cmd(positions: Sequence[float], speed: float = 1.0) -> MotorCmds_:
    cmds: List[MotorCmd_] = []
    for q in positions:
        cmds.append(
            MotorCmd_(
                mode=0,
                q=float(q),
                dq=float(speed),
                tau=0.0,
                kp=0.0,
                kd=0.0,
                reserve=[0, 0, 0],
            )
        )
    return MotorCmds_(cmds=cmds)


def _lerp(a: Sequence[float], b: Sequence[float], t: float) -> List[float]:
    t = max(0.0, min(1.0, t))
    return [a[i] + (b[i] - a[i]) * t for i in range(6)]


class BraincoHandClient:
    def __init__(self, sides: Sequence[str], hz: float = DEFAULT_HZ):
        self._sides = list(sides)
        self._hz = hz
        self._period = 1.0 / hz
        self._publishers = {
            side: ChannelPublisher(f"rt/brainco/{side}/cmd", MotorCmds_)
            for side in self._sides
        }
        self._states: dict[str, Optional[MotorStates_]] = {s: None for s in self._sides}
        self._connected: dict[str, bool] = {s: False for s in self._sides}
        self._subs = []
        for side in self._sides:
            pub = self._publishers[side]
            pub.Init()

            def _on_state(msg: MotorStates_, s=side):
                self._states[s] = msg

            sub = ChannelSubscriber(f"rt/brainco/{side}/state", MotorStates_)
            sub.Init(_on_state, 10)
            self._subs.append(sub)

    def wait_for_connection(self, timeout: float = 5.0) -> None:
        deadline = time.time() + timeout
        for side in self._sides:
            while self._states[side] is None:
                if time.time() > deadline:
                    self._connected[side] = False
                    print(f"  [警告] rt/brainco/{side}/state 未连接，该手不可用。")
                    break
                time.sleep(0.05)
            else:
                self._connected[side] = True
                print(f"  [已连接] {side} 手")

        if not any(self._connected.values()):
            raise TimeoutError("左右手均未连接，请确认 brainco_hand_server 已启动。")

    def is_connected(self, side: str) -> bool:
        return self._connected.get(side, False)

    def read_q(self, side: str) -> List[float]:
        st = self._states.get(side)
        if st is None or len(st.states) < 6:
            return list(OPEN_Q)
        return [float(st.states[i].q) for i in range(6)]

    def publish_side(self, side: str, positions: Sequence[float], speed: float = 1.0) -> None:
        if side not in self._publishers:
            return
        self._publishers[side].Write(_make_cmd(positions, speed), 0.5)

    def move_side_to(
        self,
        side: str,
        target: Sequence[float],
        ramp: float = 2.0,
        hold: float = 0.5,
        speed: float = 1.0,
    ) -> None:
        if not self.is_connected(side):
            print(f"  [{side}] 未连接，跳过。")
            return

        start = self.read_q(side)
        if ramp <= 0:
            self.publish_side(side, target, speed)
            if hold > 0:
                end = time.time() + hold
                while time.time() < end:
                    self.publish_side(side, target, speed)
                    time.sleep(self._period)
            return

        steps = max(1, int(ramp * self._hz))
        for i in range(1, steps + 1):
            pos = _lerp(start, target, i / steps)
            self.publish_side(side, pos, speed)
            time.sleep(self._period)

        if hold > 0:
            end = time.time() + hold
            while time.time() < end:
                self.publish_side(side, target, speed)
                time.sleep(self._period)

    def open_side(self, side: str, ramp: float, hold: float, speed: float) -> None:
        print(f"  [{side}] 张开...")
        self.move_side_to(side, OPEN_Q, ramp=ramp, hold=hold, speed=speed)
        print(f"  [{side}] 完成。")

    def close_side(self, side: str, ramp: float, hold: float, speed: float) -> None:
        print(f"  [{side}] 闭合...")
        self.move_side_to(side, CLOSE_Q, ramp=ramp, hold=hold, speed=speed)
        print(f"  [{side}] 完成。")

    def open_all_connected(self, ramp: float, speed: float) -> None:
        for side in self._sides:
            if self.is_connected(side):
                self.move_side_to(side, OPEN_Q, ramp=ramp, hold=0.2, speed=speed)


def run_menu(client: BraincoHandClient, args: argparse.Namespace) -> None:
    print(MENU.strip())
    while True:
        try:
            key = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if key in ("q", "quit", "exit"):
            break
        elif key == "1":
            client.close_side("left", args.ramp, args.hold, args.speed)
        elif key == "2":
            client.open_side("left", args.ramp, args.hold, args.speed)
        elif key == "3":
            client.close_side("right", args.ramp, args.hold, args.speed)
        elif key == "4":
            client.open_side("right", args.ramp, args.hold, args.speed)
        else:
            print("无效输入，请输入 1 / 2 / 3 / 4 / q")
            print(MENU.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BrainCo Revo2 灵巧手交互控制（无需 SONIC）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--iface",
        default=DEFAULT_IFACE,
        help=f"DDS 网络接口 (默认: {DEFAULT_IFACE})",
    )
    parser.add_argument("--hz", type=float, default=DEFAULT_HZ, help="发布频率 Hz")
    parser.add_argument("--ramp", type=float, default=2.0, help="动作渐变时间 (秒)")
    parser.add_argument("--hold", type=float, default=0.5, help="到达目标后保持时间 (秒)")
    parser.add_argument("--speed", type=float, default=1.0, help="手指速度 dq [0,1]")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.iface:
        ChannelFactoryInitialize(0, args.iface)
    else:
        ChannelFactoryInitialize(0)

    client = BraincoHandClient(["left", "right"], hz=args.hz)

    def _safe_open(*_):
        print("\n收到中断，安全张开已连接的手...")
        try:
            client.open_all_connected(ramp=1.0, speed=args.speed)
        except Exception as exc:
            print(f"安全张开失败: {exc}")
        sys.exit(0)

    signal.signal(signal.SIGINT, _safe_open)
    signal.signal(signal.SIGTERM, _safe_open)

    print("等待 brainco_hand_server 连接...")
    try:
        client.wait_for_connection()
    except TimeoutError as exc:
        print(f"错误: {exc}")
        print(
            "启动示例:\n"
            "  cd /home/unitree/brainco_hand_service/bin\n"
            "  sudo ./brainco_hand_server --network_interface enP8p1s0"
        )
        return 1

    run_menu(client, args)
    print("退出。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
