#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""夹持监测指标记录与绘图（仅电机力矩 tau_mean）。"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

GRIP_DROP_RATIO = 0.70


class GripMetricsLog:
    """记录夹持检测窗口力矩指标，并导出 JSON + PNG。"""

    def __init__(self, output_dir: str | None = None):
        self.output_dir = output_dir
        self._records: list[dict[str, Any]] = []
        self._t0: float | None = None

    @property
    def records(self) -> list[dict[str, Any]]:
        return self._records

    def append(
        self,
        *,
        stage_label: str,
        tau_mean: float,
        is_baseline: bool = False,
        gripped: bool | None = None,
        baseline_tau: float | None = None,
        drop_detected: bool = False,
        window_idx: int | None = None,
    ) -> None:
        now = time.monotonic()
        if self._t0 is None:
            self._t0 = now
        self._records.append(
            {
                "t_s": round(now - self._t0, 3),
                "stage_label": stage_label,
                "tau_mean": float(tau_mean),
                "window_idx": window_idx,
                "is_baseline": bool(is_baseline),
                "gripped": gripped,
                "baseline_tau": baseline_tau,
                "drop_threshold_tau": (
                    float(baseline_tau) * GRIP_DROP_RATIO
                    if baseline_tau is not None
                    else None
                ),
                "drop_detected": bool(drop_detected),
            }
        )

    def save(self, output_dir: str | None = None, tag: str | None = None) -> str | None:
        """写入 JSON 与 PNG。无记录时返回 None。"""
        if not self._records:
            return None
        out_dir = output_dir or self.output_dir
        if not out_dir:
            return None
        os.makedirs(out_dir, exist_ok=True)

        suffix = f"_{tag}" if tag else ""
        json_path = os.path.join(out_dir, f"grip_metrics{suffix}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self._records, f, ensure_ascii=False, indent=2)

        png_path = os.path.join(out_dir, f"grip_metrics{suffix}.png")
        self._plot(png_path)
        print(f"[grip_log] 指标 JSON → {json_path}")
        print(f"[grip_log] 指标曲线 → {png_path}")
        return out_dir

    def _plot(self, png_path: str) -> None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        fig.suptitle("Grip Watch Torque (250ms windows)", fontsize=13)

        blue_ts, blue_tau = [], []
        red_ts, red_tau = [], []
        for rec in self._records:
            if rec.get("is_baseline"):
                red_ts.append(rec["t_s"])
                red_tau.append(rec["tau_mean"])
            else:
                blue_ts.append(rec["t_s"])
                blue_tau.append(rec["tau_mean"])

        if blue_ts:
            ax.plot(
                blue_ts, blue_tau, marker="o", markersize=3, linewidth=1.2,
                color="tab:blue", label="tau_mean (Nm)",
            )
        if red_ts:
            ax.plot(
                red_ts, red_tau, marker="o", markersize=5, linewidth=0,
                color="red", label="第5个250ms采样点",
            )

        ax.set_ylabel("tau_mean (Nm)")
        ax.set_xlabel("time (s)")
        ax.grid(True, alpha=0.3)

        for rec in self._records:
            if rec.get("drop_detected"):
                ax.axvline(rec["t_s"], color="crimson", linestyle="--", alpha=0.8)

        # 各阶段第 5 个 250ms 的 70% 作为脱落阈值，每阶段画一条水平线（仅该段时间内）
        stage_changes: list[tuple[float, str]] = []
        prev = None
        for rec in self._records:
            label = rec["stage_label"]
            if label != prev:
                stage_changes.append((rec["t_s"], label))
                prev = label

        t_max = max(r["t_s"] for r in self._records)
        baseline_by_stage = {
            r["stage_label"]: r for r in self._records if r.get("is_baseline")
        }
        thr_labels_drawn: set[str] = set()
        for i, (t_start, stage) in enumerate(stage_changes):
            t_end = stage_changes[i + 1][0] if i + 1 < len(stage_changes) else t_max + 0.001
            rec = baseline_by_stage.get(stage)
            if rec is None:
                continue
            b = rec.get("baseline_tau")
            if b is None:
                continue
            thr = b * GRIP_DROP_RATIO
            label = (
                f"{stage} 阈值70% ({thr:.2f}Nm)"
                if stage not in thr_labels_drawn
                else None
            )
            ax.plot(
                [t_start, t_end], [thr, thr],
                color="crimson", linestyle="--", linewidth=1.2, alpha=0.75,
                label=label,
            )
            if label:
                thr_labels_drawn.add(stage)

        # 阶段之间的竖直分割线
        for t_s, _label in stage_changes[1:]:
            ax.axvline(
                t_s, color="gray", linestyle="-", alpha=0.6, linewidth=1.0,
            )

        ax.legend(loc="upper right", fontsize=7)
        fig.tight_layout()
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
