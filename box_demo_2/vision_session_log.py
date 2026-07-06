#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按次记录拍照 + VLM/SAM3 分析结果，便于事后对照调试。"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

import cv2
import numpy as np


def depth_to_vis(depth: np.ndarray) -> np.ndarray:
    """将 uint16 深度图（毫米）转为 8 位灰度，便于肉眼查看。"""
    if depth.ndim != 2:
        raise ValueError(f"depth 应为 2D 数组，实际 shape={depth.shape}")
    d = depth.astype(np.float32)
    max_val = float(d.max())
    if max_val <= 0:
        return np.zeros(depth.shape, dtype=np.uint8)
    return np.clip(d / max_val * 255.0, 0, 255).astype(np.uint8)


class VisionSessionLog:
    """一次 box_demo 运行对应一个时间戳目录，每次分析写一组文件。

    目录结构::

        img/runs/run_20260703_210530/
          manifest.json
          001_vlm_initial/
            color.png
            depth.png
            depth_vis.png
            analysis.json
          002_sam3_confirm/
            color.png
            depth.png
            depth_vis.png
            sam3_overlay_before_erode.png
            sam3_overlay_after_erode.png
            sam3_pointcloud_initial.html
            sam3_pointcloud_sor.html
            sam3_pointcloud_sor_dbscan.html
            analysis.json
    """

    def __init__(self, base_dir: str, *, enabled: bool = True, save_depth: bool = True):
        self.enabled = enabled
        self.save_depth = save_depth
        self._seq = 0
        self._entries: list[dict[str, Any]] = []
        if not enabled:
            self.root = None
            return
        stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.root = os.path.join(base_dir, stamp)
        os.makedirs(self.root, exist_ok=True)
        print(f"[vision_log] 记录目录 → {self.root}")

    def record(
        self,
        stage: str,
        color: np.ndarray,
        depth: np.ndarray | None = None,
        *,
        vlm: dict | None = None,
        sam3: dict | None = None,
        sam3_overlay: np.ndarray | None = None,
        extra: dict | None = None,
    ) -> str | None:
        """保存一帧图像及对应的 VLM / SAM3 分析 JSON。返回本子目录路径。"""
        if not self.enabled or self.root is None:
            return None

        self._seq += 1
        folder_name = f"{self._seq:03d}_{stage}"
        entry_dir = os.path.join(self.root, folder_name)
        os.makedirs(entry_dir, exist_ok=True)

        cv2.imwrite(os.path.join(entry_dir, "color.png"), color)
        if self.save_depth and depth is not None:
            cv2.imwrite(os.path.join(entry_dir, "depth.png"), depth)
            cv2.imwrite(os.path.join(entry_dir, "depth_vis.png"), depth_to_vis(depth))

        overlay_before = sam3_overlay
        if overlay_before is None and sam3 is not None:
            overlay_before = sam3.get("overlay_before_erode")
            if overlay_before is None:
                overlay_before = sam3.get("overlay")
        if overlay_before is not None:
            cv2.imwrite(
                os.path.join(entry_dir, "sam3_overlay_before_erode.png"),
                overlay_before,
            )

        overlay_after = None
        if sam3 is not None:
            overlay_after = sam3.get("overlay_after_erode")
        if overlay_after is not None:
            cv2.imwrite(
                os.path.join(entry_dir, "sam3_overlay_after_erode.png"),
                overlay_after,
            )

        _PC_HTML_FIELDS = (
            ("pointcloud_initial_html", "sam3_pointcloud_initial.html"),
            ("pointcloud_sor_html", "sam3_pointcloud_sor.html"),
            ("pointcloud_sor_dbscan_html", "sam3_pointcloud_sor_dbscan.html"),
        )
        if sam3 is not None:
            for meta_key, filename in _PC_HTML_FIELDS:
                html = sam3.get(meta_key)
                if html:
                    with open(os.path.join(entry_dir, filename), "w", encoding="utf-8") as f:
                        f.write(html)

        analysis: dict[str, Any] = {
            "seq": self._seq,
            "stage": stage,
            "time": datetime.now().isoformat(timespec="seconds"),
        }
        if vlm is not None:
            analysis["vlm"] = vlm
        if sam3 is not None:
            analysis["sam3"] = {
                k: v for k, v in sam3.items()
                if k not in (
                    "overlay", "overlay_before_erode", "overlay_after_erode",
                    "pointcloud", "pointcloud_sor_dbscan_html",
                    "pointcloud_initial_html", "pointcloud_sor_html",
                )
            }
            if sam3.get("has_overlay_before_erode") or sam3.get("has_overlay"):
                analysis["sam3"]["has_overlay_before_erode"] = True
            if sam3.get("has_overlay_after_erode"):
                analysis["sam3"]["has_overlay_after_erode"] = True
            if sam3.get("has_pointcloud_initial"):
                analysis["sam3"]["has_pointcloud_initial"] = True
            if sam3.get("has_pointcloud_sor"):
                analysis["sam3"]["has_pointcloud_sor"] = True
            if sam3.get("has_pointcloud_sor_dbscan") or sam3.get("pointcloud") is not None:
                analysis["sam3"]["has_pointcloud_sor_dbscan"] = True
        if extra:
            analysis["extra"] = extra

        analysis_path = os.path.join(entry_dir, "analysis.json")
        with open(analysis_path, "w", encoding="utf-8") as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)

        summary = {
            "seq": self._seq,
            "stage": stage,
            "dir": folder_name,
            "time": analysis["time"],
        }
        if vlm is not None:
            summary["vlm_box_visible"] = vlm.get("box_visible")
            summary["vlm_box_in_frame"] = vlm.get("box_in_frame")
        if sam3 is not None:
            summary["sam3_ok"] = sam3.get("ok")
            if sam3.get("has_overlay_before_erode") or sam3.get("has_overlay"):
                summary["sam3_has_overlay_before_erode"] = True
            if sam3.get("has_overlay_after_erode"):
                summary["sam3_has_overlay_after_erode"] = True
            if sam3.get("has_pointcloud_initial"):
                summary["sam3_has_pointcloud_initial"] = True
            if sam3.get("has_pointcloud_sor"):
                summary["sam3_has_pointcloud_sor"] = True
            if sam3.get("has_pointcloud_sor_dbscan"):
                summary["sam3_has_pointcloud_sor_dbscan"] = True
            if not sam3.get("ok"):
                summary["sam3_error"] = sam3.get("error", sam3.get("detail", ""))[:200]
            if sam3.get("mask_touches"):
                summary["mask_touches"] = sam3["mask_touches"]
        self._entries.append(summary)
        self._write_manifest()
        print(f"[vision_log] #{self._seq:03d} {stage} → {folder_name}/")
        return entry_dir

    def _write_manifest(self) -> None:
        if self.root is None:
            return
        manifest_path = os.path.join(self.root, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(self._entries, f, ensure_ascii=False, indent=2)
