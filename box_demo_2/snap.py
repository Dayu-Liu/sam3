#!/usr/bin/env python3
"""拍一张照片保存到 img/ 目录下。"""

import os
import sys
import time
import numpy as np
import pyrealsense2 as rs
import cv2

HEAD_CAMERA_SERIAL = "406122070550"
SAVE_DIR = os.path.join(os.path.dirname(__file__), "img")


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(HEAD_CAMERA_SERIAL)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    pipeline.start(config)
    print("相机已启动，等待曝光稳定...")

    for _ in range(30):
        pipeline.wait_for_frames()

    align = rs.align(rs.stream.color)
    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)

    color = np.asanyarray(aligned.get_color_frame().get_data())
    depth = np.asanyarray(aligned.get_depth_frame().get_data())

    pipeline.stop()

    ts = time.strftime("%Y%m%d_%H%M%S")
    color_path = os.path.join(SAVE_DIR, f"{ts}_color.png")
    depth_path = os.path.join(SAVE_DIR, f"{ts}_depth.png")
    depth_vis_path = os.path.join(SAVE_DIR, f"{ts}_depth_vis.png")

    cv2.imwrite(color_path, color)
    cv2.imwrite(depth_path, depth)
    depth_vis = cv2.applyColorMap(
        cv2.convertScaleAbs(depth, alpha=0.05), cv2.COLORMAP_JET
    )
    cv2.imwrite(depth_vis_path, depth_vis)

    print(f"已保存:")
    print(f"  彩色图: {color_path}")
    print(f"  深度图: {depth_path}")
    print(f"  深度可视化: {depth_vis_path}")


if __name__ == "__main__":
    main()
