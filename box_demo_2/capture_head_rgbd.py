"""
采集宇树G1头部相机 RGB + 对齐深度图并保存到本地。

videohub_pc4 已被永久禁用，可直接访问相机，无需任何服务停止操作。

亮度说明:
    RealSense 默认自动曝光，每次冷启动后需多帧才能收敛，5 帧远远不够。
    若亮度忽明忽暗，可：
    1) 增加 warmup_frames（如 50）让自动曝光稳定
    2) 使用 manual_exposure 固定曝光，获得稳定亮度

用法:
    python capture_head_rgbd.py [保存目录] [--manual-exposure]
    默认当前目录；加 --manual-exposure 可固定曝光，亮度更稳定
"""

import os
import sys
import time

import cv2
import numpy as np
import pyrealsense2 as rs

HEAD_CAMERA_SERIAL = "406122070550"

# 手动曝光参数（仅在 manual_exposure=True 时生效）
# 典型范围: exposure 78~300, gain 64~128，需根据实际环境微调
MANUAL_EXPOSURE = 150   # 曝光时间
MANUAL_GAIN = 100       # 增益


def capture_rgbd(color_w=640, color_h=480, fps=30, warmup_frames=50,
                 manual_exposure=False, max_retries=4):
    """
    采集一帧 RGB + 对齐深度图。
    若首次 wait_for_frames 超时（相机冷启动偶发），自动重启管道重试。

    Returns:
        color_image : np.ndarray (H, W, 3) uint8 BGR
        depth_image : np.ndarray (H, W)   uint16 毫米(mm)
        intrinsics  : dict {fx, fy, cx, cy, width, height, coeffs}
    """
    print("初始化 RealSense D435I 相机...")
    align = rs.align(rs.stream.color)

    for attempt in range(1, max_retries + 1):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(HEAD_CAMERA_SERIAL)
        config.enable_stream(rs.stream.color, color_w, color_h, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, color_w, color_h, rs.format.z16, fps)
        try:
            profile = pipeline.start(config)
            # Depth 传感器（IR 投影仪 + 深度 ASIC）冷启动需要 1~3s
            # 先静等再取帧，否则前几帧会超时（50 帧预热时尤其明显）
            time.sleep(2.0)

            if manual_exposure:
                color_sensor = profile.get_device().first_color_sensor()
                color_sensor.set_option(rs.option.enable_auto_exposure, 0)
                color_sensor.set_option(rs.option.exposure, MANUAL_EXPOSURE)
                color_sensor.set_option(rs.option.gain, MANUAL_GAIN)
                print(f"  已启用手动曝光: exposure={MANUAL_EXPOSURE}, gain={MANUAL_GAIN}")
                # 手动曝光生效需多帧，warmup 仍重要
                warmup_actual = max(warmup_frames, 30)
            else:
                warmup_actual = warmup_frames

            print(f"预热（丢弃 {warmup_actual} 帧{'，自动曝光收敛' if not manual_exposure else ''}）...")
            for _ in range(warmup_actual):
                pipeline.wait_for_frames(timeout_ms=8000)

            frames = pipeline.wait_for_frames(timeout_ms=8000)
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()

            if not color_frame or not depth_frame:
                raise RuntimeError("未能获取有效帧")

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            intr = color_frame.profile.as_video_stream_profile().intrinsics
            intrinsics = {
                "fx": intr.fx, "fy": intr.fy,
                "cx": intr.ppx, "cy": intr.ppy,
                "width": intr.width, "height": intr.height,
                "coeffs": list(intr.coeffs),
            }
            print(f"  ✓ RGB {color_image.shape}  Depth {depth_image.shape}")
            return color_image, depth_image, intrinsics

        except RuntimeError as e:
            try:
                pipeline.stop()
            except Exception:
                pass
            if attempt < max_retries:
                print(f"  第{attempt}次尝试失败({e})，等待 2s 后重试...")
                time.sleep(2.0)
            else:
                raise RuntimeError(f"采集失败，已重试 {max_retries} 次: {e}") from e


def main(save_dir=".", color_w=640, color_h=480, manual_exposure=False):
    os.makedirs(save_dir, exist_ok=True)

    color, depth, intr = capture_rgbd(
        color_w=color_w, color_h=color_h,
        warmup_frames=50,
        manual_exposure=manual_exposure,
    )

    # 保存 RGB
    color_path = os.path.join(save_dir, "head_color.png")
    cv2.imwrite(color_path, color)

    # 保存原始深度图（uint16，单位mm，与RGB像素对齐）
    depth_raw_path = os.path.join(save_dir, "head_depth_raw.png")
    cv2.imwrite(depth_raw_path, depth)

    # 保存伪彩色深度图（方便人眼查看）
    depth_vis = cv2.applyColorMap(
        cv2.convertScaleAbs(depth, alpha=0.05), cv2.COLORMAP_JET
    )
    depth_vis_path = os.path.join(save_dir, "head_depth_vis.png")
    cv2.imwrite(depth_vis_path, depth_vis)

    print("\n✓ 采集完成！")
    print(f"  RGB:          {color_path}  {color.shape} {color.dtype}")
    print(f"  深度原始(mm): {depth_raw_path}  {depth.shape} {depth.dtype}")
    print(f"  深度可视化:   {depth_vis_path}")
    print(f"  内参: fx={intr['fx']:.2f}, fy={intr['fy']:.2f}, "
          f"cx={intr['cx']:.2f}, cy={intr['cy']:.2f}")
    print(f"  畸变: {[f'{c:.4f}' for c in intr['coeffs']]}")
    print("\n说明:")
    print("  depth_raw.png — uint16，单位毫米(mm)，像素与RGB完全对齐")
    print("  depth_vis.png — 仅用于可视化，不含真实深度值")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="采集 G1 头部相机 RGB+深度")
    parser.add_argument("save_dir", nargs="?", default=".", help="保存目录")
    parser.add_argument("--manual-exposure", action="store_true",
                        help="使用固定曝光，避免亮度忽明忽暗")
    args = parser.parse_args()
    main(save_dir=args.save_dir, manual_exposure=args.manual_exposure)
