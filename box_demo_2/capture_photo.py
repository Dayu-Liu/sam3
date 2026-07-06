"""
快速拍照脚本 —— 从 G1 头部 RealSense D435 采集一帧 RGB 并保存。

用法:
    python capture_photo.py                    # 保存为 capture.jpg
    python capture_photo.py -o my_photo.jpg    # 指定输出路径
    python capture_photo.py --no-preview       # 不显示预览窗口
"""
import argparse
import os
import subprocess
import sys
import time

import cv2
import numpy as np
import pyrealsense2 as rs

HEAD_CAMERA_SERIAL = "406122070550"
SUDO_PASS = os.environ.get("UNITREE_SUDO_PASS", "")

# ── 释放相机 ──────────────────────────────────────────
def free_camera():
    """尝试释放被占用的相机（杀 vision_node / videohub 等常见进程）。"""
    # 先尝试优雅停止 videohub
    subprocess.run(
        f"echo '{SUDO_PASS}' | sudo -S "
        f"/unitree/sbin/start-stop-daemon --stop "
        f"--pidfile=/unitree/var/run/videohub_pc4.pid "
        f"--exec /unitree/module/video_hub_pc4/videohub_pc4",
        shell=True, capture_output=True, text=True,
    )
    # 再强制杀可能占用相机的进程
    for name in ("videohub_pc4", "vision_node"):
        subprocess.run(
            f"echo '{SUDO_PASS}' | sudo -S pkill -9 -f {name}",
            shell=True, capture_output=True, text=True,
        )
    time.sleep(0.5)


def camera_is_free():
    try:
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(HEAD_CAMERA_SERIAL)
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipe.start(cfg)
        pipe.stop()
        return True
    except Exception:
        return False


# ── 拍照 ──────────────────────────────────────────────
def capture_photo(output_path="capture.jpg", preview=True, warmup_frames=50):
    """拍一张 RGB 照片并保存。"""

    # 1. 确保相机空闲
    if not camera_is_free():
        print("相机被占用，尝试释放...")
        free_camera()
        deadline = time.time() + 10
        while time.time() < deadline:
            if camera_is_free():
                print("  ✓ 相机已释放")
                break
            print(f"  等待... ({time.time() - deadline + 10:.1f}s)")
            free_camera()
            time.sleep(0.8)
        else:
            print("✗ 无法释放相机，请手动检查:")
            subprocess.run(["sudo", "lsof", "/dev/video0"])
            sys.exit(1)

    time.sleep(1.5)  # 硬件稳定

    # 2. 打开管道
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(HEAD_CAMERA_SERIAL)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    print("启动相机...")
    pipeline.start(config)
    time.sleep(2.0)  # 传感器冷启动

    # 3. 预热（让自动曝光收敛）
    print(f"预热（丢弃 {warmup_frames} 帧）...")
    for i in range(warmup_frames):
        pipeline.wait_for_frames(timeout_ms=8000)

    # 4. 抓一帧
    print("拍照...")
    frames = pipeline.wait_for_frames(timeout_ms=5000)
    color_frame = frames.get_color_frame()
    if not color_frame:
        print("✗ 获取彩色帧失败")
        pipeline.stop()
        sys.exit(1)

    color = np.asanyarray(color_frame.get_data())

    # 5. 保存
    cv2.imwrite(output_path, color)
    print(f"✓ 已保存: {output_path}  ({color.shape[1]}x{color.shape[0]})")

    # 6. 预览
    if preview:
        print("按任意键关闭预览窗口...")
        cv2.imshow("capture", color)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    pipeline.stop()
    return color


# ── main ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="G1 头部相机快速拍照")
    parser.add_argument("-o", "--output", default="capture.jpg", help="输出文件路径")
    parser.add_argument("--no-preview", action="store_true", help="不显示预览窗口")
    parser.add_argument("--warmup", type=int, default=50, help="预热帧数")
    args = parser.parse_args()

    capture_photo(
        output_path=args.output,
        preview=not args.no_preview,
        warmup_frames=args.warmup,
    )


if __name__ == "__main__":
    main()
