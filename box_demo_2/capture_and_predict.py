"""
采集宇树G1头部相机 RGB + 深度图，直接发送给 SAM3 服务端。

亮度与稳定性（与 capture_head_rgbd 一致）:
    50 帧预热 + 2s 冷启动等待，避免亮度忽明忽暗和取帧超时。
    可选 --manual-exposure 固定曝光。

用法:
    python capture_and_predict.py [--prompt box] [--host 192.168.112.103] [--port 5300]
    [--manual-exposure]
"""

import argparse
import base64
import io
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np
import pyrealsense2 as rs
import requests

# ──────────────────────────────────────────────
# 相机配置
# ──────────────────────────────────────────────
HEAD_CAMERA_SERIAL = os.environ.get("HEAD_CAMERA_SERIAL", "406122070550")
SUDO_PASS = os.environ.get("UNITREE_SUDO_PASS", "")

# 手动曝光参数（仅在 manual_exposure=True 时生效）
MANUAL_EXPOSURE = 150
MANUAL_GAIN = 100

STOP_CMD = (
    f"echo '{SUDO_PASS}' | sudo -S "
    f"/unitree/sbin/start-stop-daemon --stop "
    f"--pidfile=/unitree/var/run/videohub_pc4.pid "
    f"--exec /unitree/module/video_hub_pc4/videohub_pc4"
)
START_CMD = (
    f"echo '{SUDO_PASS}' | sudo -S "
    f"bash -c 'export CYCLONEDDS_URI=/unitree/module/video_hub_pc4/cyclonedds.xml; "
    f"/unitree/sbin/start-stop-daemon --start --background "
    f"--make-pidfile --pidfile=/unitree/var/run/videohub_pc4.pid "
    f"--exec /unitree/module/video_hub_pc4/videohub_pc4'"
)


# ──────────────────────────────────────────────
# 相机控制
# ──────────────────────────────────────────────
def _camera_is_free():
    try:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(HEAD_CAMERA_SERIAL)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)
        pipeline.stop()
        return True
    except Exception:
        return False


def stop_videohub(timeout=10.0):
    print("[1/3] 停止 videohub_pc4，等待相机释放...")
    subprocess.run(STOP_CMD, shell=True, capture_output=True, text=True)
    time.sleep(0.5)

    deadline = time.time() + timeout
    attempt = 0
    while time.time() < deadline:
        if _camera_is_free():
            print("  ✓ 相机已释放")
            time.sleep(1.5)   # 等待相机硬件完全稳定后再开管道
            return
        attempt += 1
        subprocess.run(STOP_CMD, shell=True, capture_output=True, text=True)
        subprocess.run(
            f"echo '{SUDO_PASS}' | sudo -S pkill -9 -x videohub_pc4",
            shell=True, capture_output=True, text=True,
        )
        print(f"  等待相机释放... (第{attempt}次, 已等{time.time()-deadline+timeout:.1f}s)")
        time.sleep(0.8)

    raise RuntimeError(f"等待 {timeout}s 后相机仍被占用")


def start_videohub():
    print("[3/3] 恢复 videohub_pc4...")
    ret = subprocess.run(START_CMD, shell=True, capture_output=True, text=True)
    if ret.returncode != 0:
        print(f"  警告: 恢复失败，请手动重启 videohub 或重启机器人")
    else:
        print("  ✓ videohub_pc4 已恢复")


# ──────────────────────────────────────────────
# 持久相机会话 — 只停一次 videohub，复用管道，消除每轮的冷启动开销
# ──────────────────────────────────────────────
class CameraSession:
    """持久的 RealSense 相机会话。

    用法:
        with CameraSession() as cam:
            color, depth = cam.capture()
            color2, depth2 = cam.capture()  # 复用同一管道，无需重启
    """

    def __init__(self, color_w=640, color_h=480, fps=30,
                 warmup_frames=50, manual_exposure=False,
                 camera_url: str | None = None, serial: str | None = None):
        self._color_w = color_w
        self._color_h = color_h
        self._fps = fps
        self._warmup_frames = warmup_frames
        self._manual_exposure = manual_exposure
        self._camera_url = camera_url or os.environ.get("CAMERA_URL", "")
        self._serial = serial or os.environ.get("HEAD_CAMERA_SERIAL", HEAD_CAMERA_SERIAL)
        self._pipeline = None
        self._align = None

    def __enter__(self):
        if self._camera_url:
            print(f"[CameraSession] 使用网络 RGBD 相机 {self._camera_url}")
            self.capture()
            print("网络相机就绪。")
            return self

        stop_videohub()
        self._pipeline = rs.pipeline()
        self._align = rs.align(rs.stream.color)
        config = rs.config()
        config.enable_device(self._serial)
        config.enable_stream(rs.stream.color, self._color_w, self._color_h,
                             rs.format.bgr8, self._fps)
        config.enable_stream(rs.stream.depth, self._color_w, self._color_h,
                             rs.format.z16, self._fps)
        profile = self._pipeline.start(config)
        time.sleep(2.0)  # 深度传感器冷启动

        if self._manual_exposure:
            color_sensor = profile.get_device().first_color_sensor()
            color_sensor.set_option(rs.option.enable_auto_exposure, 0)
            color_sensor.set_option(rs.option.exposure, MANUAL_EXPOSURE)
            color_sensor.set_option(rs.option.gain, MANUAL_GAIN)
            warmup_actual = max(self._warmup_frames, 30)
        else:
            warmup_actual = self._warmup_frames

        print(f"相机预热（丢弃 {warmup_actual} 帧）...")
        for _ in range(warmup_actual):
            self._pipeline.wait_for_frames(timeout_ms=8000)
        print("相机就绪。")
        return self

    def __exit__(self, *args):
        if self._camera_url:
            return

        if self._pipeline:
            try:
                self._pipeline.stop()
            except Exception:
                pass
        start_videohub()

    def capture(self):
        """从已就绪的管道抓一帧 RGB + 对齐深度图。"""
        if self._camera_url:
            resp = requests.get(self._camera_url, timeout=8)
            if resp.status_code != 200:
                raise RuntimeError(f"网络相机请求失败 [{resp.status_code}]: {resp.text[:200]}")
            data = np.load(io.BytesIO(resp.content))
            color = data["color"]
            depth = data["depth"]
            if color.ndim != 3 or color.shape[2] != 3:
                raise RuntimeError(f"网络相机 RGB 形状异常: {color.shape}")
            if depth.ndim != 2:
                raise RuntimeError(f"网络相机深度形状异常: {depth.shape}")
            if depth.dtype != np.uint16:
                depth = depth.astype(np.uint16)
            return color, depth

        frames = self._pipeline.wait_for_frames(timeout_ms=5000)
        aligned = self._align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            raise RuntimeError("未能获取有效帧")
        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())
        return color, depth


# ──────────────────────────────────────────────
# 相机采集（与 capture_head_rgbd 一致：50 帧预热、2s 冷启动、可选手动曝光、超时重试）
# ──────────────────────────────────────────────
def capture_rgbd(color_w=640, color_h=480, fps=30, warmup_frames=50,
                 manual_exposure=False, max_retries=4):
    """
    采集一帧 RGB + 对齐深度图，返回内存中的 numpy 数组。

    若 wait_for_frames 超时（相机冷启动偶发），自动重启管道重试。

    Returns:
        color : np.ndarray (H, W, 3) uint8 BGR
        depth : np.ndarray (H, W)   uint16 毫米(mm)
    """
    print("采集 RGB + 深度图...")
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
            time.sleep(2.0)

            if manual_exposure:
                color_sensor = profile.get_device().first_color_sensor()
                color_sensor.set_option(rs.option.enable_auto_exposure, 0)
                color_sensor.set_option(rs.option.exposure, MANUAL_EXPOSURE)
                color_sensor.set_option(rs.option.gain, MANUAL_GAIN)
                print(f"  已启用手动曝光: exposure={MANUAL_EXPOSURE}, gain={MANUAL_GAIN}")
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

            color = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data())
            print(f"  ✓ RGB {color.shape}  Depth {depth.shape}")
            return color, depth

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


# ──────────────────────────────────────────────
# SAM3 请求
# ──────────────────────────────────────────────
def _decode_sam3_image_b64(result: dict, key: str, *, is_jpeg: bool = True) -> np.ndarray | None:
    """从 SAM3 JSON 响应解码 base64 图像，返回 BGR uint8 数组。"""
    image_b64 = result.get(key)
    if not image_b64:
        return None
    image_bytes = base64.b64decode(image_b64)
    flag = cv2.IMREAD_COLOR
    image_rgb = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), flag)
    if image_rgb is None:
        return None
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


def _decode_sam3_overlay(result: dict) -> np.ndarray | None:
    return _decode_sam3_image_b64(result, "overlay_image", is_jpeg=True)


def _decode_sam3_html_field(result: dict, key: str) -> str | None:
    """从 SAM3 JSON 响应解码 base64 HTML 字段。"""
    html_b64 = result.get(key)
    if not html_b64:
        return None
    try:
        return base64.b64decode(html_b64).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return None


def _decode_sam3_pointcloud_html(result: dict) -> str | None:
    return _decode_sam3_html_field(result, "pointcloud_sor_dbscan_html")


def _attach_sam3_images(meta: dict, body: dict) -> None:
    overlay_before = _decode_sam3_image_b64(body, "overlay_before_erode_image", is_jpeg=True)
    if overlay_before is None:
        overlay_before = _decode_sam3_overlay(body)
    if overlay_before is not None:
        meta["overlay_before_erode"] = overlay_before
        meta["has_overlay_before_erode"] = True
        meta["overlay"] = overlay_before
        meta["has_overlay"] = True

    overlay_after = _decode_sam3_image_b64(body, "overlay_after_erode_image", is_jpeg=True)
    if overlay_after is not None:
        meta["overlay_after_erode"] = overlay_after
        meta["has_overlay_after_erode"] = True

    if "mask_erode_px" in body:
        meta["mask_erode_px"] = body["mask_erode_px"]

    touches = body.get("mask_touches")
    if isinstance(touches, dict):
        meta["mask_touches"] = touches

    for key, meta_key, flag in (
        ("pointcloud_initial_html", "pointcloud_initial_html", "has_pointcloud_initial"),
        ("pointcloud_sor_html", "pointcloud_sor_html", "has_pointcloud_sor"),
        ("pointcloud_sor_dbscan_html", "pointcloud_sor_dbscan_html", "has_pointcloud_sor_dbscan"),
    ):
        html = _decode_sam3_html_field(body, key)
        if html is not None:
            meta[meta_key] = html
            meta[flag] = True


def predict_from_arrays(color_bgr: np.ndarray, depth_u16: np.ndarray,
                        prompt: str, host: str, port: int,
                        point_cloud_stage: str,
                        return_meta: bool = False):
    """
    把内存中的 numpy 图像直接编码后发送给 SAM3 服务端，无需先写文件。

    Args:
        color_bgr : (H,W,3) uint8 BGR
        depth_u16 : (H,W)   uint16 毫米
        prompt    : 文本提示词，如 'box'
        host/port : SAM3 服务端地址
        point_cloud_stage: sor 或 sor_dbscan（SOR+DBSCAN），用于 OBB/RANSAC
        return_meta: 为 True 时返回 (result, meta)；result 失败时为 None

    Returns:
        result dict 或 None；return_meta=True 时 (result, meta)
    """
    # RGB 编码为 JPEG bytes
    ok, rgb_buf = cv2.imencode(".jpg", color_bgr)
    if not ok:
        raise RuntimeError("RGB 编码失败")

    # 深度图编码为 PNG bytes（保留 uint16 精度）
    ok, depth_buf = cv2.imencode(".png", depth_u16)
    if not ok:
        raise RuntimeError("Depth 编码失败")

    url = f"http://{host}:{port}/predict"
    files = {
        "rgb":   ("rgb.jpg",   rgb_buf.tobytes(),   "image/jpeg"),
        "depth": ("depth.png", depth_buf.tobytes(), "image/png"),
    }
    data = {
        "prompt": prompt,
        "point_cloud_stage": point_cloud_stage,
    }

    print(f"\n发送请求 → {url}  prompt='{prompt}'  point_cloud_stage='{point_cloud_stage}'")
    resp = requests.post(url, files=files, data=data, timeout=120)

    if resp.status_code != 200:
        print(f"请求失败 [{resp.status_code}]: {resp.text}")
        meta = {
            "ok": False,
            "status_code": resp.status_code,
            "error": resp.text,
            "prompt": prompt,
            "point_cloud_stage": point_cloud_stage,
            "url": url,
        }
        try:
            err_body = resp.json()
            if isinstance(err_body, dict):
                meta["error"] = err_body.get("detail", resp.text)
                _attach_sam3_images(meta, err_body)
        except (ValueError, requests.exceptions.JSONDecodeError):
            pass
        if return_meta:
            return None, meta
        return None

    result = resp.json()

    center      = result["center"]
    extent      = result["extent"]
    R           = result["rotation_matrix"]
    length      = result["length"]
    grasp_left  = result["grasp_left"]
    grasp_right = result["grasp_right"]

    print(f"\n=== 3D Bounding Box 结果 ===")
    print(f"中心坐标  (x, y, z): ({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}) m")
    print(f"尺寸      (e0,e1,e2): ({extent[0]:.4f}, {extent[1]:.4f}, {extent[2]:.4f}) m")
    print(f"旋转矩阵 R:")
    for row in R:
        print(f"  [{row[0]:8.4f}  {row[1]:8.4f}  {row[2]:8.4f}]")
    print(f"\n=== 抓取点信息 ===")
    print(f"箱子长度 (沿n3轴):    {length:.4f} m")
    print(f"左侧面中点 (左手抓取): ({grasp_left[0]:.4f}, {grasp_left[1]:.4f}, {grasp_left[2]:.4f}) m")
    print(f"右侧面中点 (右手抓取): ({grasp_right[0]:.4f}, {grasp_right[1]:.4f}, {grasp_right[2]:.4f}) m")

    meta = {
        "ok": True,
        "prompt": prompt,
        "point_cloud_stage": result.get("point_cloud_stage", point_cloud_stage),
        "url": url,
        "center": center,
        "extent": extent,
        "rotation_matrix": R,
        "length": length,
        "grasp_left": grasp_left,
        "grasp_right": grasp_right,
        "has_overlay": "overlay_image" in result or "overlay_before_erode_image" in result,
        "has_overlay_before_erode": (
            "overlay_before_erode_image" in result or "overlay_image" in result
        ),
        "has_overlay_after_erode": "overlay_after_erode_image" in result,
        "has_pointcloud_initial": "pointcloud_initial_html" in result,
        "has_pointcloud_sor": "pointcloud_sor_html" in result,
        "has_pointcloud_sor_dbscan": "pointcloud_sor_dbscan_html" in result,
    }
    if "mask_erode_px" in result:
        meta["mask_erode_px"] = result["mask_erode_px"]
    if "mask_touches" in result:
        meta["mask_touches"] = result["mask_touches"]
    _attach_sam3_images(meta, result)
    if return_meta:
        return result, meta
    return result


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="采集头部相机图像并发送给 SAM3 服务端")
    parser.add_argument("--prompt", default="box",              help="分割提示词，默认 box")
    parser.add_argument("--point-cloud-stage", required=True,
                        choices=("sor", "sor_dbscan"),
                        help="SAM3 OBB 点云阶段：sor 或 sor_dbscan（SOR+DBSCAN）")
    parser.add_argument("--host",   default="192.168.112.103",  help="SAM3 服务端 IP")
    parser.add_argument("--port",   default=5300, type=int,     help="SAM3 服务端端口")
    parser.add_argument("--manual-exposure", action="store_true",
                        help="使用固定曝光，避免亮度忽明忽暗")
    args = parser.parse_args()

    color, depth = capture_rgbd(manual_exposure=args.manual_exposure)
    predict_from_arrays(color, depth, args.prompt, args.host, args.port,
                        args.point_cloud_stage)


if __name__ == "__main__":
    main()
