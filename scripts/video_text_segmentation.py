# Copyright (c) Meta Platforms, Inc. and affiliates.
"""使用 SAM 3 对整段视频做文本指令分割，并在时间上传播掩码。

用法示例::

    python scripts/video_text_segmentation.py

默认使用你提供的数据路径与提示词；也可通过命令行覆盖。
"""
from __future__ import annotations

import argparse
import colorsys
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from sam3.model_builder import build_sam3_video_predictor


def load_frame(frame: np.ndarray | str) -> np.ndarray:
    """与 notebook 中用法一致：本脚本实际只传入 RGB ndarray 或 JPEG 路径。"""
    if isinstance(frame, np.ndarray):
        return frame
    if isinstance(frame, str) and os.path.isfile(frame):
        bgr = cv2.imread(frame, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"无法读取图像: {frame}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    raise ValueError(f"不支持的帧类型: {type(frame)}")


def _obj_color_rgb(obj_id: int) -> np.ndarray:
    """确定性配色，不依赖 scikit-image / sklearn。"""
    h = (obj_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
    return np.array([int(r * 255), int(g * 255), int(b * 255)], dtype=np.uint8)


def render_masklet_frame(img: np.ndarray, outputs: dict, frame_idx: int | None = None, alpha: float = 0.5):
    """与 sam3.visualization_utils.render_masklet_frame 等价逻辑，仅用 numpy/cv2。"""
    if img.dtype == np.float32 or img.max() <= 1.0:
        img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
    img = np.ascontiguousarray(img[..., :3])
    height, width = img.shape[:2]
    overlay = img.copy()

    n = len(outputs["out_probs"])
    for i in range(n):
        obj_id = int(outputs["out_obj_ids"][i])
        color255 = _obj_color_rgb(obj_id)
        mask = outputs["out_binary_masks"][i]
        if mask.shape != img.shape[:2]:
            mask = cv2.resize(
                mask.astype(np.float32),
                (img.shape[1], img.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        mask_bool = mask > 0.5
        for c in range(3):
            overlay[..., c][mask_bool] = (
                alpha * float(color255[c]) + (1.0 - alpha) * overlay[..., c][mask_bool].astype(np.float32)
            ).astype(np.uint8)

    for i in range(n):
        box_xywh = outputs["out_boxes_xywh"][i]
        obj_id = int(outputs["out_obj_ids"][i])
        prob = outputs["out_probs"][i]
        color255 = tuple(int(x) for x in _obj_color_rgb(obj_id))
        x, y, w, h = box_xywh
        x1 = int(float(x) * width)
        y1 = int(float(y) * height)
        x2 = int((float(x) + float(w)) * width)
        y2 = int((float(y) + float(h)) * height)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color255, 2)
        if prob is not None:
            label = f"id={obj_id}, p={float(prob):.2f}"
        else:
            label = f"id={obj_id}"
        cv2.putText(
            overlay,
            label,
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color255,
            1,
            cv2.LINE_AA,
        )

    if frame_idx is not None:
        cv2.putText(
            overlay,
            f"Frame {frame_idx}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return overlay

_DEFAULT_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CHECKPOINT = str(_DEFAULT_REPO_ROOT / "model" / "sam3.pt")

DEFAULT_VIDEO = (
    "/data/nvme0n1/zihou/VLAdata/task_buildblocks4.9/episodes/"
    "episode_200_20260409_151318/videos/wrist_cam_right.mp4"
)
DEFAULT_TEXT = "orange things"


def load_video_frames_rgb(video_path: str) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"无法打开视频: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps_f = float(fps) if fps and fps > 1e-3 else 10.0
    frames: list[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"未从视频中读取到任何帧: {video_path}")
    return frames, fps_f


def propagate_in_video(predictor, session_id: str) -> dict[int, dict]:
    outputs_per_frame: dict[int, dict] = {}
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]
    return outputs_per_frame


def outputs_to_render_dict(outputs: dict) -> dict:
    """将 GPU tensor 转为本模块 `render_masklet_frame` 可用的 numpy 字典。"""

    def tn(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return x

    masks = tn(outputs["out_binary_masks"])
    if masks.dtype == bool:
        masks = masks.astype(np.float32)
    return {
        "out_obj_ids": tn(outputs["out_obj_ids"]),
        "out_probs": tn(outputs["out_probs"]),
        "out_boxes_xywh": tn(outputs["out_boxes_xywh"]),
        "out_binary_masks": masks,
    }


def save_overlay_video(
    video_frames: list[np.ndarray],
    outputs_per_frame: dict[int, dict],
    out_path: str,
    fps: float,
    alpha: float = 0.5,
) -> None:
    """写出 MP4。使用 ffmpeg 编码为 H.264（yuv420p + faststart），与常见采集设备输出一致，便于 Cursor 等内置播放器播放。

    OpenCV 默认的 mp4v（MPEG-4 Part 2）在部分 Electron/浏览器内核中兼容性差。
    """
    parent = os.path.dirname(os.path.abspath(out_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    indices = sorted(outputs_per_frame.keys())
    first_idx = indices[0]
    first_img = load_frame(video_frames[first_idx])
    h, w = first_img.shape[:2]
    fps_f = float(fps)

    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise RuntimeError(
            "未找到 ffmpeg，无法输出 H.264。请安装 ffmpeg（例如 apt install ffmpeg），"
            "或自行将叠加帧序列用 ffmpeg/libx264 编码为 MP4。"
        )

    cmd = [
        ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps_f),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        out_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for fi in tqdm(indices, desc="写入叠加 visualization 视频"):
            vis = outputs_to_render_dict(outputs_per_frame[fi])
            img = load_frame(video_frames[fi])
            overlay = render_masklet_frame(img, vis, frame_idx=fi, alpha=alpha)
            frame_bgr = np.ascontiguousarray(
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR), dtype=np.uint8
            )
            if frame_bgr.shape[0] != h or frame_bgr.shape[1] != w:
                frame_bgr = np.ascontiguousarray(
                    cv2.resize(frame_bgr, (w, h)), dtype=np.uint8
                )
            proc.stdin.write(frame_bgr.tobytes())
    finally:
        if proc.stdin:
            proc.stdin.close()
    stderr = proc.stderr.read().decode("utf-8", errors="replace")
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"ffmpeg 编码失败 (exit {ret}): {stderr}")
def save_union_masks(
    video_frames: list[np.ndarray],
    outputs_per_frame: dict[int, dict],
    masks_dir: str,
) -> None:
    os.makedirs(masks_dir, exist_ok=True)
    for fi in tqdm(sorted(outputs_per_frame.keys()), desc="写入合并掩码 PNG"):
        o = outputs_per_frame[fi]
        masks = o["out_binary_masks"]
        if isinstance(masks, torch.Tensor):
            masks = masks.detach().cpu().numpy()
        h, w = video_frames[fi].shape[:2]
        if masks.size == 0:
            union = np.zeros((h, w), dtype=np.uint8)
        else:
            union = (np.any(masks, axis=0)).astype(np.uint8) * 255
        if union.shape != (h, w):
            union = cv2.resize(union, (w, h), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(masks_dir, f"{fi:05d}.png"), union)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SAM 3：自然语言指令驱动的视频分割与跟踪（文本提示 + 全视频传播）",
    )
    parser.add_argument("--video", type=str, default=DEFAULT_VIDEO, help="MP4 或 JPEG 帧目录")
    parser.add_argument("--text", type=str, default=DEFAULT_TEXT, help='文本提示，如 "orange triangle"')
    parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="添加文本提示的帧索引",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录；默认在视频同目录下创建 sam3_seg_<stem>_<text>/",
    )
    parser.add_argument("--alpha", type=float, default=0.5, help="掩码叠加透明度")
    parser.add_argument("--no-overlay-video", action="store_true", help="不写出 overlay.mp4")
    parser.add_argument(
        "--no-mask-pngs",
        action="store_true",
        help="不写出 masks_union/ 下每帧合并掩码",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help='使用的 GPU 编号，逗号分隔，如 "0" 或 "0,1"；默认使用全部可见 GPU',
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=_DEFAULT_CHECKPOINT,
        help="本地 SAM3 权重 sam3.pt 的路径；默认 <仓库根>/model/sam3.pt（仅从本地加载，不访问 Hugging Face）",
    )
    args = parser.parse_args()

    video_path = os.path.abspath(args.video)
    if os.path.isdir(video_path):
        resource_path = video_path
    elif os.path.isfile(video_path):
        resource_path = video_path
    else:
        sys.exit(f"路径不存在: {video_path}")

    stem = Path(video_path).stem if os.path.isfile(video_path) else Path(video_path).name
    safe_text = "_".join(args.text.lower().split())[:80]
    out_dir = args.output_dir or str(
        (Path(video_path).parent if os.path.isfile(video_path) else Path(video_path))
        / f"sam3_seg_{stem}_{safe_text}"
    )
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    if args.gpus is not None:
        gpus_to_use = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
    else:
        gpus_to_use = list(range(torch.cuda.device_count()))
    if not gpus_to_use:
        sys.exit("需要 CUDA：未发现可用 GPU")

    ckpt_path = os.path.abspath(args.checkpoint)
    if not os.path.isfile(ckpt_path):
        sys.exit(f"找不到本地权重文件: {ckpt_path}\n请将 sam3.pt 放到该路径，或用 --checkpoint 指定。")

    predictor = build_sam3_video_predictor(
        gpus_to_use=gpus_to_use,
        checkpoint_path=ckpt_path,
    )

    try:
        if resource_path.endswith(".mp4") or os.path.isfile(resource_path):
            video_frames, fps = load_video_frames_rgb(resource_path)
        else:
            import glob

            pattern = os.path.join(resource_path, "*.jpg")
            paths = glob.glob(pattern)
            try:
                paths.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
            except ValueError:
                paths.sort()
            if not paths:
                sys.exit(f"目录中未找到 JPG 帧: {resource_path}")
            video_frames = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in paths]
            fps = 10.0

        resp = predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=resource_path,
            )
        )
        session_id = resp["session_id"]

        predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=args.frame_index,
                text=args.text,
            )
        )

        outputs_per_frame = propagate_in_video(predictor, session_id)

        meta = {
            "video_path": resource_path,
            "text_prompt": args.text,
            "prompt_frame_index": args.frame_index,
            "num_frames": len(video_frames),
            "num_result_frames": len(outputs_per_frame),
            "fps": fps,
            "gpus_to_use": gpus_to_use,
        }
        if outputs_per_frame:
            fi0 = min(outputs_per_frame.keys())
            ids = outputs_per_frame[fi0]["out_obj_ids"]
            if torch.is_tensor(ids):
                ids = ids.detach().cpu().numpy()
            meta["object_ids_first_result_frame"] = np.asarray(ids).tolist()

        with open(os.path.join(out_dir, "run_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        if not args.no_overlay_video:
            save_overlay_video(
                video_frames,
                outputs_per_frame,
                os.path.join(out_dir, "overlay.mp4"),
                fps,
                alpha=args.alpha,
            )
        if not args.no_mask_pngs:
            save_union_masks(
                video_frames,
                outputs_per_frame,
                os.path.join(out_dir, "masks_union"),
            )

        predictor.handle_request(
            request=dict(type="close_session", session_id=session_id),
        )
    finally:
        predictor.shutdown()

    print(f"完成。结果目录: {out_dir}")


if __name__ == "__main__":
    main()
