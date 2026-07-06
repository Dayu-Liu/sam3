"""
将 SAM3 生成的 mask 叠加到深度可视化图上。

自动根据 RGB 图像路径推断深度可视化图路径：
    8.jpg  ->  8_d_vis.png

用法:
    python scripts/mask_on_depth.py <image_path> [--prompt PROMPT]

    image_path : RGB 图像路径（如 assets/images/8.jpg）
    --prompt   : 分割提示词，默认 box

示例:
    python scripts/mask_on_depth.py assets/images/8.jpg
    python scripts/mask_on_depth.py assets/images/8.jpg --prompt cup
"""

import argparse
import os

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


MODEL_DIR = "/home/liuzihou/sam3/model"


def build_model():
    checkpoint_path = MODEL_DIR
    if os.path.isdir(MODEL_DIR):
        checkpoint_path = os.path.join(MODEL_DIR, "sam3.pt")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = build_sam3_image_model(
        checkpoint_path=checkpoint_path,
        load_from_HF=False,
        device="cuda",
    )
    return Sam3Processor(model, confidence_threshold=0.5)


def get_mask(processor, image_path: str, prompt: str):
    """返回得分最高的 mask，形状 (H, W) bool，H/W 为 RGB 图尺寸。"""
    image = Image.open(image_path).convert("RGB")
    state = processor.set_image(image)
    output = processor.set_text_prompt(state=state, prompt=prompt)

    masks = output["masks"]
    scores = output["scores"]

    if masks is None or masks.numel() == 0:
        raise RuntimeError(f"未找到任何 mask，prompt='{prompt}' 可能与图像不匹配。")

    best_idx = scores.argmax().item()
    mask = masks[best_idx].detach().cpu().numpy()
    mask = np.squeeze(mask)
    if mask.ndim > 2:
        mask = mask[0]
    return mask.astype(bool), image


def draw_mask_on_depth_vis(mask_rgb, depth_vis_path, out_path, prompt):
    """
    将 mask 缩放到深度可视化图尺寸后叠加。

    mask_rgb   : (H_rgb, W_rgb) bool
    depth_vis  : (H_d, W_d, 3) uint8 RGB 可视化图
    """
    depth_vis = np.array(Image.open(depth_vis_path).convert("RGB"))
    h_d, w_d = depth_vis.shape[:2]

    # 将 mask 缩放到深度图尺寸（最近邻，保持二值）
    mask_pil = Image.fromarray(mask_rgb.astype(np.uint8) * 255)
    mask_resized = np.array(
        mask_pil.resize((w_d, h_d), Image.NEAREST)
    ) > 127

    # 在深度可视化图上叠加半透明绿色
    overlay = depth_vis.copy().astype(np.float32)
    green = np.array([0, 255, 0], dtype=np.float32)
    overlay[mask_resized] = 0.5 * overlay[mask_resized] + 0.5 * green
    overlay = overlay.clip(0, 255).astype(np.uint8)

    # 绘制 mask 轮廓（纯绿色边缘）
    from PIL import ImageFilter
    mask_img = Image.fromarray(mask_resized.astype(np.uint8) * 255)
    edge = np.array(mask_img.filter(ImageFilter.FIND_EDGES)) > 0
    overlay[edge] = [0, 255, 0]

    # 保存
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    axes[0].imshow(depth_vis)
    axes[0].set_title("Depth Vis (original)", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title(f"Depth Vis + Mask  (prompt: '{prompt}')", fontsize=11)
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存: {out_path}")

    # 同时保存纯叠加图（不带对比）
    plain_path = out_path.replace("_comparison", "").replace(".png", "_plain.png")
    Image.fromarray(overlay).save(plain_path)
    print(f"已保存: {plain_path}")


def main():
    parser = argparse.ArgumentParser(description="将 SAM3 mask 叠加到深度可视化图上")
    parser.add_argument("image_path", help="RGB 图像路径（如 assets/images/8.jpg）")
    parser.add_argument("--prompt", default="box", help="分割提示词，默认 box")
    args = parser.parse_args()

    image_path = args.image_path
    prompt = args.prompt

    # 推断深度可视化图路径：8.jpg -> 8_d_vis.png
    base = os.path.splitext(image_path)[0]
    depth_vis_path = base + "_d_vis.png"
    if not os.path.exists(depth_vis_path):
        raise FileNotFoundError(
            f"找不到深度可视化图: {depth_vis_path}\n"
            f"请先运行 visualize_depth.py 生成 _d_vis.png"
        )

    out_path = base + "_mask_on_depth_comparison.png"

    print("1. 加载 SAM3 模型...")
    processor = build_model()

    print(f"2. 获取 mask（prompt='{prompt}'）...")
    mask, _ = get_mask(processor, image_path, prompt)
    print(f"   mask 形状: {mask.shape}, 前景像素: {mask.sum()}")

    print("3. 将 mask 叠加到深度可视化图...")
    draw_mask_on_depth_vis(mask, depth_vis_path, out_path, prompt)


if __name__ == "__main__":
    main()
