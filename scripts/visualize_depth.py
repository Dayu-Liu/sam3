"""
深度图可视化脚本

将 uint16 深度图用伪彩色（colormap）可视化后保存。

用法:
    python scripts/visualize_depth.py <depth_image_path> [--colormap COLORMAP]

    depth_image_path: 深度图路径（支持 .png / .tiff 等）
    --colormap: matplotlib colormap 名称，默认 plasma
               常用: plasma, viridis, inferno, jet, turbo, magma

示例:
    python scripts/visualize_depth.py assets/images/case2_d.png
    python scripts/visualize_depth.py assets/images/case2_d.png --colormap turbo
"""

import argparse
import os

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")  # 无头模式，不弹窗
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def visualize_depth(depth_path: str, colormap: str = "plasma"):
    """
    加载深度图，归一化后用伪彩色渲染，保存可视化结果。

    保存两张图：
    - *_vis.png      : 纯伪彩色（无坐标轴，适合作为素材）
    - *_vis_bar.png  : 带颜色条和标题，方便查看数值
    """
    depth = np.array(Image.open(depth_path)).astype(np.float32)

    # 有效深度（排除 0 值）
    valid = depth > 0
    if not valid.any():
        print("警告: 深度图全为 0，无有效数据。")
        depth_norm = np.zeros_like(depth)
    else:
        d_min = depth[valid].min()
        d_max = depth[valid].max()
        print(f"深度范围: {d_min:.0f} ~ {d_max:.0f}（原始单位）")
        print(f"图像尺寸: {depth.shape[1]} x {depth.shape[0]}")

        # 归一化到 [0, 1]，无效点设为 0
        depth_norm = np.where(valid, (depth - d_min) / (d_max - d_min), 0.0)

    cmap = matplotlib.colormaps.get_cmap(colormap)
    colored = cmap(depth_norm)  # (H, W, 4) RGBA, float [0,1]
    # 无效深度设为黑色
    colored[~valid] = [0, 0, 0, 1]

    colored_uint8 = (colored[:, :, :3] * 255).astype(np.uint8)

    base = os.path.splitext(depth_path)[0]

    # 1. 纯伪彩色图（无坐标轴）
    out_plain = base + f"_vis.png"
    Image.fromarray(colored_uint8).save(out_plain)
    print(f"已保存: {out_plain}")

    # 2. 带颜色条的图
    out_bar = base + f"_vis_bar.png"
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    im = ax.imshow(depth_norm, cmap=colormap, vmin=0, vmax=1)
    ax.set_title(f"Depth Map  ({os.path.basename(depth_path)})", fontsize=12)
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Depth (normalized)", fontsize=10)
    # 在颜色条上显示原始数值
    if valid.any():
        ticks = np.linspace(0, 1, 6)
        raw_vals = ticks * (d_max - d_min) + d_min
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{v:.0f}" for v in raw_vals])
    plt.tight_layout()
    plt.savefig(out_bar, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存: {out_bar}")


def main():
    parser = argparse.ArgumentParser(description="深度图伪彩色可视化")
    parser.add_argument("depth_path", help="深度图路径")
    parser.add_argument(
        "--colormap",
        default="plasma",
        help="matplotlib colormap，默认 plasma（可选 viridis/inferno/jet/turbo/magma）",
    )
    args = parser.parse_args()
    visualize_depth(args.depth_path, args.colormap)


if __name__ == "__main__":
    main()
