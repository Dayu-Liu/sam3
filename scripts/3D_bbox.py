"""
3D Bounding Box 拟合脚本（RANSAC 平面拟合版）

流程：
1. 用 SAM3 从 RGB 图像中获取 box 的 mask
2. 结合深度图，将 mask 区域反投影为 3D 点云
3. 统计离群点移除（SOR）
4. RANSAC 依次拟合箱子的两个正交面，叉积得第三轴，构建 OBB

用法:
    python scripts/3D_bbox.py [--no-viz]
    --no-viz       : 无显示环境（如 SSH）下跳过可视化窗口
    --distance-thr : RANSAC 内点距离阈值（米），默认 0.01
    --ransac-iter  : RANSAC 迭代次数，默认 1000
    --orth-thr     : 判断两平面"足够正交"的阈值（|dot|<该值），默认 0.3
"""

import argparse
import os

import numpy as np
from PIL import Image
import torch

# Open3D 需单独安装: pip install open3d
try:
    import open3d as o3d
except ImportError:
    raise ImportError("请安装 Open3D: pip install open3d")

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# ============ 配置 ============
IMAGE_PATH = "/home/liuzihou/sam3/assets/images/16.png"
DEPTH_PATH = "/home/liuzihou/sam3/assets/images/16_d.png"
PROMPT = "box"
MODEL_DIR = "/home/liuzihou/sam3/model"

# 相机内参（用户标定值）
# 内参矩阵 K: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]

# #宇树体外相机
# INTRINSIC = np.array([
#     [387.00686646, 0.0, 319.60223389],
#     [0.0, 386.47955322, 242.79476929],
#     [0.0, 0.0, 1.0],
# ])

#宇树自带头部相机
INTRINSIC = np.array([
    [603.57, 0.0, 322.58],
    [0.0, 603.26, 250.15],
    [0.0, 0.0, 1.0],
])

FX = INTRINSIC[0, 0]
FY = INTRINSIC[1, 1]
CX = INTRINSIC[0, 2]
CY = INTRINSIC[1, 2]

print(FX,FY,CX,CY)

# 深度图缩放：uint16 深度值除以该数得到米，常见为 1000（毫米→米）
DEPTH_SCALE = 1000.0
# 无效深度阈值（米），超过则过滤
DEPTH_MAX = 10.0


def build_model():
    """构建 SAM3 模型"""
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
    # model = model.to("cuda:2")
    processor = Sam3Processor(model, confidence_threshold=0.5)
    return processor


def get_mask_from_sam3(processor, image_path: str, prompt: str):
    """
    用 SAM3 获取 prompt 对应区域的 mask。
    返回: (mask: np.ndarray bool (H,W), image: PIL.Image)
    """
    image = Image.open(image_path).convert("RGB")
    inference_state = processor.set_image(image)
    output = processor.set_text_prompt(state=inference_state, prompt=prompt)

    masks = output["masks"]
    scores = output["scores"]

    if masks is None or masks.numel() == 0:
        raise RuntimeError(
            f"No masks returned. Prompt '{prompt}' may not match anything in the image."
        )

    best_idx = scores.argmax().item()
    if masks.ndim == 4:
        mask = masks[best_idx].detach().cpu().numpy()
    elif masks.ndim == 3:
        mask = masks[best_idx].detach().cpu().numpy()
    else:
        mask = masks.detach().cpu().numpy()

    mask = np.squeeze(mask)
    if mask.ndim > 2:
        mask = mask[0]
    mask = mask.astype(bool)

    return mask, image


def depth_to_point_cloud(
    depth: np.ndarray,
    rgb: np.ndarray,
    mask: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    depth_scale: float,
    depth_max: float,
):
    """
    将深度图 + mask 反投影为 3D 点云（带颜色）。

    depth: (H, W) 深度图，单位由 depth_scale 决定
    rgb: (H, W, 3) RGB 图像
    mask: (H, W) 布尔数组，True 表示保留的像素
    """
    h, w = depth.shape
    u = np.arange(w, dtype=np.float32)
    v = np.arange(h, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    # 深度转米
    z = depth.astype(np.float32) / depth_scale

    # 反投影: x = (u - cx) * z / fx,  y = (v - cy) * z / fy
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy

    # 有效深度 + mask
    valid = (z > 0) & (z < depth_max) & mask

    points = np.stack([x[valid], y[valid], z[valid]], axis=-1)
    colors = rgb[valid] / 255.0  # [0, 1] 范围

    return points, colors


def remove_outliers(
    pcd: "o3d.geometry.PointCloud",
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
    dbscan_eps: float = 0.05,
    dbscan_min_points: int = 10,
) -> "o3d.geometry.PointCloud":
    """
    两阶段去噪：
    1. SOR（统计离群点移除）：去除孤立散点
    2. DBSCAN 聚类，只保留最大簇：去除远离主体的噪声片

    nb_neighbors      : SOR 邻居数（默认 20）
    std_ratio         : SOR 标准差倍数（默认 2.0）
    dbscan_eps        : DBSCAN 邻域半径，米（默认 0.05）
    dbscan_min_points : DBSCAN 最小簇点数（默认 10）
    """
    n_before = len(pcd.points)

    # 第一阶段：SOR
    pcd_sor, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio,
    )
    print(f"   SOR: {n_before} → {len(pcd_sor.points)} 点（移除 {n_before - len(pcd_sor.points)} 个）")

    # 第二阶段：体素降采样 → DBSCAN → 映射回原始点云
    # 降采样大幅减少点数，使 DBSCAN 快几十到几百倍
    voxel_size = dbscan_eps * 0.8   # 体素边长略小于 eps，保证降采样后点间距仍在 eps 范围内
    pcd_down = pcd_sor.voxel_down_sample(voxel_size=voxel_size)
    print(f"   降采样: {len(pcd_sor.points)} → {len(pcd_down.points)} 点（体素={voxel_size:.3f}m）")

    labels = np.array(pcd_down.cluster_dbscan(
        eps=dbscan_eps,
        min_points=dbscan_min_points,
        print_progress=False,
    ))
    if labels.max() < 0:
        print("   DBSCAN: 未找到有效簇，跳过")
        return pcd_sor

    # 找降采样后最大簇的中心
    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
    largest_label = unique[counts.argmax()]
    main_pts_down = np.asarray(pcd_down.points)[labels == largest_label]
    cluster_center = main_pts_down.mean(axis=0)
    cluster_radius = np.linalg.norm(main_pts_down - cluster_center, axis=1).max()

    # 在原始点云中保留在主簇范围内的点（加一点余量）
    all_pts = np.asarray(pcd_sor.points)
    dists = np.linalg.norm(all_pts - cluster_center, axis=1)
    main_indices = np.where(dists <= cluster_radius + voxel_size)[0]
    pcd_clean = pcd_sor.select_by_index(main_indices.tolist())

    n_clusters = len(unique)
    n_removed = len(pcd_sor.points) - len(pcd_clean.points)
    print(f"   DBSCAN: {n_clusters} 个簇，保留主簇 {len(pcd_clean.points)} 点（移除 {n_removed} 个噪声点）")
    return pcd_clean


def ransac_fit_one_plane(
    pcd: "o3d.geometry.PointCloud",
    distance_threshold: float,
    num_iterations: int,
):
    """
    用 RANSAC 在 pcd 中拟合一个平面。
    返回 (法向量 np.ndarray(3,), 内点索引列表, 剩余点云)
    """
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=3,
        num_iterations=num_iterations,
    )
    a, b, c, _ = plane_model
    normal = np.array([a, b, c], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    remaining = pcd.select_by_index(inliers, invert=True)
    return normal, inliers, remaining


def fit_obb_from_ransac_planes(
    pcd: "o3d.geometry.PointCloud",
    distance_threshold: float = 0.01,
    num_iterations: int = 1000,
    orth_threshold: float = 0.3,
):
    """
    用 RANSAC 拟合箱子的两个正交平面，叉积得第三轴，投影全部点云构建 OBB。

    算法：
      1. RANSAC 拟合最大平面 → 法向量 n1
      2. 从剩余点中循环拟合平面，取第一个与 n1 足够正交的（|dot|<orth_threshold）→ n2
      3. n2 做 Gram-Schmidt 正交化保证与 n1 严格正交
      4. n3 = n1 × n2
      5. 旋转矩阵 R = [n1 | n2 | n3]，将全部点投影到 R 坐标系
      6. 各轴 min/max 确定 OBB 尺寸和中心

    返回: (obb, n1, n2, n3, plane1_orig_indices, plane2_orig_indices)
      plane*_orig_indices 为各平面内点在 pcd 中的索引（用于主函数混色）
    """
    all_pts = np.asarray(pcd.points)
    n_pts = len(all_pts)

    # 用 remaining_idx 追踪"当前剩余点"在原始 pcd 中的索引
    remaining_idx = np.arange(n_pts)
    remaining_pcd = pcd

    # ---- 第一个面（最大面）----
    plane_model1, local_inliers1 = remaining_pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=3,
        num_iterations=num_iterations,
    )
    a, b, c, _ = plane_model1
    n1 = np.array([a, b, c], dtype=np.float64)
    n1 /= np.linalg.norm(n1)

    orig_inliers1 = remaining_idx[np.array(local_inliers1)]
    # 从 remaining_idx 中移除平面1内点
    keep_mask = np.ones(len(remaining_idx), dtype=bool)
    keep_mask[np.array(local_inliers1)] = False
    remaining_idx = remaining_idx[keep_mask]
    remaining_pcd = pcd.select_by_index(remaining_idx.tolist())

    print(f"   平面1: 法向量=({n1[0]:.3f},{n1[1]:.3f},{n1[2]:.3f})  内点={len(orig_inliers1)}")

    # ---- 第二个面（与第一个面正交）----
    n2 = None
    orig_inliers2 = None
    attempt = 0
    max_attempts = 8

    while attempt < max_attempts and len(remaining_pcd.points) >= 10:
        plane_model_cand, local_inliers_cand = remaining_pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=3,
            num_iterations=num_iterations,
        )
        a, b, c, _ = plane_model_cand
        n_cand = np.array([a, b, c], dtype=np.float64)
        n_cand /= np.linalg.norm(n_cand)
        dot = abs(float(np.dot(n1, n_cand)))
        attempt += 1
        print(f"   平面2 尝试{attempt}: |dot(n1,n2)|={dot:.3f}  内点={len(local_inliers_cand)}")

        orig_inliers_cand = remaining_idx[np.array(local_inliers_cand)]

        if dot < orth_threshold:
            n2 = n_cand
            orig_inliers2 = orig_inliers_cand
            break

        # 该面与 n1 近似平行（对面），跳过继续找
        keep_mask = np.ones(len(remaining_idx), dtype=bool)
        keep_mask[np.array(local_inliers_cand)] = False
        remaining_idx = remaining_idx[keep_mask]
        remaining_pcd = pcd.select_by_index(remaining_idx.tolist())

    if n2 is None:
        raise RuntimeError(
            f"经过 {max_attempts} 次尝试仍未找到与平面1正交的第二个面。\n"
            "建议: 增大 --orth-thr 或 --distance-thr，或检查点云是否包含足够多面。"
        )

    # ---- 正交化（Gram-Schmidt）----
    n2 = n2 - np.dot(n2, n1) * n1
    n2 /= np.linalg.norm(n2)

    # 第三轴
    n3 = np.cross(n1, n2)
    n3 /= np.linalg.norm(n3)

    print(f"   正交轴: n1=({n1[0]:.3f},{n1[1]:.3f},{n1[2]:.3f})")
    print(f"           n2=({n2[0]:.3f},{n2[1]:.3f},{n2[2]:.3f})")
    print(f"           n3=({n3[0]:.3f},{n3[1]:.3f},{n3[2]:.3f})")

    # ---- 投影所有点，求 OBB ----
    R = np.column_stack([n1, n2, n3])
    proj = all_pts @ R

    mn = proj.min(axis=0)
    mx = proj.max(axis=0)
    extent = mx - mn
    center = R @ ((mn + mx) / 2.0)

    obb = o3d.geometry.OrientedBoundingBox(center=center, R=R, extent=extent)
    obb.color = (1, 0, 0)

    return obb, n1, n2, n3, orig_inliers1, orig_inliers2


def build_pcd_with_denoising(
    points: np.ndarray,
    colors: np.ndarray,
    nb_neighbors: int,
    std_ratio: float,
    dbscan_eps: float = 0.05,
    dbscan_min_points: int = 10,
):
    """构建点云并做两阶段去噪（SOR + DBSCAN），返回干净的点云。"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    print("   两阶段去噪（SOR + DBSCAN）...")
    pcd_clean = remove_outliers(
        pcd,
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio,
        dbscan_eps=dbscan_eps,
        dbscan_min_points=dbscan_min_points,
    )
    return pcd_clean


def main():
    parser = argparse.ArgumentParser(description="3D Bounding Box 拟合")
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="无显示环境时跳过 Open3D 可视化窗口",
    )
    parser.add_argument(
        "--nb-neighbors", type=int, default=20,
        help="SOR 离群点过滤：邻居数，默认 20",
    )
    parser.add_argument(
        "--std-ratio", type=float, default=2.0,
        help="SOR 离群点过滤：标准差倍数，默认 2.0",
    )
    parser.add_argument(
        "--dbscan-eps", type=float, default=0.05,
        help="DBSCAN 邻域半径（米），默认 0.05",
    )
    parser.add_argument(
        "--dbscan-min-points", type=int, default=10,
        help="DBSCAN 最小簇点数，默认 10",
    )
    parser.add_argument(
        "--distance-thr", type=float, default=0.01,
        help="RANSAC 内点距离阈值（米），默认 0.01",
    )
    parser.add_argument(
        "--ransac-iter", type=int, default=1000,
        help="RANSAC 迭代次数，默认 1000",
    )
    parser.add_argument(
        "--orth-thr", type=float, default=0.3,
        help="判断两平面正交的阈值 |dot(n1,n2)|<该值，默认 0.3",
    )
    args = parser.parse_args()

    print("1. 加载 SAM3 模型...")
    processor = build_model()

    print("2. 获取 box 的 mask...")
    mask, image = get_mask_from_sam3(processor, IMAGE_PATH, PROMPT)
    rgb_np = np.array(image)
    h_rgb, w_rgb = rgb_np.shape[:2]

    print("3. 加载并对齐深度图...")
    depth_img = Image.open(DEPTH_PATH)
    depth_np = np.array(depth_img)

    # 深度图与 RGB 尺寸可能不同，需对齐
    if depth_np.shape[:2] != (h_rgb, w_rgb):
        depth_pil = Image.fromarray(depth_np)
        depth_pil = depth_pil.resize((w_rgb, h_rgb), Image.NEAREST)
        depth_np = np.array(depth_pil)
        print(f"   深度图已从 {depth_img.size} 缩放到 ({w_rgb}, {h_rgb})")

    print("4. 反投影为 3D 点云...")
    points, colors = depth_to_point_cloud(
        depth=depth_np,
        rgb=rgb_np,
        mask=mask,
        fx=FX,
        fy=FY,
        cx=CX,
        cy=CY,
        depth_scale=DEPTH_SCALE,
        depth_max=DEPTH_MAX,
    )

    if len(points) == 0:
        print("错误: mask 区域内无有效深度，请检查深度图与相机内参。")
        return

    print(f"   有效点数: {len(points)}")

    print("5. SOR 去噪...")
    pcd = build_pcd_with_denoising(
        points, colors,
        nb_neighbors=args.nb_neighbors,
        std_ratio=args.std_ratio,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_points=args.dbscan_min_points,
    )

    print("6. RANSAC 拟合平面 → 构建 OBB...")
    obb, n1, n2, n3, plane1_idx, plane2_idx = fit_obb_from_ransac_planes(
        pcd,
        distance_threshold=args.distance_thr,
        num_iterations=args.ransac_iter,
        orth_threshold=args.orth_thr,
    )

    center = obb.center
    extent = obb.extent
    print(f"\n=== 3D Bounding Box 结果（RANSAC）===")
    print(f"中心:     ({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f})")
    print(f"尺寸(长宽高): ({extent[0]:.4f}, {extent[1]:.4f}, {extent[2]:.4f}) m")
    print(f"旋转矩阵 R:\n{obb.R}")

    pcd_viz = pcd

    if not args.no_viz:
        print("\n7. 打开 Open3D 可视化窗口...")
        print("   原始 RGB 颜色点云 + 红色框 = OBB")
        try:
            o3d.visualization.draw_geometries(
                [pcd_viz, obb],
                window_name="RANSAC 平面拟合 + OBB",
                width=1024,
                height=768,
            )
        except Exception as e:
            print(f"   可视化失败（可能无显示）: {e}")
            print("   使用 --no-viz 可跳过可视化")
    else:
        print("\n7. 已跳过可视化 (--no-viz)")

    # 保存点云（保存带混色标注的版本）
    out_pcd_path = os.path.splitext(IMAGE_PATH)[0] + "_box_pointcloud.ply"
    o3d.io.write_point_cloud(out_pcd_path, pcd_viz)
    print(f"\n点云已保存: {out_pcd_path}")


if __name__ == "__main__":
    main()
