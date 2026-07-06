"""
SAM3 3D Bounding Box 服务端

接收: RGB 图像、深度图、prompt
返回: OBB 中心坐标 + 三轴尺寸（JSON）

启动:
    conda run -n sam3 python scripts/sam_server.py
    或
    conda run -n sam3 uvicorn scripts.sam_server:app --host 0.0.0.0 --port 8000

接口:
    POST /predict
        Form 字段:
            rgb   : RGB 图像文件（jpg/png）
            depth : 深度图文件（png，uint16）
            prompt: 文本提示词（默认 "box"）
            point_cloud_stage: 点云阶段，必填。sor 或 sor_dbscan（SOR+DBSCAN）

        返回 JSON:
            {
                "center": [cx, cy, cz],        # 3D 中心坐标（米）
                "extent": [e0, e1, e2],        # 三轴尺寸（米）
                "rotation_matrix": [[...],[...],[...]]  # 3x3 旋转矩阵
            }

    GET /health
        返回服务状态
"""

import base64
import io
import json
import os
import logging

import cv2
import numpy as np
from PIL import Image, ImageFilter
import torch

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse

try:
    import open3d as o3d
except ImportError:
    raise ImportError("请安装 Open3D: pip install open3d")

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


# ============ 服务配置 ============
HOST = "0.0.0.0"
PORT = 5300
# 默认使用本仓库下的 model/；可用环境变量 SAM3_MODEL_DIR 覆盖（绝对路径）
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.environ.get("SAM3_MODEL_DIR", os.path.join(_REPO_ROOT, "model"))

# 相机内参
FX = 603.57
FY = 603.26
CX = 322.58
CY = 250.15

# 深度图参数
DEPTH_SCALE = 1000.0   # uint16 → 米
DEPTH_MAX   = 10.0     # 最大有效深度（米）

# SOR 去噪参数
NB_NEIGHBORS = 20
STD_RATIO    = 2.0

# mask 内缩（剔除边缘不可靠深度）
MASK_ERODE_PX        = 5    # 内缩像素数；边缘带不参与反投影
MASK_CORE_MIN_PIXELS = 80   # 内缩后 mask 最少像素，不足则逐步减小内缩

# DBSCAN 聚类参数（用于过滤聚集的噪声片）
DBSCAN_EPS        = 0.05   # 邻域半径（米），点间距小于此值视为同一簇
DBSCAN_MIN_POINTS = 10     # 最小簇点数，点数不足视为噪声

# 多簇合并：保留箱子多个可见面，剔除远离主体的货架/背景簇
MIN_CLUSTER_RATIO      = 0.05   # 次簇至少为最大簇点数的该比例
CLUSTER_MERGE_DIST       = 0.20 # 两簇质心距离上限（米），视为同一箱子
MAX_DEPTH_MEDIAN_DIFF    = 0.25 # 两簇深度中位数差上限（米）
MIN_KEEP_RATIO           = 0.35 # SOR 点保留率低于该值时放宽回收

# RANSAC 参数
DISTANCE_THR  = 0.01
RANSAC_ITER   = 1000
ORTH_THR      = 0.3


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="SAM3 3D BBox Server")

# 全局 processor（启动时加载一次，避免每次请求重新加载模型）
_processor: Sam3Processor = None


# ============ 模型加载 ============

def load_processor() -> Sam3Processor:
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
    logger.info(f"SAM3 实际设备: {next(model.parameters()).device}")
    return Sam3Processor(model, confidence_threshold=0.5, device="cuda")


@app.on_event("startup")
async def startup_event():
    global _processor
    logger.info("正在加载 SAM3 模型...")
    _processor = load_processor()
    logger.info("SAM3 模型加载完成，服务就绪。")


# ============ 核心计算函数（复用自 3D_bbox.py）============

def make_mask_overlay(rgb_np: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """在 RGB 图上叠加半透明绿色 mask，返回 uint8 RGB。"""
    h, w = rgb_np.shape[:2]
    mask_bool = mask.astype(bool)
    if mask_bool.shape != (h, w):
        mask_pil = Image.fromarray(mask_bool.astype(np.uint8) * 255)
        mask_bool = np.array(mask_pil.resize((w, h), Image.NEAREST)) > 127

    overlay = rgb_np.copy().astype(np.float32)
    green = np.array([0, 255, 0], dtype=np.float32)
    overlay[mask_bool] = 0.5 * overlay[mask_bool] + 0.5 * green

    edge = np.array(
        Image.fromarray(mask_bool.astype(np.uint8) * 255).filter(ImageFilter.FIND_EDGES)
    ) > 0
    overlay[edge] = green
    return overlay.clip(0, 255).astype(np.uint8)


def encode_image_jpeg_b64(image_rgb: np.ndarray, quality: int = 85) -> str:
    """将 RGB 图像编码为 base64 JPEG 字符串。"""
    buf = io.BytesIO()
    Image.fromarray(image_rgb).save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def encode_image_png_b64(image_rgb: np.ndarray) -> str:
    """将 RGB 图像编码为 base64 PNG 字符串。"""
    buf = io.BytesIO()
    Image.fromarray(image_rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def encode_overlay_jpeg_b64(overlay_rgb: np.ndarray, quality: int = 85) -> str:
    """将 RGB overlay 编码为 base64 JPEG 字符串。"""
    return encode_image_jpeg_b64(overlay_rgb, quality=quality)


def _obb_corners(center: np.ndarray, extent: np.ndarray, R: np.ndarray) -> np.ndarray:
    """返回 OBB 8 个角点，形状 (8, 3)。"""
    half = extent / 2.0
    local = np.array([
        [-half[0], -half[1], -half[2]],
        [ half[0], -half[1], -half[2]],
        [ half[0],  half[1], -half[2]],
        [-half[0],  half[1], -half[2]],
        [-half[0], -half[1],  half[2]],
        [ half[0], -half[1],  half[2]],
        [ half[0],  half[1],  half[2]],
        [-half[0],  half[1],  half[2]],
    ], dtype=np.float64)
    return local @ R.T + center


def encode_text_b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def _sample_pointcloud_arrays(
    pcd: o3d.geometry.PointCloud,
    max_points: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(pcd.points, dtype=np.float32)
    if len(pts) == 0:
        raise ValueError("点云为空，无法渲染。")
    cols = np.asarray(pcd.colors, dtype=np.float32)
    if cols.shape[0] != pts.shape[0]:
        cols = np.full((len(pts), 3), 0.6, dtype=np.float32)
    if cols.max() <= 1.0:
        cols = np.clip(cols, 0.0, 1.0)
    else:
        cols = np.clip(cols / 255.0, 0.0, 1.0)
    if max_points is not None and len(pts) > max_points:
        idx = np.linspace(0, len(pts) - 1, max_points, dtype=int)
        pts = pts[idx]
        cols = cols[idx]
    return pts, cols


def build_pointcloud_html(
    pcd: o3d.geometry.PointCloud,
    *,
    center: np.ndarray | None = None,
    extent: np.ndarray | None = None,
    R: np.ndarray | None = None,
    max_points: int | None = None,
    title: str = "SAM3 Point Cloud",
    hud_text: str = "左键旋转 · 右键平移 · 滚轮缩放 · RGB 点云",
) -> str:
    """生成交互式 RGB 点云 HTML（Three.js，浏览器可拖动旋转）。"""
    pts, cols = _sample_pointcloud_arrays(pcd, max_points=max_points)
    pos_b64 = base64.b64encode(pts.tobytes()).decode("ascii")
    col_b64 = base64.b64encode((cols * 255.0).astype(np.uint8).tobytes()).decode("ascii")

    obb_json = "null"
    if center is not None and extent is not None and R is not None:
        corners = _obb_corners(center, extent, R).tolist()
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7],
        ]
        obb_json = json.dumps({"corners": corners, "edges": edges})

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    html, body {{ margin: 0; height: 100%; overflow: hidden; background: #111; }}
    #hud {{
      position: absolute; top: 10px; left: 10px; z-index: 2;
      color: #fff; background: rgba(0,0,0,.55); padding: 8px 12px;
      border-radius: 6px; font: 13px/1.4 sans-serif;
    }}
    canvas {{ display: block; }}
  </style>
</head>
<body>
  <div id="hud">{hud_text}</div>
  <script type="importmap">
  {{
    "imports": {{
      "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
      "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
    }}
  }}
  </script>
  <script type="module">
    import * as THREE from "three";
    import {{ OrbitControls }} from "three/addons/controls/OrbitControls.js";

    const NUM_POINTS = {len(pts)};
    const POS_B64 = "{pos_b64}";
    const COL_B64 = "{col_b64}";
    const OBB = {obb_json};

    function b64ToArrayBuffer(b64) {{
      const binary = atob(b64);
      const len = binary.length;
      const bytes = new Uint8Array(len);
      for (let i = 0; i < len; i++) bytes[i] = binary.charCodeAt(i);
      return bytes.buffer;
    }}

    const renderer = new THREE.WebGLRenderer({{ antialias: true }});
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111111);

    const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.01, 20);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;

    const positions = new Float32Array(b64ToArrayBuffer(POS_B64));
    const colors = new Uint8Array(b64ToArrayBuffer(COL_B64));
    const geometry = new THREE.BufferGeometry();
    const threePos = new Float32Array(NUM_POINTS * 3);
    for (let i = 0; i < NUM_POINTS; i++) {{
      const x = positions[i * 3];
      const y = positions[i * 3 + 1];
      const z = positions[i * 3 + 2];
      // 相机光学系 (X右 Y下 Z前) → Three.js 世界系 (Y上)
      // lookAt 从 -Z 侧朝 +Z 看时，屏幕右轴 = 世界 -X，故 X 取反才能与 color/overlay 左右一致
      threePos[i * 3] = -x;
      threePos[i * 3 + 1] = -y;
      threePos[i * 3 + 2] = z;
    }}
    const threeCol = new Float32Array(NUM_POINTS * 3);
    for (let i = 0; i < NUM_POINTS; i++) {{
      threeCol[i * 3] = colors[i * 3] / 255;
      threeCol[i * 3 + 1] = colors[i * 3 + 1] / 255;
      threeCol[i * 3 + 2] = colors[i * 3 + 2] / 255;
    }}
    geometry.setAttribute("position", new THREE.BufferAttribute(threePos, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(threeCol, 3));
    const material = new THREE.PointsMaterial({{
      size: 0.007,
      sizeAttenuation: true,
      vertexColors: true,
    }});
    scene.add(new THREE.Points(geometry, material));

    if (OBB) {{
      const obbGroup = new THREE.Group();
      const lineMat = new THREE.LineBasicMaterial({{ color: 0xff3333 }});
      for (const [a, b] of OBB.edges) {{
        const ca = OBB.corners[a];
        const cb = OBB.corners[b];
        const pts = new Float32Array([
          -ca[0], -ca[1], ca[2],
          -cb[0], -cb[1], cb[2],
        ]);
        const g = new THREE.BufferGeometry();
        g.setAttribute("position", new THREE.BufferAttribute(pts, 3));
        obbGroup.add(new THREE.Line(g, lineMat));
      }}
      scene.add(obbGroup);
    }}

    geometry.computeBoundingSphere();
    const c = geometry.boundingSphere.center;
    const r = geometry.boundingSphere.radius;
    // 从相机侧（光学系 Z≈0）朝 +Z 看场景，与 RealSense 视角一致；勿放在 z+r 侧（会从物体背后看，左右镜像）
    camera.position.set(c.x, c.y + r * 0.15, c.z - r * 2.0);
    controls.target.set(c.x, c.y, c.z);
    controls.update();

    function onResize() {{
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    }}
    window.addEventListener("resize", onResize);

    function animate() {{
      requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    }}
    animate();
  </script>
</body>
</html>
"""


def get_mask(processor: Sam3Processor, rgb_image: Image.Image, prompt: str):
    """从 RGB 图和 prompt 获取分割 mask。"""
    image = rgb_image.convert("RGB")
    state = processor.set_image(image)
    output = processor.set_text_prompt(state=state, prompt=prompt)

    masks = output["masks"]
    scores = output["scores"]

    if masks is None or masks.numel() == 0:
        raise ValueError(f"未找到任何 mask，prompt='{prompt}' 可能与图像不匹配。")

    best_idx = scores.argmax().item()
    mask = masks[best_idx].detach().cpu().numpy()
    mask = np.squeeze(mask)
    if mask.ndim > 2:
        mask = mask[0]
    return mask.astype(bool)


def mask_border_touches(mask: np.ndarray) -> dict[str, bool]:
    """检测分割 mask 是否触及图像四边。"""
    h, w = mask.shape[:2]
    return {
        "top": bool(np.any(mask[0, :])),
        "bottom": bool(np.any(mask[h - 1, :])),
        "left": bool(np.any(mask[:, 0])),
        "right": bool(np.any(mask[:, w - 1])),
    }


def build_mask_core(mask: np.ndarray, erode_px: int = MASK_ERODE_PX) -> tuple[np.ndarray, np.ndarray, int]:
    """mask 内缩得到可靠核心区域，边缘带剔除不参与反投影。

    返回 (mask_core, mask_edge, applied_erode_px)。
    若内缩后像素过少，会逐步减小 erode 直至满足 MASK_CORE_MIN_PIXELS。
    """
    mask_bool = mask.astype(bool)
    px = max(0, int(erode_px))
    mask_core = mask_bool

    while True:
        if px == 0:
            mask_core = mask_bool
            break
        k = 2 * px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask_core = cv2.erode(mask_bool.astype(np.uint8), kernel, iterations=1).astype(bool)
        if int(mask_core.sum()) >= MASK_CORE_MIN_PIXELS:
            break
        if px == 0:
            break
        px -= 1
        logger.warning(
            f"mask 内缩 {erode_px}px 后仅剩 {int(mask_core.sum())} 像素，"
            f"降为 {px}px"
        )

    mask_edge = mask_bool & ~mask_core
    return mask_core, mask_edge, px


def depth_to_point_cloud(depth, rgb, mask):
    """将深度图 + mask 反投影为 3D 点云。"""
    h, w = depth.shape
    u = np.arange(w, dtype=np.float32)
    v = np.arange(h, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    z = depth.astype(np.float32) / DEPTH_SCALE
    x = (uu - CX) * z / FX
    y = (vv - CY) * z / FY

    valid = (z > 0) & (z < DEPTH_MAX) & mask
    points = np.stack([x[valid], y[valid], z[valid]], axis=-1)
    colors = rgb[valid] / 255.0
    return points, colors


def apply_sor(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """SOR 统计离群点移除。"""
    n_before = len(pcd.points)
    pcd_sor, _ = pcd.remove_statistical_outlier(
        nb_neighbors=NB_NEIGHBORS,
        std_ratio=STD_RATIO,
    )
    logger.info(f"SOR: {n_before} → {len(pcd_sor.points)} 点（移除 {n_before - len(pcd_sor.points)} 个）")
    return pcd_sor


def _cluster_depth_median(pts: np.ndarray) -> float:
    return float(np.median(pts[:, 2]))


def _select_merged_box_clusters(
    labels: np.ndarray,
    points: np.ndarray,
    *,
    merge_dist: float | None = None,
    depth_diff: float | None = None,
) -> set[int]:
    """DBSCAN 后选取并合并属于同一箱子的多个簇（如两个可见面）。"""
    merge_dist = CLUSTER_MERGE_DIST if merge_dist is None else merge_dist
    depth_diff = MAX_DEPTH_MEDIAN_DIFF if depth_diff is None else depth_diff

    valid = labels >= 0
    if not np.any(valid):
        return set()

    unique, counts = np.unique(labels[valid], return_counts=True)
    order = np.argsort(-counts)
    unique = unique[order]
    counts = counts[order]

    largest_count = int(counts[0])
    min_cluster_pts = max(DBSCAN_MIN_POINTS, int(largest_count * MIN_CLUSTER_RATIO))

    cluster_centers: dict[int, np.ndarray] = {}
    cluster_z_medians: dict[int, float] = {}
    for lbl, cnt in zip(unique, counts):
        lbl = int(lbl)
        pts = points[labels == lbl]
        cluster_centers[lbl] = pts.mean(axis=0)
        cluster_z_medians[lbl] = _cluster_depth_median(pts)

    def compatible(lbl_a: int, lbl_b: int) -> bool:
        if float(np.linalg.norm(cluster_centers[lbl_a] - cluster_centers[lbl_b])) > merge_dist:
            return False
        if abs(cluster_z_medians[lbl_a] - cluster_z_medians[lbl_b]) > depth_diff:
            return False
        return True

    kept: set[int] = {int(unique[0])}
    rejected: list[tuple[int, int, str]] = []

    for lbl, cnt in zip(unique[1:], counts[1:]):
        lbl = int(lbl)
        cnt = int(cnt)
        if cnt < min_cluster_pts:
            rejected.append((lbl, cnt, "过小"))
            continue

        if any(compatible(lbl, kept_lbl) for kept_lbl in kept):
            kept.add(lbl)
        else:
            rejected.append((lbl, cnt, "远离主体"))

    logger.info(
        f"多簇合并: 保留 {len(kept)} 簇 {sorted(kept)}，"
        f"拒绝 {rejected}（merge_dist={merge_dist:.2f}m）"
    )
    return kept


def _recover_sor_indices_from_clusters(
    pcd_sor: o3d.geometry.PointCloud,
    pcd_down: o3d.geometry.PointCloud,
    labels: np.ndarray,
    kept_labels: set[int],
    recover_radius: float,
) -> list[int]:
    """将降采样簇标签映射回 SOR 全分辨率点云。"""
    kept_mask = np.isin(labels, list(kept_labels))
    kept_idx = np.where(kept_mask)[0].tolist()
    if not kept_idx:
        return []

    pcd_kept_down = pcd_down.select_by_index(kept_idx)
    kdtree = o3d.geometry.KDTreeFlann(pcd_kept_down)
    all_pts = np.asarray(pcd_sor.points)
    main_indices: list[int] = []
    for i, pt in enumerate(all_pts):
        k, _, _ = kdtree.search_radius_vector_3d(pt, recover_radius)
        if k > 0:
            main_indices.append(i)
    return main_indices


def apply_dbscan(pcd_sor: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """体素降采样 + DBSCAN，多簇保留合并后映射回 SOR 点云。"""
    n_sor = len(pcd_sor.points)
    if n_sor < DBSCAN_MIN_POINTS:
        logger.info(f"DBSCAN: 点数 {n_sor} 不足，跳过")
        return pcd_sor

    voxel_size = DBSCAN_EPS * 0.8
    pcd_down = pcd_sor.voxel_down_sample(voxel_size=voxel_size)
    logger.info(f"降采样: {n_sor} → {len(pcd_down.points)} 点（体素={voxel_size:.3f}m）")

    labels = np.array(pcd_down.cluster_dbscan(
        eps=DBSCAN_EPS,
        min_points=DBSCAN_MIN_POINTS,
        print_progress=False,
    ))
    if labels.max() < 0:
        logger.warning("DBSCAN: 未找到有效簇，跳过")
        return pcd_sor

    pts_down = np.asarray(pcd_down.points)
    kept_labels = _select_merged_box_clusters(labels, pts_down)

    recover_radius = DBSCAN_EPS + voxel_size
    main_indices = _recover_sor_indices_from_clusters(
        pcd_sor, pcd_down, labels, kept_labels, recover_radius,
    )

    if len(main_indices) < n_sor * MIN_KEEP_RATIO:
        logger.warning(
            f"DBSCAN 保留率 {len(main_indices)}/{n_sor} "
            f"({100 * len(main_indices) / n_sor:.1f}%) 过低，放宽合并与回收"
        )
        kept_labels = _select_merged_box_clusters(
            labels, pts_down,
            merge_dist=CLUSTER_MERGE_DIST * 1.5,
            depth_diff=MAX_DEPTH_MEDIAN_DIFF * 1.5,
        )
        recover_radius = DBSCAN_EPS + voxel_size * 2.5
        main_indices = _recover_sor_indices_from_clusters(
            pcd_sor, pcd_down, labels, kept_labels, recover_radius,
        )

    if len(main_indices) < n_sor * MIN_KEEP_RATIO:
        logger.warning(
            f"DBSCAN 保留率仍仅 {100 * len(main_indices) / n_sor:.1f}%，回退 SOR 点云"
        )
        return pcd_sor

    if not main_indices:
        logger.warning("DBSCAN: 映射后无有效点，回退 SOR")
        return pcd_sor

    pcd_clean = pcd_sor.select_by_index(main_indices)
    n_clusters = len(np.unique(labels[labels >= 0]))
    logger.info(
        f"DBSCAN: {n_clusters} 个簇，合并保留 {len(kept_labels)} 簇 → "
        f"{len(pcd_clean.points)} 点（移除 {n_sor - len(pcd_clean.points)} 个，"
        f"保留率 {100 * len(pcd_clean.points) / n_sor:.1f}%）"
    )
    return pcd_clean


def remove_outliers(pcd):
    """两阶段去噪：SOR + DBSCAN 多簇合并。"""
    return apply_dbscan(apply_sor(pcd))


def fit_obb_from_ransac_planes(pcd):
    """RANSAC 拟合两个正交平面，构建 OBB。"""
    all_pts = np.asarray(pcd.points)
    n_pts = len(all_pts)
    remaining_idx = np.arange(n_pts)
    remaining_pcd = pcd

    # 平面1
    plane_model1, local_inliers1 = remaining_pcd.segment_plane(
        distance_threshold=DISTANCE_THR,
        ransac_n=3,
        num_iterations=RANSAC_ITER,
    )
    a, b, c, _ = plane_model1
    n1 = np.array([a, b, c], dtype=np.float64)
    n1 /= np.linalg.norm(n1)

    keep_mask = np.ones(len(remaining_idx), dtype=bool)
    keep_mask[np.array(local_inliers1)] = False
    remaining_idx = remaining_idx[keep_mask]
    remaining_pcd = pcd.select_by_index(remaining_idx.tolist())
    logger.info(f"平面1: 法向量=({n1[0]:.3f},{n1[1]:.3f},{n1[2]:.3f})  内点={len(local_inliers1)}")

    # 平面2（与平面1正交）
    n2 = None
    for attempt in range(8):
        if len(remaining_pcd.points) < 10:
            break
        plane_model_cand, local_inliers_cand = remaining_pcd.segment_plane(
            distance_threshold=DISTANCE_THR,
            ransac_n=3,
            num_iterations=RANSAC_ITER,
        )
        a, b, c, _ = plane_model_cand
        n_cand = np.array([a, b, c], dtype=np.float64)
        n_cand /= np.linalg.norm(n_cand)
        dot = abs(float(np.dot(n1, n_cand)))
        logger.info(f"平面2 尝试{attempt+1}: |dot|={dot:.3f}  内点={len(local_inliers_cand)}")

        if dot < ORTH_THR:
            n2 = n_cand
            break

        keep_mask = np.ones(len(remaining_idx), dtype=bool)
        keep_mask[np.array(local_inliers_cand)] = False
        remaining_idx = remaining_idx[keep_mask]
        remaining_pcd = pcd.select_by_index(remaining_idx.tolist())

    if n2 is None:
        raise ValueError("未找到与平面1正交的第二个面，请调整 ORTH_THR 或 DISTANCE_THR。")

    # 正交化 + 第三轴
    n2 = n2 - np.dot(n2, n1) * n1
    n2 /= np.linalg.norm(n2)
    n3 = np.cross(n1, n2)
    n3 /= np.linalg.norm(n3)

    # 投影所有点求 OBB
    R = np.column_stack([n1, n2, n3])
    proj = all_pts @ R
    mn, mx = proj.min(axis=0), proj.max(axis=0)
    extent = mx - mn
    center = R @ ((mn + mx) / 2.0)

    return center, extent, R


def build_initial_point_cloud(
    rgb_image: Image.Image,
    depth_image: Image.Image,
    mask: np.ndarray,
) -> o3d.geometry.PointCloud:
    """mask + 深度反投影，得到初始彩色点云（未经 SOR/DBSCAN）。"""
    rgb_np = np.array(rgb_image.convert("RGB"))
    h_rgb, w_rgb = rgb_np.shape[:2]

    depth_np = np.array(depth_image)
    if depth_np.shape[:2] != (h_rgb, w_rgb):
        depth_pil = Image.fromarray(depth_np)
        depth_pil = depth_pil.resize((w_rgb, h_rgb), Image.NEAREST)
        depth_np = np.array(depth_pil)

    points, colors = depth_to_point_cloud(depth_np, rgb_np, mask)
    if len(points) == 0:
        raise ValueError("mask 区域内无有效深度点，请检查深度图或相机内参。")
    logger.info(f"有效点数: {len(points)}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def prepare_point_cloud_stages(
    rgb_image: Image.Image,
    depth_image: Image.Image,
    mask: np.ndarray,
    *,
    mask_core: np.ndarray | None = None,
    erode_px: int | None = None,
) -> tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
    """返回 (初始点云, SOR后点云, SOR+DBSCAN后点云)。

    初始点云使用 mask 内缩后的核心区域，边缘带已剔除。
    若已传入 mask_core，则不再重复计算内缩。
    """
    if mask_core is None:
        mask_core, mask_edge, erode_px = build_mask_core(mask)
        logger.info(
            f"mask 内缩: erode={erode_px}px，核心 {int(mask_core.sum())} px，"
            f"边缘剔除 {int(mask_edge.sum())} px"
        )
    pcd_raw = build_initial_point_cloud(rgb_image, depth_image, mask_core)
    pcd_sor = apply_sor(pcd_raw)
    pcd_clean = apply_dbscan(pcd_sor)
    return pcd_raw, pcd_sor, pcd_clean


def normalize_point_cloud_stage(raw: str) -> str:
    """将客户端传入的点云阶段规范为 sor 或 sor_dbscan。"""
    key = raw.strip().lower().replace("+", "_").replace(" ", "")
    aliases = {
        "sor": "sor",
        "sor_dbscan": "sor_dbscan",
        "sordbscan": "sor_dbscan",
    }
    if key not in aliases:
        raise ValueError(
            f"point_cloud_stage 无效: {raw!r}，必须为 sor 或 sor_dbscan（SOR+DBSCAN）"
        )
    return aliases[key]


def select_point_cloud_for_obb(
    pcd_sor: o3d.geometry.PointCloud,
    pcd_dbscan: o3d.geometry.PointCloud,
    stage: str,
) -> o3d.geometry.PointCloud:
    if stage == "sor":
        return pcd_sor
    return pcd_dbscan


def encode_pointcloud_stage_html(
    pcd: o3d.geometry.PointCloud,
    *,
    title: str,
    hud_text: str,
    center: np.ndarray | None = None,
    extent: np.ndarray | None = None,
    R: np.ndarray | None = None,
) -> str:
    return encode_text_b64(build_pointcloud_html(
        pcd, center=center, extent=extent, R=R, title=title, hud_text=hud_text,
    ))


def prepare_processed_point_cloud(
    rgb_image: Image.Image,
    depth_image: Image.Image,
    mask: np.ndarray,
) -> o3d.geometry.PointCloud:
    """mask → 点云 → SOR → DBSCAN，返回送给 RANSAC 前的点云。"""
    _, _, pcd_clean = prepare_point_cloud_stages(rgb_image, depth_image, mask)
    return pcd_clean


def compute_3d_bbox(
    rgb_image: Image.Image,
    depth_image: Image.Image,
    prompt: str,
    *,
    mask: np.ndarray | None = None,
    pcd: o3d.geometry.PointCloud | None = None,
):
    """完整流程：mask → 点云 → 去噪 → RANSAC OBB，返回 center, extent, R, mask, pcd。"""
    if mask is None:
        mask = get_mask(_processor, rgb_image, prompt)
    if pcd is None:
        pcd = prepare_processed_point_cloud(rgb_image, depth_image, mask)

    center, extent, R = fit_obb_from_ransac_planes(pcd)
    return center, extent, R, mask, pcd


# ============ API 接口 ============

@app.get("/health")
async def health():
    """健康检查接口。"""
    ready = _processor is not None
    return {"status": "ready" if ready else "loading", "model": "SAM3"}


@app.post("/predict")
async def predict(
    rgb: UploadFile = File(..., description="RGB 图像（jpg/png）"),
    depth: UploadFile = File(..., description="深度图（uint16 png）"),
    prompt: str = Form(default="box", description="分割提示词"),
    point_cloud_stage: str = Form(
        ..., description="点云阶段：sor 或 sor_dbscan（SOR+DBSCAN），用于 OBB/RANSAC",
    ),
):
    """
    计算 3D Bounding Box 及机器人抓取点。

    返回:
        center          : OBB 中心坐标 [x, y, z]（米，相机坐标系）
        extent          : OBB 三轴尺寸 [e0, e1, e2]（米）
        rotation_matrix : 3x3 旋转矩阵（列为 n1, n2, n3 轴方向）
        length          : 沿 n3 轴（第三平面法向量）方向的箱子长度（米）
        grasp_left      : 箱子左侧面中点坐标 [x, y, z]（左手抓取点）
        grasp_right     : 箱子右侧面中点坐标 [x, y, z]（右手抓取点）
        point_cloud_stage: 本次 OBB 使用的点云阶段（sor / sor_dbscan）
        overlay_image   : 内缩前 mask 叠加图（base64 JPEG），同 overlay_before_erode_image
        overlay_before_erode_image: SAM3 原始 mask 叠加图（base64 JPEG）
        overlay_after_erode_image : 内缩后 mask_core 叠加图（base64 JPEG）
        mask_erode_px   : 实际应用的内缩像素数
        overlay_format  : overlay 图像格式，固定为 "jpeg"
        pointcloud_sor_dbscan_html : SOR+DBSCAN 后点云 HTML（成功时若用该阶段则含 OBB）
        pointcloud_initial_html : 初始点云 HTML（mask 反投影，无 OBB）
        pointcloud_sor_html     : SOR 后点云 HTML（成功时若用该阶段则含 OBB）
        pointcloud_format: 点云格式，固定为 "html"
    """
    if _processor is None:
        raise HTTPException(status_code=503, detail="模型尚未加载完成，请稍后重试。")

    try:
        pcd_stage = normalize_point_cloud_stage(point_cloud_stage)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    overlay_before_b64 = None
    overlay_after_b64 = None
    mask_erode_px = None
    pointcloud_initial_html_b64 = None
    pointcloud_sor_html_b64 = None
    pointcloud_sor_dbscan_html_b64 = None
    mask_touches = None
    try:
        rgb_bytes = await rgb.read()
        depth_bytes = await depth.read()

        rgb_image = Image.open(io.BytesIO(rgb_bytes)).convert("RGB")
        depth_image = Image.open(io.BytesIO(depth_bytes))

        logger.info(f"收到请求: prompt='{prompt}', rgb={rgb_image.size}, depth={depth_image.size}")

        mask = get_mask(_processor, rgb_image, prompt)
        mask_touches = mask_border_touches(mask)
        rgb_np = np.array(rgb_image.convert("RGB"))
        mask_core, mask_edge, mask_erode_px = build_mask_core(mask)
        logger.info(
            f"mask 内缩: erode={mask_erode_px}px，核心 {int(mask_core.sum())} px，"
            f"边缘剔除 {int(mask_edge.sum())} px"
        )
        overlay_before_b64 = encode_overlay_jpeg_b64(make_mask_overlay(rgb_np, mask))
        overlay_after_b64 = encode_overlay_jpeg_b64(make_mask_overlay(rgb_np, mask_core))

        pcd_raw, pcd_sor, pcd_clean = prepare_point_cloud_stages(
            rgb_image, depth_image, mask,
            mask_core=mask_core, erode_px=mask_erode_px,
        )
        pcd_for_obb = select_point_cloud_for_obb(pcd_sor, pcd_clean, pcd_stage)
        stage_label = "SOR" if pcd_stage == "sor" else "SOR+DBSCAN"
        logger.info(f"OBB 点云阶段: {stage_label} ({len(pcd_for_obb.points)} 点)")

        pointcloud_initial_html_b64 = encode_pointcloud_stage_html(
            pcd_raw,
            title="SAM3 Point Cloud (Initial)",
            hud_text="左键旋转 · 右键平移 · 滚轮缩放 · 初始点云（mask 内缩后反投影）",
        )
        pointcloud_sor_html_b64 = encode_pointcloud_stage_html(
            pcd_sor,
            title="SAM3 Point Cloud (SOR)",
            hud_text="左键旋转 · 右键平移 · 滚轮缩放 · SOR 后点云",
        )
        pointcloud_sor_dbscan_html_b64 = encode_pointcloud_stage_html(
            pcd_clean,
            title="SAM3 Point Cloud (SOR+DBSCAN)",
            hud_text="左键旋转 · 右键平移 · 滚轮缩放 · SOR+DBSCAN 后点云",
        )

        center, extent, R, _, _ = compute_3d_bbox(
            rgb_image, depth_image, prompt, mask=mask, pcd=pcd_for_obb,
        )
        obb_html = encode_pointcloud_stage_html(
            pcd_for_obb,
            title=f"SAM3 Point Cloud ({stage_label})",
            hud_text=f"左键旋转 · 右键平移 · 滚轮缩放 · {stage_label} 后点云 + OBB",
            center=center, extent=extent, R=R,
        )
        if pcd_stage == "sor":
            pointcloud_sor_html_b64 = obb_html
        else:
            pointcloud_sor_dbscan_html_b64 = obb_html

        # R 的列向量分别是 n1, n2, n3（n3 = n1 × n2，即第三平面的法向量方向）
        # extent[2] 是点云沿 n3 方向的跨度，即箱子在该方向上的长度
        n3 = R[:, 2]                      # 第三轴方向（单位向量）
        length = float(extent[2])          # 沿 n3 方向的长度

        # 箱子两侧面中点 = 中心 ± (length/2) * n3
        # n3 方向由 RANSAC 结果决定，正负不固定，需根据相机坐标系 X 轴判断左右：
        # 相机坐标系中 X 越小 = 越靠左，X 越大 = 越靠右
        grasp_plus  = center + (length / 2.0) * n3
        grasp_minus = center - (length / 2.0) * n3

        if grasp_plus[0] < grasp_minus[0]:
            # grasp_plus 的 X 更小，在相机视角下更靠左
            grasp_left  = grasp_plus
            grasp_right = grasp_minus
        else:
            grasp_left  = grasp_minus
            grasp_right = grasp_plus

        result = {
            "center":           center.tolist(),
            "extent":           extent.tolist(),
            "rotation_matrix":  R.tolist(),
            "length":           length,
            "grasp_left":       grasp_left.tolist(),
            "grasp_right":      grasp_right.tolist(),
            "mask_touches":     mask_touches,
            "mask_erode_px":    mask_erode_px,
            "point_cloud_stage": pcd_stage,
            "overlay_image":    overlay_before_b64,
            "overlay_before_erode_image": overlay_before_b64,
            "overlay_after_erode_image": overlay_after_b64,
            "overlay_format":   "jpeg",
            "pointcloud_sor_dbscan_html": pointcloud_sor_dbscan_html_b64,
            "pointcloud_initial_html":   pointcloud_initial_html_b64,
            "pointcloud_sor_html":       pointcloud_sor_html_b64,
            "pointcloud_format":         "html",
        }

        logger.info(
            f"point_cloud_stage={pcd_stage}  "
            f"center={[f'{v:.4f}' for v in center]}  "
            f"extent={[f'{v:.4f}' for v in extent]}  "
            f"length={length:.4f}  "
            f"grasp_left={[f'{v:.4f}' for v in grasp_left]}  "
            f"grasp_right={[f'{v:.4f}' for v in grasp_right]}"
        )
        return JSONResponse(content=result)

    except ValueError as e:
        content: dict = {"detail": str(e)}
        if mask_touches is not None:
            content["mask_touches"] = mask_touches
        if overlay_before_b64:
            content["overlay_image"] = overlay_before_b64
            content["overlay_before_erode_image"] = overlay_before_b64
            content["overlay_format"] = "jpeg"
        if overlay_after_b64:
            content["overlay_after_erode_image"] = overlay_after_b64
        if mask_erode_px is not None:
            content["mask_erode_px"] = mask_erode_px
        if pointcloud_initial_html_b64:
            content["pointcloud_initial_html"] = pointcloud_initial_html_b64
        if pointcloud_sor_html_b64:
            content["pointcloud_sor_html"] = pointcloud_sor_html_b64
        if pointcloud_sor_dbscan_html_b64:
            content["pointcloud_sor_dbscan_html"] = pointcloud_sor_dbscan_html_b64
            content["pointcloud_format"] = "html"
        return JSONResponse(status_code=422, content=content)
    except Exception as e:
        logger.exception("处理请求时发生错误")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
