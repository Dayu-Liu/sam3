"""
SAM3 3D BBox 客户端示例

用法:
    python scripts/sam_client.py --rgb <rgb_path> --depth <depth_path> [--prompt <prompt>] [--host <host>] [--port <port>]

示例:
    python scripts/sam_client.py \\
        --rgb assets/images/13.jpg \\
        --depth assets/images/13_d.png \\
        --prompt box
"""

import argparse
import json
import sys

import requests


def predict(rgb_path: str, depth_path: str, prompt: str, host: str, port: int,
            point_cloud_stage: str):
    url = f"http://{host}:{port}/predict"

    with open(rgb_path, "rb") as f_rgb, open(depth_path, "rb") as f_depth:
        files = {
            "rgb":   (rgb_path,   f_rgb,   "image/jpeg"),
            "depth": (depth_path, f_depth, "image/png"),
        }
        data = {
            "prompt": prompt,
            "point_cloud_stage": point_cloud_stage,
        }

        print(f"发送请求到 {url}")
        print(f"  RGB:   {rgb_path}")
        print(f"  Depth: {depth_path}")
        print(f"  Prompt: {prompt}")
        print(f"  point_cloud_stage: {point_cloud_stage}")

        resp = requests.post(url, files=files, data=data, timeout=120)

    if resp.status_code != 200:
        print(f"请求失败 [{resp.status_code}]: {resp.text}")
        sys.exit(1)

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

    return result


def main():
    parser = argparse.ArgumentParser(description="SAM3 3D BBox 客户端")
    parser.add_argument("--rgb",    required=True, help="RGB 图像路径")
    parser.add_argument("--depth",  required=True, help="深度图路径")
    parser.add_argument("--prompt", default="box", help="分割提示词，默认 box")
    parser.add_argument("--point-cloud-stage", required=True,
                        choices=("sor", "sor_dbscan"),
                        help="点云阶段：sor 或 sor_dbscan（SOR+DBSCAN）")
    parser.add_argument("--host",   default="127.0.0.1", help="服务端 IP，默认 127.0.0.1")
    parser.add_argument("--port",   default=5300, type=int, help="服务端端口，默认 8000")
    args = parser.parse_args()

    predict(args.rgb, args.depth, args.prompt, args.host, args.port,
            args.point_cloud_stage)


if __name__ == "__main__":
    main()
