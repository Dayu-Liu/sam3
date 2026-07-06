#!/usr/bin/env python3
"""测量 ONNX locomotion policy 的推理频率。

用法:
    python benchmark_policy_freq.py              # 测全部模型
    python benchmark_policy_freq.py --full       # 只测 full-body ONNX
    python benchmark_policy_freq.py --lower      # 只测 lower-body ONNX
    python benchmark_policy_freq.py -n 1000      # 跑 1000 次 (默认 500)
"""

import argparse
import os
import sys
import time
import statistics

import numpy as np

_MODEL_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "locomotion", "RoboJuDo_zihou2", "assets", "models", "g1", "mjlab_loco",
)

MODELS = {
    "full_onnx": {
        "path": os.path.join(_MODEL_DIR, "g1_velocity.onnx"),
        "obs_dim": 96,
        "action_dim": 29,
        "label": "Full-body ONNX (29 DOF)",
    },
    "lower_onnx": {
        "path": os.path.join(_MODEL_DIR, "policy.onnx"),
        "obs_dim": 79,
        "action_dim": 12,
        "label": "Lower-body ONNX (12 DOF)",
    },
}


def bench_onnx(path, obs_dim, action_dim, n_iters):
    import onnxruntime as ort

    # -- session 创建时间 --
    t0 = time.perf_counter()
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    t_load = time.perf_counter() - t0

    inp_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    dummy_obs = np.zeros(obs_dim, dtype=np.float32)

    # -- warmup --
    for _ in range(20):
        sess.run([out_name], {inp_name: dummy_obs[None, :]})

    # -- benchmark --
    latencies = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        sess.run([out_name], {inp_name: dummy_obs[None, :]})
        latencies.append(time.perf_counter() - t0)

    return t_load, latencies


def print_result(label, t_load, latencies):
    lat_ms = [l * 1000 for l in latencies]
    mean_ms = statistics.mean(lat_ms)
    median_ms = statistics.median(lat_ms)
    min_ms = min(lat_ms)
    max_ms = max(lat_ms)
    std_ms = statistics.stdev(lat_ms) if len(lat_ms) > 1 else 0
    hz = 1000.0 / mean_ms

    print(f"  {label}")
    print(f"    加载时间:    {t_load*1000:.1f} ms")
    print(f"    推理延迟:")
    print(f"      mean   {mean_ms:7.2f} ms  ({hz:7.1f} Hz)")
    print(f"      median {median_ms:7.2f} ms")
    print(f"      min    {min_ms:7.2f} ms")
    print(f"      max    {max_ms:7.2f} ms")
    print(f"      std    {std_ms:7.2f} ms")
    target_50hz = 20.0  # ms
    margin = target_50hz - mean_ms
    print(f"    50Hz 余量:   {margin:.1f} ms {'✓ 够用' if margin > 0 else '✗ 不够'}")
    print()


def main():
    parser = argparse.ArgumentParser(description="测量 locomotion policy 推理频率")
    parser.add_argument("-n", type=int, default=500, help="推理迭代次数 (默认 500)")
    parser.add_argument("--full",  action="store_true", help="只测 full-body ONNX")
    parser.add_argument("--lower", action="store_true", help="只测 lower-body ONNX")
    args = parser.parse_args()

    if args.full:
        keys = ["full_onnx"]
    elif args.lower:
        keys = ["lower_onnx"]
    else:
        keys = list(MODELS.keys())

    print("=" * 55)
    print("  Policy Inference Benchmark")
    print(f"  iterations: {args.n}")
    print("=" * 55)
    print()

    for k in keys:
        info = MODELS[k]
        if not os.path.isfile(info["path"]):
            print(f"  [跳过] {info['label']} — 文件不存在: {info['path']}")
            print()
            continue
        t_load, lat = bench_onnx(info["path"], info["obs_dim"], info["action_dim"], args.n)
        print_result(info["label"], t_load, lat)


if __name__ == "__main__":
    main()
