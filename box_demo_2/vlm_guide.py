#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VLM 引导模块：将机器人头部相机 RGB 图像发送给千问/Qwen VLM，
判断箱子是否在画面中完整可见（裁切分类）。
"""

import base64
import json
import re

import cv2
import httpx
import numpy as np
from openai import OpenAI

_SYSTEM_PROMPT = """\
You are a robot vision system. The robot has a downward-tilted head camera.
You will receive ONE RGB image from that camera.

Your task:
1. Is there a box/cardboard box visible in the image?
2. If yes, is the box fully visible or clipped by the IMAGE border (not the box's own faces)?
   Judge ONLY from whether the box silhouette touches the image edge:
   - "complete": the whole box is inside the frame; no side touches the image border
   - "left_cut": the box is cut off on the LEFT edge of the image
   - "right_cut": the box is cut off on the RIGHT edge of the image
   - "both_cut": the box is cut off on both left AND right sides of the image
   - "top_cut": the box is cut off on the TOP edge of the image
   - "bottom_cut": the box is cut off on the BOTTOM edge of the image

Important:
- The visible TOP FACE of a box (horizontal lid) is NOT "top_cut". "top_cut" means the
  box extends beyond the top border of the image.
- If only the lower front of the box is missing at the bottom border, use "bottom_cut".
- Do not infer distance or suggest walking; the robot uses fixed moves per label.

Respond in valid JSON only, no markdown, no extra text:
{"box_visible": true/false, "box_in_frame": "complete"/"left_cut"/"right_cut"/"both_cut"/"top_cut"/"bottom_cut",
 "confidence": "high"/"medium"/"low", "description": "brief explanation in English"}"""


class VLMGuide:
    """调用千问 VLM API 获取箱子是否在画面中完整可见。"""

    def __init__(self, endpoint: str, api_key: str, model: str = "qwen-vl-max"):
        self._model = model
        # trust_env=False: ignore system proxy env vars (http_proxy/all_proxy) so a
        # mis-configured proxy can never hijack the cloud VLM call; explicit timeout
        # so a hung request can't stall the grasp loop.
        self._client = OpenAI(
            api_key=api_key, base_url=endpoint,
            http_client=httpx.Client(trust_env=False, timeout=30.0),
        )

    def query(self, image_bgr: np.ndarray, depth_u16: np.ndarray = None) -> dict:
        """
        发送 RGB 图像给 VLM，返回箱子可见性与裁切分类。

        Args:
            image_bgr: BGR 格式的 numpy 图像数组 (H, W, 3)
            depth_u16: 保留参数以兼容旧调用；VLM 裁切判断只用 RGB，不传深度图。

        Returns:
            dict: {"box_visible": bool, "box_in_frame": str,
                   "confidence": str, "description": str}
            出错时返回 {"box_visible": False, "error": str}
        """
        ok, rgb_jpg = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return {"box_visible": False, "error": "RGB JPEG encode failed"}
        rgb_uri = f"data:image/jpeg;base64,{base64.b64encode(rgb_jpg.tobytes()).decode('utf-8')}"

        content = [
            {
                "type": "text",
                "text": (
                    "Is the box fully inside the frame? "
                    "Classify box_in_frame using image borders only. Reply in JSON."
                ),
            },
            {"type": "image_url", "image_url": {"url": rgb_uri, "detail": "auto"}},
        ]

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_completion_tokens=512,
            )
            raw = response.choices[0].message.content or ""
            print(f"  [VLM 原始返回] {raw[:500]}")
        except IndexError:
            return {"box_visible": False, "error": "Empty response from VLM (no choices)"}
        except Exception as e:
            print(f"  [VLM 调用失败] {e}")
            return {"box_visible": False, "error": f"API call failed: {e}"}

        parsed = self._parse_response(raw)
        print(f"  [VLM 解析结果] {parsed}")
        return parsed

    @staticmethod
    def _parse_response(raw: str) -> dict:
        """从 VLM 返回文本中提取 JSON。"""
        raw = raw.strip()
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return {"box_visible": False, "error": f"No JSON found in: {raw[:200]}"}

        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError as e:
            return {"box_visible": False, "error": f"JSON parse error: {e}"}

        return {
            "box_visible": bool(data.get("box_visible", False)),
            "box_in_frame": str(data.get("box_in_frame", "complete")),
            "confidence": str(data.get("confidence", "low")),
            "description": str(data.get("description", "")),
        }


class NullVLMGuide:
    """VLM disabled (--no-vlm): always report the box as fully visible and
    centered so the pipeline skips the VLM framing step and proceeds straight to
    SAM3 detection. Assumes the box is roughly in front of the robot (the
    operator walks it close first); SAM3 then does the real grasp-point
    detection and near/far adjustment.
    """

    def query(self, image_bgr=None, depth_u16=None) -> dict:
        return {
            "box_visible": True,
            "box_in_frame": "complete",
            "confidence": "low",
            "description": "VLM disabled (SAM3-only)",
        }
