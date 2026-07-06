import os

import numpy as np
from PIL import Image

import torch
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


IMAGE_PATH = "/home/liuzihou/sam3/assets/images/13.jpg"
PROMPT = "box"
MODEL_DIR = "/home/liuzihou/sam3/model"


def build_model():
    checkpoint_path = MODEL_DIR
    if os.path.isdir(MODEL_DIR):
        checkpoint_path = os.path.join(MODEL_DIR, "sam3.pt")

    # 使用 GPU（需要你已经安装支持 RTX 5090 的 PyTorch）
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = build_sam3_image_model(
        checkpoint_path=checkpoint_path,
        load_from_HF=False,
        device="cuda",
    )
    processor = Sam3Processor(model, confidence_threshold=0.5)
    return processor


def run_single_image(processor, image_path: str, prompt: str):
    image = Image.open(image_path).convert("RGB")
    inference_state = processor.set_image(image)
    output = processor.set_text_prompt(state=inference_state, prompt=prompt)

    masks = output["masks"]
    scores = output["scores"]

    # 没有任何预测结果
    if masks is None or masks.numel() == 0:
        print("No masks returned. Maybe the prompt does not match anything in the image.")
        print(f"Prompt: {prompt}")
        return

    # 选择得分最高的一个 mask
    best_idx = scores.argmax().item()
    if masks.ndim == 4:
        mask = masks[best_idx].detach().cpu().numpy() > 0.5
    elif masks.ndim == 3:
        mask = masks[best_idx].detach().cpu().numpy() > 0.5
    else:
        mask = masks.detach().cpu().numpy() > 0.5

    # 完整 squeeze，得到 (H, W)
    mask = np.squeeze(mask)
    if mask.ndim > 2:
        mask = mask[0]

    img_np = np.array(image)
    color = np.array([255, 0, 0], dtype=np.uint8)
    overlay = img_np.copy()
    if mask.shape != img_np.shape[:2]:
        print(f"Mask shape {mask.shape} does not match image shape {img_np.shape[:2]}, skip saving.")
        return
    overlay[mask] = (0.5 * overlay[mask] + 0.5 * color).astype(np.uint8)

    out_path = os.path.splitext(image_path)[0] + "_case_mask.png"
    Image.fromarray(overlay).save(out_path)
    print(f"Saved segmented image to: {out_path}")


def main():
    processor = build_model()
    run_single_image(processor, IMAGE_PATH, PROMPT)


if __name__ == "__main__":
    main()
