# SAM3 start.py 调试说明

## 前置条件

1. 已安装 **Python 扩展**（含 debugpy）
2. 已激活 `sam3` conda 环境
3. 在 VS Code / Cursor 中打开 `sam3` 项目根目录（`/home/liuzihou/sam3`）

## 选择 Python 解释器

1. `Ctrl+Shift+P` → 输入 `Python: Select Interpreter`
2. 选择 `sam3` conda 环境，例如：`~/miniconda3/envs/sam3/bin/python`

## 调试配置

项目已包含 `.vscode/launch.json`，提供两种配置：

| 配置名 | 说明 |
|--------|------|
| **SAM3: start.py** | 固定运行 `scripts/start.py` |
| **SAM3: start.py (当前文件)** | 运行当前打开的 Python 文件 |

## 开始调试

1. 在 `scripts/start.py` 中需要暂停的位置设置断点（行号左侧点击）
2. `F5` 或 运行 → 启动调试
3. 在左侧「运行和调试」面板选择 **SAM3: start.py**，再点击绿色播放按钮

## 常用断点位置

| 行号 | 位置 | 说明 |
|------|------|------|
| 36 | `output = processor.set_text_prompt(...)` | 查看模型输出 |
| 39 | `masks = output["masks"]` | 查看 masks 形状和内容 |
| 49 | `best_idx = scores.argmax().item()` | 查看选中的 mask 索引 |
| 56 | `mask = np.squeeze(mask)` | 查看 squeeze 后的 mask 形状 |

## 环境变量

- `CUDA_VISIBLE_DEVICES`：默认 `0`，可在 `launch.json` 的 `env` 中修改使用的 GPU

## 修改调试参数

在 `launch.json` 中可调整：

```json
{
    "env": {
        "CUDA_VISIBLE_DEVICES": "0"   // 改用其他 GPU 时改为 "1" 等
    },
    "justMyCode": true   // 设为 true 可跳过库代码，加快调试
}
```

## 常见问题

- **找不到 sam3 模块**：确认 `cwd` 为项目根目录，且已选择 `sam3` 环境
- **CUDA 报错**：确认 PyTorch 支持当前 GPU（如 RTX 5090 需 nightly + CUDA 12.8+）
