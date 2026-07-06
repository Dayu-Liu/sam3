# SAM3 Deployment Notes on 5080

Last updated: 2026-06-18

This file records the current SAM3 deployment on `5080-lan` for the
`box_demo_2` perception path.

## Host and Paths

- Host: `5080-lan`
- Linux user: `wjzh`
- SAM3 project: `/home/wjzh/sam3`
- Python virtual environment: `/home/wjzh/sam3/.venv`
- SAM3 model directory: `/home/wjzh/sam3/model`
- Box demo integration path: `/home/wjzh/agile_boxdeploy/box_demo_2`
- SAM3 server tmux session: `sam3-server`
- SAM3 server log: `/tmp/sam3_server.log`
- SAM3 service URL on the same host: `http://127.0.0.1:5300`

## What Was Deployed

The original project files were copied from:

```bash
zhangjin@183.238.119.251:/home/zhangjin/sam3
```

to:

```bash
/home/wjzh/sam3
```

The first pass copied code, docs, configs, scripts, notebooks, tokenizer/config
files, and example assets while excluding large model weights.

The two large weight files were then transferred separately from the same source
and verified on `5080-lan`:

```text
/home/wjzh/sam3/model/sam3.pt
/home/wjzh/sam3/model/model.safetensors
```

Expected SHA256:

```text
9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e  /home/wjzh/sam3/model/sam3.pt
6d06f0a5f84e435071fe6603e61d0b4cc7b40e0d39d487cfd4d67d8cc11cc14a  /home/wjzh/sam3/model/model.safetensors
```

Verify them with:

```bash
sha256sum /home/wjzh/sam3/model/sam3.pt \
          /home/wjzh/sam3/model/model.safetensors
```

## Python Environment

README.md recommends Python 3.12 and PyTorch 2.7.0 with CUDA 12.6. On this
machine, the GPU is an RTX 5080 Laptop GPU with compute capability `sm_120`.
The `cu126` PyTorch wheel fails with:

```text
CUDA error: no kernel image is available for execution on the device
```

The deployed environment therefore uses the CUDA 12.8 wheel:

```text
Python 3.12.13
torch 2.7.0+cu128
torchvision 0.22.0+cu128
torchaudio 2.7.0+cu128
numpy 1.26.4
sam3 0.1.0
```

The environment was created with `uv`:

```bash
cd /home/wjzh/sam3
/snap/bin/uv venv --python 3.12 .venv
/snap/bin/uv pip install --python .venv/bin/python \
  torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128
/snap/bin/uv pip install --python .venv/bin/python \
  -e . "numpy<2" fastapi "uvicorn[standard]" python-multipart \
  open3d requests opencv-python einops decord pycocotools
```

Check CUDA:

```bash
cd /home/wjzh/sam3
.venv/bin/python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0))
print("capability", torch.cuda.get_device_capability(0))
x = torch.ones((1024, 1024), device="cuda")
print("cuda_tensor_sum", float(x.sum().item()))
PY
```

## Starting the SAM3 Server

The SAM3 HTTP server is:

```text
/home/wjzh/sam3/scripts/sam_server.py
```

It loads:

```text
/home/wjzh/sam3/model/sam3.pt
```

and listens on port `5300`.

Start or restart the service:

```bash
ssh 5080-lan
tmux kill-session -t sam3-server 2>/dev/null || true
tmux new-session -d -s sam3-server \
  "cd /home/wjzh/sam3 && \
   export SAM3_MODEL_DIR=/home/wjzh/sam3/model && \
   export PYTHONUNBUFFERED=1 && \
   .venv/bin/python scripts/sam_server.py 2>&1 | tee /tmp/sam3_server.log"
```

Attach to the server:

```bash
tmux attach -t sam3-server
```

Detach from tmux with `Ctrl+B`, then `D`.

Stop the server:

```bash
tmux kill-session -t sam3-server
```

Watch logs:

```bash
tail -f /tmp/sam3_server.log
```

Check health:

```bash
curl -sS http://127.0.0.1:5300/health
```

Expected:

```json
{"status":"ready","model":"SAM3"}
```

Check GPU usage:

```bash
nvidia-smi
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```

The running SAM3 server usually holds about 4.5 GB GPU memory.

## Smoke Test

Run a local RGBD request against the running server:

```bash
cd /home/wjzh/sam3
.venv/bin/python scripts/sam_client.py \
  --rgb assets/images/13.jpg \
  --depth assets/images/13_d.png \
  --prompt box \
  --host 127.0.0.1 \
  --port 5300
```

This should return a 3D bounding box plus `grasp_left` and `grasp_right`.

The `box_demo_2` local client was also tested successfully:

```bash
cd /home/wjzh/agile_boxdeploy/box_demo_2
. /home/wjzh/sam3/.venv/bin/activate
python sam3_client.py \
  --rgb /home/wjzh/sam3/assets/images/13.jpg \
  --depth /home/wjzh/sam3/assets/images/13_d.png \
  --prompt box \
  --host 127.0.0.1 \
  --port 5300
```

## Box Demo Integration

`box_demo_2/box_demo_main.py` calls SAM3 over HTTP. It does not import SAM3
directly.

Important arguments in `box_demo_main.py`:

```text
--prompt     default: box
--host       SAM3 server host
--port       SAM3 server port
```

`/home/wjzh/agile_boxdeploy/box_demo_2/start_agile_box.sh` was updated to make
the SAM3 endpoint explicit:

```bash
SAM3_HOST="${SAM3_HOST:-127.0.0.1}"
SAM3_PORT="${SAM3_PORT:-5300}"
```

It now accepts:

```text
--sam3-host HOST
--sam3-port PORT
```

and passes them into `box_demo_main.py`:

```bash
--host "$SAM3_HOST" --port "$SAM3_PORT"
```

This avoids accidentally using the old default SAM3 host
`192.168.112.198:5300`.

Run the AGILE box demo with the local SAM3 server:

```bash
cd /home/wjzh/agile_boxdeploy/box_demo_2
./start_agile_box.sh \
  --vlm-endpoint "$VLM_ENDPOINT" \
  --sam3-host 127.0.0.1 \
  --sam3-port 5300 \
  --iface enp130s0
```

If `VLM_ENDPOINT` is already exported, this is enough:

```bash
cd /home/wjzh/agile_boxdeploy/box_demo_2
./start_agile_box.sh --vlm-endpoint "$VLM_ENDPOINT" --iface enp130s0
```

The 5080 network interface observed during deployment was:

```text
enp130s0  10.16.62.158/17
```

## Operational Notes

- Keep the SAM3 server running before starting `box_demo_main.py`.
- Use `127.0.0.1:5300` when `box_demo_main.py` runs on the same 5080 host.
- Use `10.16.62.158:5300` only for another machine on the same LAN.
- The SAM3 repo on 5080 is not a Git checkout; it was copied from another host.
- The model is already local, so normal operation does not require HF access.
- `HF_ENDPOINT=https://hf-mirror.com` was tested, but `facebook/sam3` is gated
  and requires an approved Hugging Face token if downloading from the Hub.
