# GR00T-WBC + box_demo — 真机测试(方案1:box_demo 跑机器人)

box_demo 跑在 **机器人(unitree-002)** 上(头部 RealSense USB 直连),运控命令经 **HTTP 桥** 转给 **5080** 上的 GR00T 底座。SAM3 在 5080。这是原始设计的拓扑(相机在机器人上),也是已决策的方案1。

> 旧版 `HARDWARE_TEST_GROOT.md` 是 box_demo 跑在 5080 的拓扑(`--locomotion groot`),已不适用本流程。

---

## 拓扑

```
机器人 unitree-002 (aarch64, robojudo env)          5080 (192.168.123.222)
┌──────────────────────────────┐                  ┌──────────────────────────────────┐
│ box_demo_main.py             │  POST 图片(:5300) │ SAM3 (sam_server)                │
│  相机 D435I USB 直连(本机读) │ ───────────────> │                                  │
│  --locomotion remote         │  HTTP /cmd…(:5001)│ agile_http_ipc_server            │
│  RemoteMover ────────────────┼────────────────> │   写 units:agile IPC             │
│                              │                  │   ▼ /tmp/robojudo_ext_cmd.json   │
│  rt/arm_sdk ── DDS ──────────┼──────┐           │ GR00T adapter ─> rt/lowcmd_rl    │
└──────────────────────────────┘      │           │ merger ─> rt/lowcmd ─> 机器人    │
        enP8p1s0 192.168.123.164  └─DDS┴──────────>│ enp130s0 192.168.123.222         │
└─────────────────────────────────────────────────┴──────────────────────────────────┘
```

## 已部署 / 关键事实(2026-06-24)

- **机器人** `unitree-002`:wifi `10.33.2.42`,到 5080 的 DDS 网口 **`enP8p1s0` = 192.168.123.164**。box_demo_2 在 **`/home/unitree/zihou/box_demo_2`**(旧 4 月版已备份 `box_demo_2.apr16.bak`,当前版已 rsync 覆盖)。
- **运行环境 = `robojudo`**(有 `unitree_sdk2py` + 臂 IK + cv2/pyrealsense2/torch;**不是** README 写的 `robojudo_zihou2`,那个缺 `unitree_sdk2py`)。已 `pip install requests httpx openai`。
- **5080** = `192.168.123.222`(enp130s0)。底座/桥用 `hdmi` env。box_demo_2 在 `~/agile_boxdeploy/box_demo_2`。
- **相机**:D435I,serial `254322075278`(= capture_and_predict 默认),USB 直连机器人。停 `videohub_pc4` 释放相机需 `UNITREE_SUDO_PASS=123`(运行时传,**不写进任何文件**)。
- **VLM = 千问 Qwen**(DashScope,OpenAI 兼容)。`openai` 只是客户端库,**不需要 OpenAI key**。机器人暂无 `QWEN_API_KEY` → 先 `--no-vlm`;要开 VLM 就在机器人 `export QWEN_API_KEY=...`(从 5080 `~/.bashrc` 拷)再去掉 `--no-vlm`。
- **已修 OpenSSL 坑**:box_demo_main.py 顶部第一行 `import ssl`(cv2/unitree_sdk2py 会先拉系统旧 libcrypto,导致 openai→ssl 报 `OPENSSL_3.3.0 not found`;先 import ssl 绑到 env 的 OpenSSL 3.5.7)。
- **IPC 单写者**:5080 上 HTTP 桥是唯一写 `/tmp/robojudo_ext_cmd.json` 的进程。**5080 上不要再开键盘或本机 box_demo**。

---

## 运行流程

全程机器人吊架吊着 / 急停在手。

### A. 5080 上起 3 个服务

```bash
# 1) GR00T 底座(merger + adapter)。起来后 Ctrl-C 掉 pane3 键盘,只留 merger + adapter
cd ~/agile_boxdeploy/box_demo_2
bash start_groot_wbc_manual.sh --iface enp130s0 \
  --groot-repo ~/GR00T-WholeBodyControl --wbc-venv ~/GR00T-WholeBodyControl/.venv_wbc --ros-distro humble

# 2) SAM3 服务(单独终端)
cd ~/sam3 && ~/sam3/.venv/bin/uvicorn scripts.sam_server:app --host 0.0.0.0 --port 5300 &
#    等模型加载完: curl -s 127.0.0.1:5300/docs >/dev/null && echo SAM3_OK

# 3) HTTP IPC 桥(收机器人 HTTP 命令 → 写 units:agile IPC 给 adapter)
~/miniconda3/envs/hdmi/bin/python ~/agile_boxdeploy/box_demo_2/agile_http_ipc_server.py \
  --backend legacy-ipc --host 0.0.0.0 --port 5001 &
#    自检: curl -s 127.0.0.1:5001/status
```

### B. 机器人上跑 box_demo

```bash
ssh unitree-002
conda activate robojudo
cd /home/unitree/zihou/box_demo_2

# 先验证整条链路用 --no-vlm(只测 走位 + SAM3 + 抓取)
UNITREE_SUDO_PASS=123 python box_demo_main.py --locomotion remote --no-vlm \
  --ipc-url http://192.168.123.222:5001 --host 192.168.123.222 --port 5300 \
  --iface enP8p1s0
```

流程:相机拍照 → (VLM 关时跳过框选,直接) SAM3 测两个抓取面 → 自动小步对齐(打印每步 cm,经 HTTP 转给 5080 底座)→ 快/完整 IK → **抓取前停下等 Enter** → 双臂抱箱(rt/arm_sdk)→ 切 RL_LOWER 腿保持平衡。

要开完整 pipeline(VLM):机器人 `export QWEN_API_KEY=...` 后去掉 `--no-vlm`。

---

## 上机前自检(机器人侧,已验证过一次)

```bash
# robojudo 环境里 box_demo 全部 import OK(含 openai/httpx,ssl 修复后)
ssh unitree-002 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate robojudo \
  && cd /home/unitree/zihou/box_demo_2 && python -c "import box_demo_main; print(\"IMPORT_OK\")"'
# 相机空闲?
ssh unitree-002 'rs-enumerate-devices -s'      # 应列出 D435I 254322075278
# 5080 桥/SAM3 从机器人可达?
ssh unitree-002 'curl -s http://192.168.123.222:5001/status; curl -s -o /dev/null -w "%{http_code}\n" http://192.168.123.222:5300/docs'
```

## 跑起来盯三处

1. **相机**:`[1/3] 停止 videohub_pc4…` 后能否拿到相机。报"相机仍被占用"→ 先 `rs-enumerate-devices` 看是否空闲;videohub 停不掉就检查 `UNITREE_SUDO_PASS`。
2. **HTTP 命令链**:机器人 `RemoteMover` → `/cmd?vx&duration`(5080:5001)→ IPC → adapter。5080 上看 adapter 日志的 `fsm/cmd` 有没有跟着动;`curl http://192.168.123.222:5001/status` 确认桥活着。
3. **rt/arm_sdk DDS 回路(第一次实测)**:box_demo(机器人)发 rt/arm_sdk → 5080 merger。看 merger 日志收不收得到 arm 帧;收不到多半是 DDS 域/网口不匹配(机器人 box_demo 与 5080 merger 要在同一 192.168.123 DDS 网)。

## 急停 / 回退

- **软急停**:机器人 `Ctrl+C`(box_demo 退 → RemoteMover 不再发);或在任意机器 `curl http://192.168.123.222:5001/damp`(写 DAMP)。
- **硬件急停**:无线遥控 `select`。
- **回退到 box_demo 跑 5080**:用旧 `HARDWARE_TEST_GROOT.md` 的 `--locomotion groot`(相机问题另说)。
- **回退机器人旧代码**:`box_demo_2.apr16.bak`。
- **回退 AGILE 底座**:`start_agile_box.sh`(热备)。

## 后续提升 / TODO

见 `HARDWARE_TEST_GROOT.md` 的「后续提升 / TODO」(base 转角接 approach、adapter 安全审计、闭环精度、精度表、walk_scale 标定等)。本拓扑额外项:rt/arm_sdk 跨机 DDS 回路一旦实测有问题,可考虑把 merger 也搬到机器人侧消除回路。
