# GR00T-WBC + box_demo — 真机测试指令

box_demo 现已可用 GR00T-WBC 当运控底座(`--locomotion groot`)。流程:**手动走到箱子前 → box_demo 感知微调 → 抱箱子**。

> 三个角色: `groot_mover.py`(写命令文件) → `groot_wbc_boxdemo_adapter.py`(跑 GR00T policy, 发 rt/lowcmd_rl) → `merge_lowcmd_arm_sdk.py`(合 arm_sdk → rt/lowcmd)。base 真机已验证过。
> **IPC 单写者原则**: keyboard 和 box_demo 都写 `/tmp/robojudo_ext_cmd.json`,**别同时主动写**(谁的 timestamp 新用谁)。手动驱动时不要让 box_demo 同时发指令。
>
> **VLM 默认开 + 代理已修(2026-06-24)**: 完整 pipeline 需要 VLM(千问 Qwen)做箱子粗检测/框选,SAM3 做精确抓取点。VLM 现**默认开启**,key 从 `~/.bashrc` 的 `QWEN_API_KEY` 读,**不用再传 `--vlm-endpoint`**。千问国内直连不走代理:`vlm_guide.py` 用 `trust_env=False` 忽略系统 socks 代理(`~/.bashrc` 的 `ALL_PROXY=socks5://...` 之前会让 httpx 在构造时崩),SAM3/相机走 localhost 也加了 `no_proxy` 绕过。仅 SAM3(不要 VLM)时加 `--no-vlm`。改动文件均有 `*.orig` 备份。

> **cv2 not found / 跑错环境(2026-06-24 已修)**: 若启动 shell 里激活了某个 python venv(如 `g1_deploy`,提示符 `(g1_deploy)`),tmux pane 会继承它、`conda activate hdmi` 盖不掉 → box_demo 用错 python 报 `No module named 'cv2'`。`start_groot_wbc_box.sh` 已改成 `unset VIRTUAL_ENV` + 用**绝对路径 hdmi python**(`~/miniconda3/envs/hdmi/bin/python`)启动 box_demo/keyboard,彻底免疫。**手动跑 python 时同理**:先 `deactivate`(若有 venv),或直接用绝对路径 `~/miniconda3/envs/hdmi/bin/python`,别只 `conda activate hdmi`。

把下面所有命令里的 `--iface` 换成你机器人实际的 DDS 网口(跟你上次 GR00T 真机测试用的一致,通常 `enp130s0` 或 `enP8p1s0`)。

---

## Stage 0 — dry-run 自检(不发电机指令)

```bash
cd ~/agile_boxdeploy/box_demo_2
bash start_groot_wbc_box.sh \
  --sam3-host 127.0.0.1 --sam3-port 5300 \
  --iface enp130s0 --groot-repo ~/GR00T-WholeBodyControl \
  --wbc-venv ~/GR00T-WholeBodyControl/.venv_wbc --ros-distro humble \
  --box-conda-env hdmi --dry-run
```
检查: adapter import OK、ROS humble 起得来、`rt/lowstate` 能收到、pane4 banner 显示 `VLM: https://dashscope...`(=VLM 开)。dry-run **不发** rt/lowcmd。OK 后 Ctrl-C。

## Stage 1 — 手动 bring-up + 验证 groot_mover 真机距离(吊带/有人保护)

先单独起底座(不起 box_demo),确认能站、能走:
```bash
bash start_groot_wbc_manual.sh \
  --iface enp130s0 --groot-repo ~/GR00T-WholeBodyControl \
  --wbc-venv ~/GR00T-WholeBodyControl/.venv_wbc --ros-distro humble
# pane3 键盘: w/s 前后, a/d 左右, q/e 转, z/x 高度, space 停, o 阻尼急停
```
**关键一步(替代我们跳过的 sim 精度表)** — 验证 groot_mover 在真机走对距离。新开一个终端(adapter 保持运行),量一下:
```bash
conda activate hdmi && cd ~/agile_boxdeploy/box_demo_2
python -c "from groot_mover import GrootMover as M; m=M(); m.initialize(); m.move_forward_cm(20); m.rotate_deg(15)"
# 用卷尺量实际前进 ≈? cm、转角 ≈? 度。误差大就调 walk-scale 或 groot_mover.py 里的 FWD_CRUISE。
```
> 小目标(<~5cm / <~3.4°)会被速度 floor 顶成约一步并打印 `[floored->overshoot]`——这是预期(步态最小增量),靠 box_demo 的"再感知"收敛。

用键盘把机器人**手动开到箱子正前方**(箱子在正前方,约 0.4–0.6m)。然后 Ctrl-C 关掉这个 manual session(下一步用完整 launcher)。

## Stage 2 — 完整 box_demo(带确认,一步步)

```bash
bash start_groot_wbc_box.sh \
  --sam3-host 127.0.0.1 --sam3-port 5300 \
  --iface enp130s0 --groot-repo ~/GR00T-WholeBodyControl \
  --wbc-venv ~/GR00T-WholeBodyControl/.venv_wbc --ros-distro humble \
  --box-conda-env hdmi --confirm --no-keyboard
```
- VLM 默认开(千问),无需传 endpoint;若只想用 SAM3 加 `--no-vlm`。
- `--confirm`: 抓取**前**会停下等你按 Enter("按 Enter 开始执行")——给你检查 IK/姿态的机会。
- `--no-keyboard`: 不起键盘 pane,避免和 box_demo 抢 IPC(你已经手动到位了)。
- pane4 (box_demo) 流程: 采图 → VLM 看框 → SAM3 测两面 → **自动微调位置**(strafe / 前后,每步打印 "前移 3cm/侧移 …cm",都是小步) → 快/完整 IK → **停下等 Enter** → 双臂抱箱(`controller.run()`)→ 结束切 RL_LOWER 让腿保持平衡。

**第一次真机抓取建议**: 手放急停;微调的小步是自动的但都打印,看着不对就 `o` 键阻尼 / 硬件遥控 select 急停。确认无误再按 Enter 让它抱。

---

## 急停 / 回退

- 软急停: IPC 写 DAMP(键盘 `o`),或 `python -c "from groot_mover import GrootMover as M; M().damp()"`。
- 硬件急停: 无线遥控 `select`(adapter 解码 wireless_remote → 阻尼)。
- 退瘫软: `Ctrl+C`(adapter 收到 LIMP → kp=kd=0)。
- **回退到 AGILE 底座**: 整条链换 `start_agile_box.sh`(box_demo 不传 --locomotion 即默认 agile)。AGILE 仍是热备。

## 要观察 / 可能要调的

- `walk-scale`(launcher `--walk-scale`,默认 1.0): AGILE 当年用它补欠追踪。GR00T 跟踪不同,真机看实际走的距离再调(走不够→>1,走过头→<1)。
- groot_mover 常量(`groot_mover.py` 顶部): `FWD_CRUISE 0.12 / YAW_CRUISE 0.15 / V_FLOOR 0.08 / MIN_DURATION 0.6`。太快/过冲就降 cruise;最小增量太大就降 floor/min_duration(但别低于 Walk 阈值 0.05)。
- 别动 `merge_lowcmd_arm_sdk.py` 和 GR00T policy。原始文件备份在 `*.orig`。

---

## 后续提升 / TODO（基础抱箱跑通后,按优先级）

1. **base 转角接入 approach 阶段（已决策,未接）** — 现在位置对齐只靠 strafe(横移)+前后,箱子若有偏角(不在正前方/箱体本身有 yaw)靠不上去。决策=**接近阶段**用 base `rotate(度)` 对准,**抓取过程中仍只用腰(`WaistRotator`)**,避免 base 转角和手臂 IK 抢坐标系(抓取中 base 转会让手目标漂移)。改点:`box_demo_main.py` grasp loop(491-501 的 `mover.rotate(found_angle)` 本就注释 + 依赖被禁的"转腰搜箱"),需接 `groot_mover.rotate_deg()` + 一个"箱子偏角→是否转 base"的判据,且**只在 handoff 之前**。
2. **GR00T adapter 安全审计** — adapter 的急停/守护覆盖度 vs AGILE pipeline 未审计:tilt>50°、关节超速>30 rad/s、非有限 obs/action、无线遥控 select 急停、stale 时**持续发阻尼**(merger 自己 stale 会静默不阻尼,阻尼责任在底座)。**脱吊带/无人保护前必须补齐**。
3. **开环→闭环精度** — 当前纯开环定速(距离≈速度×时长),无里程计;`floating_base_pose` 在 obs 里但 policy 没用 xyz。要真闭环需加 base-pose 积分器或外部里程计。先靠"感知→移动→再感知"收敛,精度不够再做。
4. **goal-vs-actual 精度表** — `measure_precision.py` + `start_groot_wbc_precision_test.sh` 已就绪但还没在带显示的 mujoco / 真机上跑出表。跑一版量化每步误差,校准 `walk-scale` 和 `groot_mover` cruise。
5. **walk_scale 真机标定** — 开环距离补偿系数,真机跟踪和 sim 不同,按实测走的距离调(走不够→>1,走过头→<1)。
6. **autonomous face-box（自动找箱子）** — 现在假设箱子在正前方;可加"转一圈找箱 + 自动对准",免去手动开到箱前(依赖 #1 的 base 转角)。
7. **单写者协调** — keyboard 与 box_demo 抢 IPC 是隐患;可加文件锁或显式模式切换,杜绝误同时写。
