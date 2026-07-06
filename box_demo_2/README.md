# Box Demo 2 — 宇树 G1 人形机器人视觉抓取系统

基于视觉感知的双臂自主抓取系统，运行于宇树 G1 人形机器人。系统集成了 VLM（视觉语言模型）目标检测、SAM3 3D 位姿估计、逆运动学求解和 RL 行走控制，实现了从"发现箱子"到"双臂抓取"的全自主闭环。

---

## 目录

- [系统架构](#系统架构)
- [系统总览架构图](#系统总览架构图)
- [工作流程](#工作流程)
- [机器人移动机制详解](#机器人移动机制详解)
- [视觉感知链路详解](#视觉感知链路详解)
- [逆运动学（IK）详解](#逆运动学ik详解)
- [双臂抓取流程详解](#双臂抓取流程详解)
- [RL 行走策略详解](#rl-行走策略详解)
- [核心常量与参数表](#核心常量与参数表)
- [文件说明](#文件说明)
- [环境配置](#环境配置)
- [快速启动](#快速启动)
- [命令行参数](#命令行参数)
- [模块详解](#模块详解)
- [坐标系定义](#坐标系定义)
- [DDS 话题说明](#dds-话题说明)
- [辅助工具](#辅助工具)
- [注意事项](#注意事项)

---

## 系统架构

系统由三个独立进程通过 DDS（数据分发服务）协同工作，由 `merge_lowcmd_arm_sdk.py` 将行走与手臂指令合成为最终电机控制命令：

```
┌──────────────────────────────────────────────────────────────────┐
│                    merge_lowcmd_arm_sdk.py                       │
│              合并 rt/lowcmd_rl + rt/arm_sdk → rt/lowcmd          │
└──────────┬────────────────────────────┬──────────────────────────┘
           │ 下半身(0-11)               │ 上半身(12-29)
           ▼                            ▼
┌─────────────────────┐    ┌──────────────────────────────────────┐
│  run_pipeline.py    │    │        box_demo_main.py              │
│  (RL 行走策略)       │    │  VLM检测 → SAM3预测 → IK求解 → 抓取  │
│  rt/lowcmd_rl       │    │  rt/arm_sdk                          │
└─────────────────────┘    └──────────────────────────────────────┘
```

**核心感知与控制链路：**

```
RealSense D435 → VLM (Qwen) → SAM3 3D BBox → 坐标转换 → IK → 双臂运动
       │              │              │             │        │       │
    RGB+深度图    箱子检测+定位   抓取点预测    相机→躯体系   逆运动学  arm_sdk
```

---

## 系统总览架构图

下面这张图把"数据流向 + 进程边界 + 外部服务 + 控制权切换"四个维度叠在一起，是理解整个系统最快的一张图。

### 全景架构

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                              外部服务（局域网）                              ║
║  ┌──────────────────────┐        ┌──────────────────────────────────────┐  ║
║  │ Qwen VLM API         │ HTTPS  │ SAM3 3D BBox 服务端                   │  ║
║  │ qwen-vl-max          │ ◀────▶ │ 192.168.112.198:5300                 │  ║
║  │ 箱子检测 + 裁切判断   │        │ RGB+Depth → 3D 包围盒 + 抓取点        │  ║
║  └──────────┬───────────┘        └─────────────────┬────────────────────┘  ║
╚═════════════│═══════════════════════════════════════│══════════════════════╝
             │ JPEG80+JET60                       │ POST multipart
             │                                     │ (RGB JPEG + Depth PNG)
             ▼                                     ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║  进程 ③: box_demo_main.py（决策层 / 主循环）                                  ║
║                                                                              ║
║  ┌──────────┐   ┌───────────┐   ┌───────────┐   ┌──────────┐   ┌─────────┐ ║
║  │ VLM 引导 │──▶│ SAM3 预测 │──▶│ 坐标转换  │──▶│ IK 检查  │──▶│ 双臂抓取│ ║
║  │vlm_guide │   │capture_   │   │cam_to_    │   │ArmKine-  │   │DualArm  │ ║
║  │          │   │predict    │   │torso+补偿 │   │matics    │   │Controller║
║  └────┬─────┘   └───────────┘   └───────────┘   └────┬─────┘   └────┬────┘ ║
║       │ 裁切/移动决策                              IK 不通过      rt/arm_sdk║
║       ▼                                              │  ↓ 几何估算    │       ║
║  ┌──────────┐  写 IPC JSON                          ▼  + 衰减       │       ║
║  │RobotMover│──────────────┐               ┌──────────────┐       │       ║
║  │robot_move│              │               │ 触发移动调整  │       │       ║
║  └──────────┘              │               └──────┬───────┘       │       ║
║       │                    │                      │               │       ║
║   rt/arm_sdk               │                      ▼               │       ║
║   (腰旋转)                  │               调用 RobotMover         │       ║
║       │                    │                      │               │       ║
╚═══════│════════════════════│══════════════════════│═══════════════│═══════╝
        │                    │                      │               │
        │ DDS                │ 文件 IPC             │ IPC           │ DDS
        │ rt/arm_sdk         │ /tmp/robojudo_       │               │ rt/arm_sdk
        │                    │ ext_cmd.json         │               │ (权重槽=1)
        │                    ▼                      │               │
        │           ╔══════════════════╗            │               │
        │           ║ 进程 ②:          ║            │               │
        │           ║ run_pipeline.py  ║◀───────────┘               │
        │           ║ (RL 行走策略)    ║ 读 IPC → ONNX 推理           │
        │           ║                  ║                            │
        │           ║ rt/lowcmd_rl     ║                            │
        │           ╚════════╤═════════╝                            │
        │                    │                                      │
        ▼                    ▼                                      ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║  进程 ①: merge_lowcmd_arm_sdk.py（指令合成器 / 仲裁中心 / 500Hz）             ║
║                                                                              ║
║   rt/lowcmd_rl ─┐                       ┌── rt/arm_sdk                       ║
║  (电机 0–34)    ├── 仲裁 ──────────────┤── (电机 12–29)                     ║
║                 │   默认全段 RL          │   motor[29].q>1e-3 则覆盖 12–29     ║
║                 │   + 新鲜度检查         │                                    ║
║                 ▼                        ▼                                    ║
║              ┌──────────────────────────────┐                                 ║
║              │  rt/lowcmd（CRC 校验）        │ ─── 500Hz ───▶ G1 实机电机      ║
║              └──────────────────────────────┘                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

### 三进程职责速查

| 进程 | 文件 | 频率 | 核心职责 | 输入 | 输出 |
|------|------|------|---------|------|------|
| ① 合成器 | `merge_lowcmd_arm_sdk.py` | 500Hz | RL + 手臂仲裁 | `rt/lowcmd_rl` + `rt/arm_sdk` | `rt/lowcmd` → 电机 |
| ② RL 行走 | `run_pipeline.py`（外部） | ~50Hz | 步态策略推理 | IPC 文件 + `rt/lowstate` | `rt/lowcmd_rl` |
| ③ 决策层 | `box_demo_main.py` | 事件驱动 | 感知 + IK + 抓取编排 | D435 + VLM + SAM3 | IPC + `rt/arm_sdk` |

### 控制权状态机

整个系统在两种模式间切换，由 IPC 文件的 `fsm` 字段驱动：

```
        ┌─────────────────────────────────────────────┐
        │                                             │
        ▼                                             │
  ┌──────────┐  抓取准备    ┌──────────┐  抓取完成    │
  │ RL_FULL  │─────────────▶│ RL_LOWER │─────────────┘
  │ 全身 RL  │              │ 下身 RL  │
  │ 行走搜索 │◀─────────────│ 上身手臂 │
  └──────────┘   释放       └──────────┘
        │   motor[29].q        ▲
        │   淡出 1.0→0.0       │ motor[29].q=1.0
        │                      │ rt/arm_sdk 接管 12–29
        ▼                      │
    机器人行走            双臂执行抓取
```

| 模式 | 谁控制 0–11（下身） | 谁控制 12–29（腰+臂） | 触发条件 |
|------|-------------------|---------------------|---------|
| `RL_FULL` | RL 策略 | RL 策略（默认）或 arm_sdk（若权重>阈值） | `mover.initialize()` |
| `RL_LOWER` | RL 策略 | arm_sdk（必由 `motor[29].q=1.0` 接管） | `mover.shutdown()` |

---

## 工作流程

`box_demo_main.py` 的主循环分为以下步骤：

```
┌─────────────────────────────────────────────────┐
│  1. VLM 检测：拍照 → 发送给 VLM 判断是否有箱子     │
│     └─ 箱子被裁切？→ 侧移/后退调整后重拍           │
├─────────────────────────────────────────────────┤
│  2. SAM3 预测：RGB+深度 → SAM3 获取 3D 抓取点     │
│     └─ 只看到一个面？→ 根据深度前移/后退后重拍     │
├─────────────────────────────────────────────────┤
│  3. 坐标转换：相机光学系 → torso_link 坐标系       │
│     └─ 含手长补偿（X退16cm, Y偏5cm, Z抬9cm）      │
├─────────────────────────────────────────────────┤
│  3.x 工作空间检查与位置调整                       │
│     ├─ 太近？→ 后退                              │
│     ├─ 横向偏移 > 15cm？→ 侧移对齐（×1.2 预补偿）  │
│     ├─ 抓取点太远？→ 前移                         │
│     └─ IK 不可达？→ 几何估算移动距离（含自适应衰减）│
├─────────────────────────────────────────────────┤
│  4. 完整 IK 求解 → 17个关节角（左臂7+右臂7+腰3）   │
├─────────────────────────────────────────────────┤
│  5. 交权：RL_FULL → RL_LOWER，手臂接管上半身      │
├─────────────────────────────────────────────────┤
│  6. 执行双臂运动 → 抓取完成                       │
├─────────────────────────────────────────────────┤
│  7. 回到步骤1，检查是否还有箱子                    │
└─────────────────────────────────────────────────┘
```

---

## 机器人移动机制详解

机器人移动是整个抓取系统的关键能力之一。G1 的行走并非由 `box_demo_main.py` 直接驱动电机，而是通过**文件 IPC + RL 行走策略 + DDS 指令合成**三层架构间接完成。下面自底向上剖析这套机制。

### 总体架构

```
┌──────────────────────────────────────────────────────────────┐
│                  box_demo_main.py（决策层）                    │
│   VLM/SAM3/IK → 算出"需要前进多少 cm / 侧移多少 cm"             │
│                          │                                    │
│                          ▼                                    │
│              RobotMover.move_forward / move_left              │
│   把"距离"换算成"归一化速度 × 持续时长"                          │
└──────────────────────────┬───────────────────────────────────┘
                           │ 写入
                           ▼
              /tmp/robojudo_ext_cmd.json（文件 IPC）
              {
                "fsm": "RL_FULL",
                "velocity": {"forward": vx, "lateral": vy, "yaw": vyaw},
                "timestamp": ...
              }
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│       run_pipeline.py（RL 行走策略，独立进程）                  │
│  1. 每个控制步读取 /tmp/robojudo_ext_cmd.json                  │
│  2. 将归一化速度映射为实际线/角速度                              │
│     forward∈[-0.5,1.0] m/s, lateral∈[-0.5,0.5] m/s,           │
│     yaw∈[-0.5,0.5] rad/s                                      │
│  3. ONNX 策略推理 → 12 个下半身关节角                            │
│  4. 发布 rt/lowcmd_rl（仅电机 0–11）                            │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│     merge_lowcmd_arm_sdk.py（DDS 指令合成器，500Hz）           │
│  · 默认：rt/lowcmd_rl 全段直通 → rt/lowcmd                     │
│  · 若 rt/arm_sdk.motor_cmd[29].q > 0.001：                    │
│       用 arm_sdk 覆盖电机 12–29（腰+双臂）                     │
│  · CRC 校验后发布 rt/lowcmd                                   │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
                    宇树 G1 实机电机
```

### 第一层：RobotMover（决策到 IPC 的适配器）

`robot_move.py` 中 `RobotMover` 把"目标距离"翻译成"归一化速度 × 行走时长"，然后高频写入 IPC 文件。

#### IPC 文件格式

`/tmp/robojudo_ext_cmd.json` 是与 RL Pipeline 约定的协议：

```json
{
  "fsm": "RL_FULL",
  "velocity": {"forward": 0.5, "lateral": 0.0, "yaw": 0.0},
  "timestamp": 1718426400.123
}
```

- `fsm`：FSM 状态机切换标志，可选 `RL_FULL`（全身行走）、`RL_LOWER`（仅下半身保持平衡，上半身交给手臂）、`null`（不切换）
- `velocity`：三个归一化分量（**注意是归一化值，不是实际速度**），由 RL Pipeline 映射到实际量
- 写入使用 **`tempfile + os.rename` 原子替换**，避免 RL 端读到半截 JSON

#### 速度归一化映射常量

```python
_FWD_MAX = 1.0   # 前进 max=1.0 m/s
_LAT_MAX = 0.5   # 侧移 max=0.5 m/s
_YAW_MAX = 0.5   # 偏航 max=0.5 rad/s
```

RL Pipeline 在内部把 `forward ∈ [-1, 1]` 映射到 `[-0.5, 1.0] m/s`、`lateral` 映射到 `[-0.5, 0.5] m/s`。

#### 前进/后退：`move_forward(distance_m)`

关键设计——**RL 策略存在最小行走进度门槛**，归一化速度过低只会抖动而不前进。因此采用"夹紧 + 反算时长"策略：

1. 先把目标速度 `velocity / _FWD_MAX` 夹紧到 `[-1, 1]`
2. 若归一化速度 `< _MIN_FWD_NORM = 0.5`，强制抬到 `0.5`（保留符号），避免短距离时根本走不动
3. 用实际归一化速度反算：`duration_s = distance_m / (vx × _FWD_MAX)`
4. 强制下限 `duration_s ≥ _MIN_WALK_DURATION = 0.5s`，给 RL 启动加速留时间

#### 侧移：`move_left(distance_m)`

侧移比前进更"难驱动"——RL 策略对侧向命令的响应弱、欠追踪严重，因此**始终使用最大归一化侧向速度** `_MIN_LAT_NORM = 1.0`（与键盘 a/d 一致），仅通过时长控制位移：

- `vy = ±1.0`，`vy_raw = ±0.5 m/s`
- `duration_s = |distance| / 0.5`，下限 `0.6s`
- 调用方（`box_demo_main.py`）会额外把目标距离乘 `1.2` 来预补偿 RL 的侧移欠追踪

#### 原地转向：`rotate(angle_rad)`

- `yaw_speed` 默认 `0.35 rad/s`（约 20°/s）
- `vyaw = ±yaw_speed / _YAW_MAX` 归一化
- `duration_s = |angle| / yaw_speed`
- 注意：`box_demo_main.py` 默认不用 `mover.rotate()`，而是用 `WaistRotator` 通过 `rt/arm_sdk` 直接转腰（更精确、更安全），全身转向只在注释里保留

#### 高频刷新：`_walk_with_refresh`

```python
chunk = 0.1  # 100ms
while time.time() < deadline:
    _write_cmd(fsm="RL_FULL", forward=vx, lateral=vy, yaw=vyaw)
    time.sleep(chunk)
_write_cmd(fsm="RL_FULL", forward=0.0, lateral=0.0, yaw=0.0)  # 停住
time.sleep(0.5)  # 等惯性收敛
```

为什么必须每 100ms 刷新一次？因为 RL Pipeline 把 IPC 文件当作"持续按下的键盘按键"处理——若文件停止更新或被删除，pipeline 会回到默认模式（站立/松开），机器人会立刻停。**持续刷新 = 持续按住方向键**。

#### FSM 模式切换

| 模式 | 含义 | 何时使用 |
|------|------|---------|
| `RL_FULL` | RL 策略同时控制下半身 + 上半身 | 行走阶段（搜索/对齐/调整） |
| `RL_LOWER` | RL 仅控制下半身（电机 0–11），上半身交给 `arm_sdk` | 抓取阶段，让手臂接管 |
| `null` + 删除文件 | 不再持续写 IPC | 抓取结束，恢复手动遥控 |

关键坑：`shutdown()` 切到 `RL_LOWER` 后**不能删除 IPC 文件**，否则 pipeline 会切回默认模式重新控制上肢，与 `arm_sdk` 抢权导致全身抽搐。

### 第二层：RL 行走策略（`run_pipeline.py`）

这是真正驱动 G1 双腿的进程，位于 `locomotion/RoboJuDo_zihou2/`。它的核心职责：

1. **FSM 管理**：启动时通过 `prepare()` 让机器人从任意姿态站立到默认位姿
2. **IPC 读取**：每个控制步读 `/tmp/robojudo_ext_cmd.json`，把归一化速度 + FSM 状态注入观测向量
3. **ONNX 策略推理**：基于强化学习训练的步态策略，输入当前关节状态 + 期望速度，输出 12 个下肢关节目标角
4. **输出**：发布 `rt/lowcmd_rl`，仅含电机 0–11（髋、膝、踝、躯干底部等下半身关节）

> RL 策略本身是黑盒 ONNX 模型，本仓库未包含其权重和训练代码。可通过 `benchmark_policy_freq.py` 测试其推理频率。

### 第三层：DDS 指令合成（`merge_lowcmd_arm_sdk.py`）

这是整个系统的"神经中枢"，以 **500Hz** 把两个来源的命令合成成最终发给电机的 `rt/lowcmd`：

#### 仲裁逻辑

```python
# 1. RL 命令是基础（电机 0–34 全段先复制 RL）
_copy_full(out, rl)

# 2. 判断是否启用 arm_sdk 覆盖
use_arm = (
    arm is not None               # 收到过 arm_sdk 报文
    and arm_age <= 0.25s          # 报文未过期（< 250ms）
    and arm.motor_cmd[29].q > 1e-3  # 权重槽 > 阈值
)

# 3. 若启用，覆盖电机 12–29（腰3 + 左臂7 + 右臂7 + arm_sdk权重槽）
if use_arm:
    for i in range(12, 30):
        _copy_motor(out, arm, i)

# 4. CRC 校验后发布
out.crc = self._crc.Crc(out)
self._pub.Write(out)
```

#### 关键设计点

- **电机编号分区**：`0–11` 下半身（RL 独占）、`12–14` 腰部、`15–28` 双臂、`29` arm_sdk 权重槽、`30–34` 其他
- **`motor_cmd[29].q` 作为"软开关"**：手臂控制器通过把它设为 `1.0` 来声明"我要接管上半身"，设为 `0.0` 表示"我放手"。这比硬切换更平滑
- **新鲜度检查**：`arm_sdk` 超过 250ms 没更新就认为掉线，自动回退到 RL 上半身，避免手臂僵死
- **线程安全**：所有 DDS 回调通过 `threading.Lock` 保护快照，500Hz 工作线程只读快照

#### 交权流程（RL_FULL → RL_LOWER）

```
box_demo_main.py                RobotMover                 RL Pipeline          merge_lowcmd
     │                              │                          │                     │
     │  mover.shutdown()            │                          │                     │
     ├─────────────────────────────▶│                          │                     │
     │                              │ 写入 fsm="RL_LOWER"      │                     │
     │                              ├─────────────────────────▶│                     │
     │                              │                          │ 上肢不再被 RL 驱动  │
     │  controller.run()            │                          │                     │
     │  发布 rt/arm_sdk[q29=1.0] ───────────────────────────────────────────────────▶│
     │                              │                          │     merge 用 arm_sdk│
     │                              │                          │     覆盖 12–29      │
     │                              │                          │                     ▼
     │                              │                          │              rt/lowcmd → 电机
```

抓取完成后 `mover.initialize()` 再写回 `fsm="RL_FULL"`，上半身控制权还给 RL，进入下一轮搜索。

### 第四层：决策层移动触发点（`box_demo_main.py`）

主循环中触发移动的所有场景，按优先级顺序：

| 步骤 | 触发条件 | 动作 | 调用 |
|------|---------|------|------|
| VLM 裁切修正 | 箱子被画面边缘裁切（`left_cut` / `right_cut` / `both_cut`） | 侧移 6cm 或后退 5cm 让箱子完整显示 | `mover.move_left(±0.06)` / `mover.move_forward(-0.05)` |
| VLM 顶部裁切 | `top_cut`（太近，顶部出框） | 前移 4cm | `_walk(0.04, 0.0)` |
| 太近检查 | 最小维度 < 5cm 且最大维度 > 10cm 且 X < 35cm | 后退到 X=30cm | `mover.move_forward(-(0.3-x))` |
| 侧向对齐 | 箱子横向偏移 > 15cm（一次性） | 侧移偏移量 × 1.2 | `mover.move_left(y×1.2)` |
| 距离前移 | 抓取点平均 X > 45cm | 前移 (X-40) × 0.7 | `_walk((x-0.4)*0.7, 0)` |
| IK 工作空间兜底 | 抓取点 IK 误差 > 40mm | 几何估算前/横移，带衰减 | `_walk(safe_x, safe_y)` |
| IK 完全失败 | 完整 IK 不收敛 | 前移 10cm 重试 | `_walk(0.10, 0.0)` |

### IK 不可达时的几何估算与衰减策略

当 SAM3 给出的抓取点超出双臂工作空间时，系统不会盲目硬冲，而是用**纯几何估算 + 自适应衰减**逐步逼近：

#### 几何估算（`_compute_move_distance`）

```python
move_x = left_target[0] - IDEAL_REACH_X   # IDEAL_REACH_X = 0.40m
move_y = box_center_y                      # 箱子中心的横向偏移
```

即"想让左抓取点落在 X=40cm 处需要前/后退多少；同时把箱子中心拉到 Y=0"。

#### 衰减因子（防过冲）

每次 IK 失败重试都会更新衰减因子 `_move_decay`：

```python
_MOVE_DECAY_FACTOR = 0.6   # 衰减乘子
_MOVE_DECAY_FLOOR   = 0.6  # 衰减下限
_MOVE_MAX_X         = 0.50 # 单次最大前/后移 50cm
_MOVE_MAX_Y         = 0.30 # 单次最大横移 30cm

if cur_err < prev_err * 0.90:           # IK 误差明显缩小（进步了）
    _move_decay = min(1.0, _move_decay * 1.5)   # 衰减恢复（最大回到 1.0）
else:                                    # 没进步或更糟
    _move_decay = max(0.6, _move_decay * 0.6)   # 继续衰减（最低 0.6）

safe_x = clip(move_x * _move_decay, ±0.50)
safe_y = clip(move_y * _move_decay, ±0.30)
```

这套机制保证：靠近目标时不会一次冲过头，遇到 SAM3 噪声 / RL 欠追踪也不会在两点之间反复横跳。一旦 IK 通过、Step 4 会把衰减参数重置回 `1.0 / inf`。

### 行走距离补偿系数 `--walk-scale`

由于 RL 策略普遍存在"欠追踪"（命令走 50cm，实际只走 40cm），主入口提供 `--walk-scale` 全局补偿：

```python
def _walk(dist_x, dist_y):
    if abs(dist_x) >= 0.03:
        scaled = dist_x * args.walk_scale   # 关键：距离放大
        mover.move_forward(scaled)
```

- `--walk-scale 1.0`：不补偿（默认）
- `--walk-scale 1.2 ~ 1.5`：补偿 RL 欠追踪，侧移场景尤其推荐
- 仅作用于 `_walk()`（前后向），直接走 `mover.move_left()` 的侧移逻辑由调用方自行加 1.2 倍

### 移动流程时序图（一次典型对齐）

```
t=0.000s  box_demo: [侧移] 箱子中心偏左 22cm，跨步 26cm
t=0.000s  RobotMover: 左移 26cm (v=+0.50m/s, norm=+1.00, 0.52s)
          └─ 写 IPC：{fsm:RL_FULL, lateral:+1.0}，开始 100ms 刷新循环
t=0.100s  └─ 刷新 IPC
t=0.200s  └─ 刷新 IPC
   ...
t=0.520s  └─ 写 IPC：{lateral:0.0}（停住）
t=1.020s  sleep(0.5) 收尾，等惯性 + RL 稳定
t=1.520s  box_demo: 继续下一帧 SAM3 预测
```

---

## 视觉感知链路详解

视觉感知是整个抓取系统的"眼睛"，由 **VLM 引导 → SAM3 3D 预测 → 坐标转换** 三级串联完成"从像素到 torso 系坐标"的映射。下面逐级剖析。

### 第一级：VLM 引导（`vlm_guide.py`）

VLM（视觉语言模型）负责"粗判断"——告诉机器人箱子在不在画面里、是否被裁切、大致多远。

#### 输入参数

`VLMGuide.query(image_bgr, depth_u16)` 接收两路输入：

- **RGB 图像**：BGR 格式 numpy 数组，JPEG 编码质量 80
- **深度图**（可选）：uint16 毫米，经 `cv2.applyColorMap(..., COLORMAP_JET)` 转成 JET 色彩图（红=近、蓝=远），JPEG 质量 60，让 VLM "看懂"距离

#### System Prompt 设计

系统提示词约定了严格的输出协议（节选）：

```
你的任务：
1. 画面里有没有箱子？
2. 箱子是否完整？返回 complete / left_cut / right_cut / both_cut / top_cut
3. 用深度图估算距离，给出移动毫秒数（机器人行走速度约 0.2 m/s）

只返回 JSON：
{"box_visible": bool, "box_in_frame": "...",
 "move_x_ms": number, "move_y_ms": number,
 "confidence": "high/medium/low", "description": "..."}
```

#### 输出解析（容错）

`_parse_response` 用正则 `\{.*\}` 从 VLM 自由文本里抠出 JSON，再 `json.loads`。即便 VLM 多嘴加了 markdown 代码块或解释文字，也能稳定解析。失败时返回 `{"box_visible": False, "error": ...}`，主流程据此跳过本轮。

#### 主流程对 VLM 输出的使用

| 字段 | 用途 |
|------|------|
| `box_visible` | 决定是否进入 SAM3 预测阶段 |
| `box_in_frame` | 裁切类型 → 触发侧移/后退调整后重拍（见[机器人移动机制详解](#机器人移动机制详解)的决策触发点表） |
| `move_x_ms` / `move_y_ms` | 始终未参与决策的行走距离建议（实际抓取循环里以 SAM3+IK 为准，VLM 建议主要用于裁切修正） |
| `confidence` | 日志参考，不参与决策 |

### 第二级：SAM3 3D 预测（`capture_and_predict.py` + `sam3_client.py`）

SAM3 服务端负责"精细预测"——从 RGB+深度图直接回归出箱子的 3D 包围盒和双臂抓取点。

#### HTTP 协议

```
POST http://192.168.112.198:5300/predict
Content-Type: multipart/form-data

files:
  rgb:   rgb.jpg   (image/jpeg)   ← BGR 编码的 JPEG
  depth: depth.png (image/png)    ← uint16 毫米深度图，PNG 无损保留精度
data:
  prompt: "box"                ← 文本提示词
```

**关键设计**：深度图必须用 **PNG**（不能 JPEG），因为 JPEG 有损压缩会破坏 uint16 毫米精度，直接导致 SAM3 的 3D 回归出错。

#### 返回的 3D 包围盒

```python
{
  "center": [x, y, z],          # 箱子中心（相机光学系，米）
  "extent": [e0, e1, e2],      # 三轴跨度（米）
  "rotation_matrix": [[...]],   # 3×3 旋转矩阵
  "length": float,            # 沿长轴长度（米）
  "grasp_left":  [x, y, z],   # 左手抓取点（箱子左侧面中点）
  "grasp_right": [x, y, z]   # 右手抓取点（箱子右侧面中点）
}
```

抓取点不是箱子中心，而是**箱子两个窄面的中点**——这样双臂从两侧夹住箱子，符合人手"抱"的直觉。

#### 失败重试逻辑

主流程检测 SAM3 输出：如果 `min(extent) < 5cm 且 max(extent) > 10cm 且 X < 35cm`，判定"只看到一个面（太近）"，触发后退重拍。这是 SAM3 在近距离时只能回归出一个面的已知局限。

### 第三级：坐标转换（`box_demo_main.py: cam_to_torso`）

SAM3 返回的点在**相机光学坐标系**（Z=前、X=右、Y=下），必须转到 **torso_link**（X=前、Y=左、Z=上）才能喂给 IK。

#### 三段固定变换

```python
# 1. 光学系 → d435 body 系（固定旋转）
_R_BODY_OPTICAL = [[0,0,1],[-1,0,0],[0,-1,0]]
#   含义: X_body = Z_opt, Y_body = -X_opt, Z_body = -Y_opt

# 2. d435 body → torso_link（俯仰旋转 + 平移）
_PITCH = 0.8308 rad  ≈ 47.6°（来自 URDF d435_joint）
_R_TORSO_BODY = Ry(_PITCH)
_T_TRANS = [0.0576, 0.0175, 0.4299]  # 安装位置（米）

# 总变换：p_torso = R_TORSO_BODY @ R_BODY_OPTICAL @ p_optical + T_TRANS
```

#### 手长补偿（关键）

SAM3 给的抓取点是箱子表面上的点，但机器人手掌中心到手腕还有 12cm（`_RIGHT_EE = [0.12, 0, 0]`）。如果不补偿，IK 求出的关节角会让手腕"穿进"箱子。主流程对左右抓取点统一施加：

```python
left_torso[0]  -= 0.16;  left_torso[1]  += 0.05;  left_torso[2]  += 0.09
right_torso[0] -= 0.16;  right_torso[1] += 0.05; right_torso[2] += 0.09
```

| 分量 | 值 | 含义 |
|------|----|------|
| X: −0.16m | 后退 16cm | 让手腕落在箱子表面外侧，而不是穿进去 |
| Y: +0.05m | 左偏 5cm | 补偿手臂展开的自然偏移，让两掌心对齐箱子中线 |
| Z: +0.09m | 上抬 9cm | 补偿手掌厚度，避免蹭到箱子顶面 |

这套补偿量是实测调优的，改 URDF 或换箱子尺寸时可能要重调。

### 感知链路全景时序

```
D435 拍 RGB+Depth
     │
     ▼ JPEG80 + JET色图60（VLM 用）
     │
[VLM]  ← 判断: 箱子在不在？裁切？
     │ box_in_frame != complete → 侧移/后退 → 重拍
     ▼ box_in_frame == complete
     │
[SAM3] ← POST /predict (RGB JPEG + Depth PNG)
     │
     ▼ 返回 center/extent/R/grasp_left/grasp_right（相机光学系）
     │
[cam_to_torso]  ← R_BODY_OPTICAL → Ry(pitch) → +T_TRANS
     │
     ▼ 手长补偿 (X−0.16, Y+0.05, Z+0.09)
     │
     ▼ left_torso / right_torso（torso_link 系，米）
     │
     ▼ 喂入 IK
```

---

## 逆运动学（IK）详解

IK 是"从手心位置反推 7 个关节角"的过程，是双臂抓取能否执行的最后关卡。本系统的 IK 完全部在 `dual_arm_target_reach.py` 的 `ArmKinematics` 类里，**不依赖 pinocchio / PyBullet 等运行时**，参数直接来自 G1 URDF。

### 关节链模型

每条手臂是 **7 自由度**链：

```
肩俯仰 → 肩横滚 → 肩偏航 → 肘 → 腕横滚 → 腕俯仰 → 腕偏航
  15       16       17      18    19       20         21   (左臂)
  22       23       24      25    26       27         28   (右臂)
```

每个关节在 URDF 里有固定的 `(xyz, rpy, rotation_axis)` 三元组（父链接到关节系的齐次变换 + 关节转动轴）。`forward_kinematics` 就是把这 7 个 4×4 矩阵连乘，再乘末端执行器偏移 `_EE = [0.12, 0, 0]`。

### 逆运动学算法：阻尼最小二乘（DLS）

```python
for _ in range(max_iter):
    pos, _ = forward_kinematics(q, left)
    err = target - pos
    if norm(err) < tol: break
    J  = jacobian(q, left)         # 3×7 数值雅可比
    dq = J.T @ solve(J @ J.T + damping²·I, err)  # 阻尼最小二乘步
    q += dq
    q  = clip(q, joint_limits)      # 投到关节限位
```

- **阻尼项** `damping² · I` 保证 `J @ J.T` 奇异时仍有解，避免单点发散
- **关节限位** 来自 URDF（如左肘 `[−1.0472, 2.0944] rad`），防止求出不存在的姿态
- **数值雅可比**：用前向差分（`eps=1e-6`）逐列扰动算，不依赖自动微分框架

### 多起点重启（防局部最优）

单个起点容易陷局部最优。`inverse_kinematics` 默认从 **10 个不同随机起点**各跑一次 DLS，取误差最小的解：

```python
def inverse_kinematics(target_pos, left=True, q0=None,
               max_iter=1000, tol=1e-4, damping=0.02,
               num_restarts=10):
    best_q, best_err = None, inf
    for _ in range(num_restarts):
        q_start = q0 or random_in_limits()
        q, err = _ik_single(target_pos, left, q_start, ...)
        if err < best_err:
            best_q, best_err = q, err
    return best_q, best_err
```

### 双层 IK 策略（快检 + 完整检）

主流程为速度起见，用了**两套 IK**：

| 阶段 | 用途 | 参数 | 耗时 |
|------|------|------|------|
| **快检 IK** (`_ik_fast`) | 工作空间检查、计算移动距离 | `num_restarts=3, max_iter=250, tol=默认` | 250 次迭代×3 起点，亚秒级，用于判断"这个点抓得到吗" |
| **完整 IK** | 真正执行抓取前的最终求解 | `num_restarts=10, max_iter=1000, tol=1e-4` | 毫秒级，确保关节数精准 |

快检误差阈值 `IK_ERR_LIMIT = 0.040m`（40mm）——比完整 IK 的收敛阈值松 400 倍，但**只要快检通过，完整 IK 必定收敛**（设计冗余）。快检不通过则触发几何估算移动（见[机器人移动机制详解](#机器人移动机制详解)的衰减策略）。

### IK 精度参考

实测末端位置误差 **< 0.1mm**（`dual_arm_target_reach.py` 文件头注释）。精度来源：

1. 关节参数直接抄自 G1 URDF 的 joint origin（xyz/rpy 各 6 位小数）
2. 末端偏移 `[0.12, 0, 0]` 精确到手掌心
3. DLS 收敛阈值 `1e-4 m`（0.1mm）

---

## 双臂抓取流程详解

当 IK 通过、机器人已对齐，进入"真抓取"阶段。这一阶段由 `DualArmController`（在 `dual_arm_target_reach.py`）驱动，发布 `rt/arm_sdk` 控制电机 12–29。

### 预备姿态（ZERO_Q）

抓取前先把双臂摆到"双手叉腰"预备位（不是零位），避免手臂乱晃打到货架：

```python
ZERO_Q = [0.0] * 17   # 左臂7 + 右臂7 + 腰3
# 左臂（索引 0–6）
ZERO_Q[0] = _ZERO_ShoulderPITCH    # 84.9°  肩前抬
ZERO_Q[1] = _ZERO_ShoulderROLL     #  6.8°  肩外张
ZERO_Q[2] = _ZERO_ShoulderYAW      # 13.8°
ZERO_Q[3] = -_ZERO_ELBOW        # -42.4° 肘曲
ZERO_Q[5] = -_ZERO_WristPitch   # -38.7°
ZERO_Q[6] = _ZERO_WristYAW       # 7.7°
ZERO_Q[7..13] = 镜像反号   # 右臂（索引 7–13）
# 腰（索引 14–16）保持 0
```

### 运动阶段（cosine 插值）

每段运动用**余弦插值** `0.5*(1−cos(πt/T))`，起止速度都为零，无超调：

```
当前位姿 ──3s──→ ZERO_Q（预备）
ZERO_Q  ──5s──→ 抓取位（IK 解）
抓取位 ──2s──→ 闭合位（Y 内移 0.11m，夹住箱子）
闭合位 ──2s──→ 抬升位（Z 上抬 0.20m，提起箱子）
抬升位 ──2s──→ 闭合位（放下）
闭合位 ──2s──→ 抓取位（退回）
抓取位 ──3s──→ ZERO_Q（回预备）
ZERO_Q  ──2s──→ 逐渐释放控制权（motor_cmd[29].q 淡出）
```

### arm_sdk 权重槽淡出（防抖）

最后阶段不直接撒手（会让 RL 上半身猛然接管，引发全身抖动），而是把 `motor_cmd[29].q` 从 1.0 线性淡出到 0.0：

```python
for t in range(淡出步数):
    weight = 1.0 - t / 淡出步数     # 1.0 → 0.0
    publish_arm_sdk(target_q, weight=weight)
```

merge 端检测到 `motor_cmd[29].q < 1e-3` 后自动回退到 RL 上半身，完成无感交权。

### 腰部控制（WaistRotator）

抓取前的"转腰搜索"和"对齐"用 `WaistRotator`，不走 RL 行走：

- 通过 `rt/arm_sdk` 同时发布腰 yaw 目标 + 双臂保持位姿
- 余弦插值，3.5s 完成旋转，期间 20Hz 刷新
- 旋转完启 keepalive 线程（20Hz）持位，防腰回弹
- `sync_yaw()` 在 RL 接管腰后重新读 `lowstate.motor_state[12].q` 同步累计角，避免腰角"跳变"

`KP_WAIST=20, KD_WAIST=1.5`（软伺服，顺滑）；`KP_ARM=40, KD_ARM=2.0`（手臂锁紧防走动时晃）。

### 交权时序（RL_FULL ↔ RL_LOWER）

```
[搜索/对齐阶段]  fsm=RL_FULL  →  RL 控全身，手臂由 ZERO_Q keepalive 持定位
        │
        ▼ IK 通过，准备好抓取
        │
[抓取阶段]      fsm=RL_LOWER →  RL 仅控下半身
        │                publish rt/arm_sdk[motor[29].q=1.0]
        │                merge 用 arm_sdk 覆盖电机 12–29
        │
        ▼ 抓取完成，motor[29].q 淡出
        │
[下一轮搜索]    fsm=RL_FULL  →  RL 重新控全身
```

---

## RL 行走策略详解

G1 的行走能力来自一个基于强化学习训练的步态策略，由独立进程 `run_pipeline.py` 加载运行。训练代码和权重在 `locomotion/RoboJuDo_zihou2/`（外部仓库，不包含在本项目中），本节基于仓库中可观测的接口、模型规格和依赖描述其工作方式。

### 两个 ONNX 模型

从 `benchmark_policy_freq.py` 可以确认，系统同时维护**两个**策略模型：

| 模型 | 文件 | 观测维度 | 动作维度 | 对应模式 |
|------|------|---------|---------|---------|
| `full_onnx` | `g1_velocity.onnx` | 96 | 29 | `RL_FULL`（全身 29 DOF） |
| `lower_onnx` | `policy.onnx` | 79 | 12 | `RL_LOWER`（仅下半身 12 DOF） |

- **full_onnx**：观测 96 维（关节角 + 关节速度 + 上一步动作 + 命令速度 + 重力投影等），输出 29 个关节目标角，对应 `RL_FULL` 模式下控制全身
- **lower_onnx**：观测 79 维，**只输出 12 个下肢关节角**，对应 `RL_LOWER` 模式——上半身留给 `arm_sdk` 接管

模型路径（相对仓库根）：

```
locomotion/RoboJuDo_zihou2/assets/models/g1/mjlab_loco/
├── g1_velocity.onnx   ← full
└── policy.onnx        ← lower
```

### 训练栈推断

从 `requirements_system.txt` 和文件命名可以推断训练技术栈：

| 依赖 | 版本 | 推断用途 |
|------|------|---------|
| `mujoco==3.8.1` | 3.8.1 | 物理仿真器，RL 训练环境 |
| `onnxruntime==1.23.2` | 1.23.2 | 实机推理引擎（CPU provider） |
| `RoboJuDo==1.5.0` | 1.5.0 | 宇树机器人 RL 训练框架（`MjlabLocoPolicy`） |
| `scipy==1.8.0` | 1.8.0 | 信号处理（动作滤波） |
| `msgpack` + `msgpack-numpy` | — | 进程间观测/动作序列化 |

从启动命令 `run_pipeline.py -c g1_mjlab_loco_real_merge` 中的 **`mjlab`** 可知训练用的是 **MJxLab**（基于 MuJoCo 的 RL 训练库），**`real_merge`** 后缀表明这是为"实机 + merge 合成"场景专门训练的版本，能配合 `merge_lowcmd_arm_sdk.py` 工作。

### 推理循环（运行时）

`run_pipeline.py` 在实机上的典型推理循环（基于 IPC 协议和 50Hz 目标推断）：

```
┌──────────────────────────────────────────────────────────┐
│  每个控制步（目标 50Hz / 20ms）：                          │
│                                                          │
│  1. 读 rt/lowstate → 12–29 个关节的 (q, dq)              │
│  2. 读 /tmp/robojudo_ext_cmd.json                        │
│     → 提取 (forward, lateral, yaw) 归一化命令             │
│     → 提取 fsm 状态，决定用 full 还是 lower 模型          │
│  3. 组装观测向量 obs（96 或 79 维）：                      │
│     - 本体感受：关节角 sin/cos + 关节速度                 │
│     - 速度命令：(vx, vy, vyaw) 归一化值                   │
│     - 上一步动作                                          │
│     - IMU / 重力投影                                      │
│  4. onnx_session.run(obs) → 12 或 29 个目标关节角         │
│  5. （可选）动作低通滤波，平滑输出                         │
│  6. 填充 LowCmd_（kp/kd/q/dq），发布 rt/lowcmd_rl         │
└──────────────────────────────────────────────────────────┘
```

### 命令映射（归一化 → 实际速度）

`RobotMover` 写入的归一化值，在 pipeline 端被映射成实际速度（与 `robot_move.py` 的常量呼应）：

| 字段 | 归一化范围 | 实际范围 | 缩放 |
|------|-----------|---------|------|
| `forward` | [−1, 1] | [−0.5, 1.0] m/s | `_FWD_MAX = 1.0` |
| `lateral` | [−1, 1] | [−0.5, 0.5] m/s | `_LAT_MAX = 0.5` |
| `yaw` | [−1, 1] | [−0.5, 0.5] rad/s | `_YAW_MAX = 0.5` |

注意 forward 的实际范围是 **[−0.5, 1.0]**——后退最大只有前进最大的一半，这是训练时为了安全人为限制的（人形机器人后退比前进更容易摔倒）。

### 为什么短距离要走"满速"

`robot_move.py` 里有几条看似奇怪的常量，根源都在 RL 策略的训练分布：

| 常量 | 值 | 训练层面的原因 |
|------|----|--------------|
| `_MIN_FWD_NORM = 0.5` | 前进归一化下限 | 训练时命令速度集中在 0.5–1.0，低于 0.5 的样本少，策略会"抖动"而不前进 |
| `_MIN_LAT_NORM = 1.0` | 侧移始终满速 | 侧移训练样本稀疏，只有满档命令才能可靠触发侧步步态 |
| `_MIN_WALK_DURATION = 0.5s` | 最短时长 | 策略启动加速需 0.3–0.5s，更短的命令机器人还没动起来就停了 |
| `--walk-scale` 距离补偿 | 默认 1.0 | sim-to-real gap：仿真训练的位移跟踪在实机偏小 |

这些"经验值"本质上是 RL 策略 sim-to-real 迁移的补偿——策略不是在所有命令下都表现一致，而是有其"舒适区"。

### FSM 状态机（pipeline 端）

pipeline 内部维护一个 FSM，`prepare()` 阶段让机器人从任意姿态站立到默认位姿，之后根据 IPC 的 `fsm` 字段切换：

```
            启动
             │
             ▼
      ┌────────────┐
      │  PREPARE   │  prepare()：PD 控制器驱动到站立姿态
      └─────┬──────┘
            │ 站立完成
            ▼
      ┌────────────┐  IPC: fsm=RL_FULL   ┌────────────┐
      │   IDLE     │────────────────────▶│  RL_FULL   │
      │ (默认站立)  │◀────────────────────│  (full模型) │
      └─────┬──────┘   停止命令           └─────┬──────┘
            │                                  │ IPC: fsm=RL_LOWER
            │                                  ▼
            │                           ┌────────────┐
            └───────────────────────────│  RL_LOWER  │
               IPC: fsm=null/删除文件    │ (lower模型)│
                                        └────────────┘
```

**关键坑**（来自 `robot_move.py` 注释）：切到 `RL_LOWER` 后**不能删除 IPC 文件**。如果删除，pipeline 会回到默认 IDLE 模式重新控制上肢，与 `arm_sdk` 抢权，导致全身抽搐。`mover.shutdown()` 因此只写 `RL_LOWER` 而不删文件。

### 性能基准

用 `benchmark_policy_freq.py` 测量推理延迟（CPU provider，Jetson 平台）：

```
python benchmark_policy_freq.py          # 测全部模型
python benchmark_policy_freq.py --full   # 只测 full-body
python benchmark_policy_freq.py -n 1000  # 跑 1000 次
```

输出示例（目标 50Hz / 20ms 余量）：

```
Full-body ONNX (29 DOF)
  加载时间:    ~150 ms
  推理延迟:
    mean   ~2-5 ms   (200-500 Hz)   ✓ 远超 50Hz 需求
    p99    ~8 ms
  50Hz 余量:   ~15 ms ✓ 够用
```

**结论**：ONNX 推理本身远不是瓶颈（亚 10ms），实际 50Hz 上限主要由 IPC 文件 IO + DDS 序列化 + PD 控制器刷新决定。

### 训练定制点（基于配置名推断）

启动配置 `g1_mjlab_loco_real_merge` 暗示的训练定制：

| 标记 | 含义 |
|------|------|
| `g1` | 针对 G1 29 DOF 机器人训练 |
| `mjlab` | 使用 MJxLab 训练库（基于 MuJoCo） |
| `loco` | locomotion（行走）任务 |
| `real` | 为实机部署训练（含 domain randomization 抗 sim-to-real gap） |
| `merge` | 配合 `merge_lowcmd_arm_sdk.py` 使用，能输出 `rt/lowcmd_rl` 而非直接 `rt/lowcmd` |

如果你想训练自己的策略或调整观测/动作空间，需要进入 `locomotion/RoboJuDo_zihou2/` 仓库修改训练配置——本项目只消费其 ONNX 产物。

---

## 核心常量与参数表

散落在各文件里的关键常量集中索引，方便调参。

### 相机与坐标（`box_demo_main.py`）

| 常量 | 值 | 含义 |
|------|----|------|
| `HEAD_CAMERA_SERIAL` | `406122070550` | D435 序列号 |
| `_T_TRANS` | `[0.0576, 0.0175, 0.4299]` m | 相机相对 torso_link 安装位置 |
| `_PITCH` | `0.8308` rad (47.6°) | 相机俯仰角 |
| 手长补偿 X | `-0.16` m | 后退 |
| 手长补偿 Y | `+0.05` m | 左偏 |
| 手长补偿 Z | `+0.09` m | 上抬 |

### 行走与对齐（`box_demo_main.py` + `robot_move.py`）

| 常量 | 值 | 含义 |
|------|----|------|
| `IDEAL_REACH_X` | `0.40` m | 理想抓取距离（手臂半伸） |
| `MAX_REACH_X` | `0.45` m | 超过此则前移 |
| 太近阈值 | `X < 0.35` 且 `min(extent)<0.05` | 触发后退 |
| 侧移阈值 | `|Y| > 0.15` m | 触发侧移对齐 |
| `_MIN_FWD_NORM` | `0.5` | 前进归一化下限（低则抖） |
| `_MIN_LAT_NORM` | `1.0` | 侧移始终满归一化 |
| `_MIN_WALK_DURATION` | `0.5` s | 最短行走时长 |
| 侧移预补偿 | `×1.2` | 多走 20% 补 RL 欠追踪 |

### IK（`box_demo_main.py` + `dual_arm_target_reach.py`）

| 常量 | 值 | 含义 |
|------|----|------|
| `IK_ERR_LIMIT` | `0.040` m | 快检通过阈值（40mm） |
| `IK_FAST_RESTARTS` | `3` | 快检起点数 |
| `IK_FAST_MAX_ITER` | `250` | 快检迭代数 |
| 完整检 `num_restarts` | `10` | 完整检起点数 |
| 完整检 `max_iter` | `1000` | 完整检迭代数 |
| 完整检 `tol` | `1e-4` m | 完整检收敛阈值（0.1mm） |
| `damping` | `0.02` | DLS 阻尼系数 |

### 衰减策略（`box_demo_main.py`）

| 常量 | 值 | 含义 |
|------|----|------|
| `_MOVE_DECAY_FACTOR` | `0.6` | 衰减乘子 |
| `_MOVE_DECAY_FLOOR` | `0.6` | 衰减下限 |
| `_MOVE_MAX_X` | `0.50` m | 单次最大前/后移 |
| `_MOVE_MAX_Y` | `0.30` m | 单次最大横移 |
| 恢复因子 | `×1.5` | IK 误差缩小时恢复衰减 |

### DDS 合成（`merge_lowcmd_arm_sdk.py`）

| 常量 | 默认值 | 含义 |
|------|------|------|
| `--hz` | `500` | 发布 rt/lowcmd 频率 |
| `--weight-threshold` | `1e-3` | arm_sdk 权重槽生效阈值 |
| `--arm-stale-s` | `0.25` s | arm_sdk 超时回退 RL |
| `--rl-stale-s` | `0.5` s | RL 超时停止写 lowcmd |
| `OVERLAY_LO` | `12` | arm_sdk 覆盖起始电机 |
| `OVERLAY_HI` | `30` (exclusive) | arm_sdk 覆盖结束电机 |

---

## 文件说明

| 文件 | 说明 |
|------|------|
| `box_demo_main.py` | **主入口**，完整的搜索+抓取循环，包含腰部旋转控制、VLM检测、坐标转换、IK检查和抓取执行 |
| `dual_arm_target_reach.py` | 双臂末端位置控制器，基于 G1 URDF 的逆运动学（IK精度 < 0.1mm），负责关节轨迹插值和运动执行 |
| `capture_and_predict.py` | 相机采集与 SAM3 预测封装，管理 RealSense D435 的预热、启停、采集，并将图像发送给 SAM3 服务端 |
| `vlm_guide.py` | VLM（视觉语言模型）引导模块，调用千问/Qwen VLM API 检测箱子位置和移动建议 |
| `robot_move.py` | 机器人行走控制，通过文件 IPC (`/tmp/robojudo_ext_cmd.json`) 与 RL Pipeline 通信 |
| `sam3_client.py` | SAM3 3D BBox 客户端，向 SAM3 服务端发送 RGB+深度图获取 3D 包围盒和抓取点 |
| `merge_lowcmd_arm_sdk.py` | **指令合成器**，将 RL 行走指令 (`rt/lowcmd_rl`) 与手臂指令 (`rt/arm_sdk`) 合成为最终 `rt/lowcmd` |
| `start.sh` | 一键启动脚本，创建 tmux 三窗格布局分别运行 merge、RL pipeline 和 box_demo |
| `env_robojudo_zihou2.yml` | Conda 环境配置文件（Python 3.10，aarch64 Linux） |
| `requirements_system.txt` | pip 依赖清单（含系统级特殊包说明） |
| `box_demo_main_变量注释.txt` | `box_demo_main.py` 全部变量和参数的详细中文注释 |

---

## 环境配置

### 硬件要求

- **机器人**: 宇树 G1（29 DOF 版本）
- **头部相机**: Intel RealSense D435（序列号 `406122070550`）
- **计算平台**: Jetson / aarch64 Linux（机器人板载）
- **网络**: 机器人与 SAM3 服务端在同一局域网

### 软件依赖

#### Conda 环境（推荐）

```bash
conda env create -f env_robojudo_zihou2.yml
conda activate robojudo_zihou2
```

#### pip 安装

```bash
pip install -r requirements_system.txt
```

#### 系统级依赖

以下包需要特殊安装方式，不通过 pip：

| 包 | 安装方式 |
|----|---------|
| `pyrealsense2` | Intel RealSense SDK 安装脚本 |
| `Jetson.GPIO` | Jetson 平台自带 |
| `tensorrt` / `triton` | NVIDIA TensorRT/Triton |
| `CUDA/cuDNN` | NVIDIA 驱动附带 |

#### 外部服务

| 服务 | 用途 | 默认地址 |
|------|------|---------|
| **SAM3 3D BBox 服务端** | 3D 包围盒预测与抓取点计算 | `192.168.112.198:5300` |
| **Qwen VLM API** | 箱子检测与移动引导 | `https://dashscope.aliyuncs.com/compatible-mode/v1` |

需要设置环境变量：

```bash
export QWEN_API_KEY="your-api-key-here"
```

#### CycloneDDS

脚本启动时会自动加载 `~/cyclonedds-0.10-install/lib/libddsc.so.0`，也可通过环境变量指定：

```bash
export CYCLONEDDS_HOME=/path/to/cyclonedds-install
```

---

## 快速启动

### 启动前准备（必做）

1. **设置 VLM API Key**（三个终端都需要，建议写入 `~/.bashrc`）：

```bash
export QWEN_API_KEY="your-api-key-here"
```

2. **确认 SAM3 服务端在线**（默认 `192.168.112.198:5300`）：

```bash
curl http://192.168.112.198:5300/
```

3. **确认机器人状态**：G1 已通电、急停已解除、DDS 网络接口名（本例 `enP8p1s0`）正确：

```bash
ip addr show enP8p1s0
```

4. **设置 `LD_LIBRARY_PATH`**（避免加载错误的 `libddsc.so`，三个终端都需要）：

```bash
export LD_LIBRARY_PATH=/home/unitree/miniconda3/envs/robojudo_zihou2/lib:$LD_LIBRARY_PATH
```

### 一键启动（推荐）

```bash
./start.sh --vlm-endpoint https://dashscope.aliyuncs.com/compatible-mode/v1 --iface enP8p1s0
```

该脚本会创建名为 `g1-grasp` 的 tmux session，包含三个窗格：

| 窗格 | 程序 | 说明 |
|------|------|------|
| 左上 | `merge_lowcmd_arm_sdk.py` | 指令合成 |
| 右侧 | `run_pipeline.py` | RL 行走策略 |
| 左下 | `box_demo_main.py` | 主抓取流程 |

**tmux 操作：**

- `Ctrl+B` 方向键 — 切换窗格
- `Ctrl+B D` — 退出（后台继续运行）
- `tmux attach -t g1-grasp` — 重新连入

### 手动启动（三个终端，严格按顺序）

> **启动顺序：merge → RL pipeline → box_demo_main**，否则 DDS 话题可能冲突。

#### 终端 1 — 指令合成（必须最先启动）

```bash
conda activate robojudo_zihou2
export LD_LIBRARY_PATH=/home/unitree/miniconda3/envs/robojudo_zihou2/lib:$LD_LIBRARY_PATH
cd /home/unitree/unitree/unitree
python zihou/box_demo_2/merge_lowcmd_arm_sdk.py --iface enP8p1s0
```

**作用**：500Hz 把 `rt/lowcmd_rl`（行走）与 `rt/arm_sdk`（手臂）合成为最终 `rt/lowcmd` 发给电机，是整个系统的中枢。

#### 终端 2 — RL 行走策略（第二个启动）

```bash
conda activate robojudo_zihou2
cd ~/unitree/unitree/locomotion/RoboJuDo_zihou2
python scripts/run_pipeline.py -c g1_mjlab_loco_real_merge
```

**作用**：加载 `g1_velocity.onnx`（全身）和 `policy.onnx`（仅下半身）两个策略模型，读取 `/tmp/robojudo_ext_cmd.json` IPC 文件，输出 `rt/lowcmd_rl`，并响应 `fsm` 字段在 `RL_FULL`/`RL_LOWER` 间切换。

#### 终端 3 — 主抓取流程（最后启动）

```bash
conda activate robojudo_zihou2
export LD_LIBRARY_PATH=/home/unitree/miniconda3/envs/robojudo_zihou2/lib:$LD_LIBRARY_PATH
cd /home/unitree/unitree/unitree

python zihou/box_demo_2/box_demo_main.py \
    --iface enP8p1s0 \
    --vlm-endpoint https://dashscope.aliyuncs.com/compatible-mode/v1 \
    --vlm-api-key $QWEN_API_KEY \
    --vlm-model qwen-vl-max \
    --walk-scale 1.2
```

### 推荐运行参数（终端 3）

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `--iface` | `enP8p1s0` | DDS 网卡（必填） |
| `--walk-scale` | `1.2` | 补偿 RL 侧移欠追踪（仅作用于前后向） |
| `--no-confirm` | 视情况 | 跳过每次抓取前的 Enter 确认（**首次运行建议保留确认**） |
| `--vlm-max-iter` | `5` | 最大抓取尝试次数 |
| `--vlm-model` | `qwen-vl-max` | VLM 模型 |

### 启动前安全检查

启动终端 3 前确认：

- [ ] 机器人周围 1.5m 内无障碍物，手臂可自由活动
- [ ] 终端 1 日志显示 500Hz 发布 `rt/lowcmd`（无报错）
- [ ] 终端 2 日志显示 RL pipeline 已进入 IDLE / prepare 完成状态
- [ ] D435 相机就绪（脚本会自动停止宇树自带的 `videohub_pc4` 服务）
- [ ] 急停按钮在手边可触达，`Ctrl+C` 可随时中断

### 常见启动坑

| 现象 | 原因 | 解决 |
|------|------|------|
| 报错找不到 `unitree_sdk2py` / `pyrealsense2` | 未 `conda activate` | 三个终端都要先 `conda activate robojudo_zihou2` |
| DDS 话题收不到 / 电机不动 | 加载到错误版本的 `libddsc.so` | 显式 `export LD_LIBRARY_PATH=...`（见启动前准备） |
| VLM 调用 401 / 主循环卡在第一次拍照 | `QWEN_API_KEY` 未设置或为空 | `export QWEN_API_KEY="..."` |
| 启动后机器人抽搐 | 启动顺序错误，merge 抢不到 `rt/lowstate` | 严格按 merge → RL → box_demo 顺序 |
| 第一次抓取位置不对 | 没看 IK 输出的坐标就执行 | **首次运行不要加 `--no-confirm`**，按 Enter 前核对目标坐标 |

---

## 命令行参数

### box_demo_main.py

```
python box_demo_main.py [OPTIONS]
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--prompt` | `box` | SAM3 提示词，告诉模型要检测的物体类别 |
| `--host` | `192.168.112.198` | SAM3 服务端 IP 地址 |
| `--port` | `5300` | SAM3 服务端端口 |
| `--iface` | 自动 | DDS 网络接口名（如 `eth0`、`enP8p1s0`） |
| `--no-confirm` | `False` | 跳过执行前的用户确认提示 |
| `--end-behavior` | `handoff_rl_lower` | 结束行为：`handoff_rl_lower`（对齐 RL 后释放）或 `release`（直接释放） |
| `--vlm-endpoint` | DashScope | VLM API 地址（OpenAI 兼容格式） |
| `--vlm-api-key` | `$QWEN_API_KEY` | VLM API Key |
| `--vlm-model` | `qwen-vl-max` | VLM 模型名称 |
| `--vlm-max-iter` | `5` | 最大抓取尝试次数 |
| `--vlm-velocity` | `0.7` | 行走速度 (m/s) |
| `--walk-scale` | `1.0` | 行走距离补偿系数（>1 走更远，补偿 RL 欠追踪） |
| `--vlm-min-move` | `200` | 最小移动时长 (ms) |

### merge_lowcmd_arm_sdk.py

```
python merge_lowcmd_arm_sdk.py --iface enP8p1s0
```

| 参数 | 说明 |
|------|------|
| `--iface` | DDS 网络接口名（必填） |

### start.sh

```bash
./start.sh --vlm-endpoint URL [--iface enP8p1s0] [--walk-scale 1.0] [--confirm]
```

| 参数 | 说明 |
|------|------|
| `--vlm-endpoint` | VLM API 地址（必填） |
| `--iface` | DDS 网络接口（默认 `enP8p1s0`） |
| `--walk-scale` | 行走距离补偿系数（默认 `1.0`） |
| `--confirm` | 启用执行前确认提示 |

---

## 模块详解

### VLM 引导 (`vlm_guide.py`)

调用千问 VLM（`qwen-vl-max`）API，发送 RGB 图像和 JET 色彩映射的深度图。使用 OpenAI 兼容接口格式。

#### VLM 返回字段说明

VLM 模块分析机器人头部相机的 RGB 图像和深度图后，返回一个 **JSON 字典**：

```python
{
    "box_visible": bool,        # 是否检测到箱子
    "box_in_frame": str,        # 箱子在画面中的完整程度
    "move_x_ms": float,         # 前后移动建议（毫秒）
    "move_y_ms": float,         # 左右移动建议（毫秒）
    "confidence": str,          # 置信度
    "description": str              # 简短文字描述
}
```

出错时返回 `{"box_visible": False, "error": str}`。

##### 各字段含义

**1. `box_visible`（布尔）**

是否在 RGB 图像中检测到箱子/纸箱。主循环用它判断"还有箱子吗，要不要结束"（见 `box_demo_main.py:423`）。

**2. `box_in_frame`（字符串）— 关键字段**

箱子在画面中的位置状态，取值 5 种（见 `vlm_guide.py:26-31`）：

| 取值 | 含义 | 主程序对应动作 |
|------|------|----------------|
| `"complete"` | 完整可见 | 进入 SAM3 抓取流程 |
| `"left_cut"` | 左侧被裁出画面 | 机器人**左移 6cm** |
| `"right_cut"` | 右侧被裁出画面 | 机器人**右移 6cm** |
| `"both_cut"` | 两侧都被裁（太宽/太近） | **后退 5cm** |
| `"top_cut"` | 上方被裁（机器人太近） | **前移 4cm** |

这段逻辑在 `box_demo_main.py:429-458` 实现了一个侧移对齐循环：箱子不整就持续微调位置，直到 `box_in_frame == "complete"` 才进入下一步 SAM3 抓取。

**3. `move_x_ms` / `move_y_ms`（浮点，单位毫秒）**

VLM 根据深度图估算箱子距离后给出的移动时长建议（机器人速度约 0.2 m/s）：

- `move_x_ms`：正 = 前进，负 = 后退
- `move_y_ms`：正 = 左移，负 = 右移

> 注：主流程里目前主要使用 `box_visible` 和 `box_in_frame` 这两个字段来控制对齐动作；`move_x_ms`/`move_y_ms` 是 VLM 的辅助建议字段。

**4. `confidence`（字符串）**

`"high"` / `"medium"` / `"low"` — VLM 自报的判断置信度。

**5. `description`（字符串）**

英文简短描述，例如 `"A brown cardboard box is fully visible about 1.2m ahead"`。

#### 解析流程

VLM 原始返回是 JSON 文本（可能被包在 markdown 代码块里），`_parse_response` 方法用正则 `\{.*\}` 提取并解析（见 `vlm_guide.py:112-136`），同时做了类型强制转换（`bool` / `float` / `str`）。

#### 实际调用方式

在主程序 `box_demo_main.py:421`：

```python
color, depth = camera.capture()        # 拍照
vlm_result = vlm.query(color, depth)   # 发给 VLM
# 然后判断 vlm_result["box_visible"] 和 vlm_result["box_in_frame"]
```

VLM 的设计目标是**让机器人能视觉对齐箱子**：先用大模型"看清"箱子在画面的什么位置，再通过侧移/前后退让箱子完整呈现在视野中央，最后才交给下游的 SAM3 模型预测具体抓取点。

### SAM3 3D 预测 (`capture_and_predict.py` + `sam3_client.py`)

- 向 SAM3 服务端发送 RGB 图像和深度图（PNG 格式）
- 服务端返回 3D 包围盒结果：
  - `center` — 箱子中心坐标（相机系）
  - `extent` — 箱子尺寸（三轴跨度）
  - `rotation_matrix` — 箱子旋转矩阵
  - `grasp_left` / `grasp_right` — 左/右手抓取点（相机系）
  - `length` — 箱子沿长轴方向长度

### 坐标转换 (`box_demo_main.py`)

将相机光学坐标系下的点转换到 `torso_link` 坐标系：

1. 光学系 → 相机 body 系（固定旋转 `_R_BODY_OPTICAL`）
2. 相机 body 系 → `torso_link`（俯仰旋转 `_R_TORSO_BODY` + 平移 `_T_TRANS`）
3. 抓取点手长补偿：X 退 16cm、Y 偏 5cm、Z 抬 9cm

### 双臂控制 (`dual_arm_target_reach.py`)

- 基于 G1 URDF 的解析逆运动学，精度 < 0.1mm
- 支持多起点重启的数值优化 IK
- 运动分阶段执行：当前位姿 → 零位 → 预备位 → 抓取位 → 闭合 → 抬起 → 释放
- 通过 `rt/arm_sdk` 话题发布上半身关节指令
- **结束行为**（`--end-behavior`）：
  - `handoff_rl_lower`（默认）：抓取完成后先把双臂对齐到 `RL_LOWER_HANDOFF_Q` 姿态（双手自然下垂、肘微曲），再线性淡出 `motor_cmd[29].q`，让 RL 上半身平滑接管，避免猛然释放引发抖动
  - `release`：抓取完成后直接淡出权重槽，立即交权
- **PD 增益分关节配置**：肩俯仰 `KP=150`、肘 `KP=130`、腕 `KP=180`、腰 `KP=250`，让惯量大的关节响应稳，惯量小的关节跟随快

### 行走控制 (`robot_move.py`)

`RobotMover` 是面向上层的 API 封装，把"距离"语义封装成"归一化速度 × 时长"的 IPC 写入序列。

- **IPC 协议**：`/tmp/robojudo_ext_cmd.json`，字段 `fsm` / `velocity{forward,lateral,yaw}` / `timestamp`
- **原子写入**：`tempfile.mkstemp + os.rename`，避免 RL 端读到半截 JSON
- **三种动作**：
  - `move_forward(d)` — 前进/后退；短距离自动抬到最小归一化速度 0.5，避免 RL 抖动
  - `move_left(d)` — 侧移；始终用满归一化速度 1.0，仅靠时长控制距离
  - `rotate(θ)` — 原地转向（默认未启用，主流程用 `WaistRotator` 转腰替代）
- **高频刷新**：行走期间每 100ms 重写 IPC，相当于"持续按住键盘方向键"
- **FSM 切换**：`initialize()` → `RL_FULL`、`shutdown()` → `RL_LOWER`、`release()` → 删除 IPC
- **关键坑**：切到 `RL_LOWER` 后必须保留 IPC 文件，否则 pipeline 回退默认模式引发上肢抽搐

> 详细原理见 [机器人移动机制详解](#机器人移动机制详解)。

### 指令合成 (`merge_lowcmd_arm_sdk.py`)

500Hz 把 `rt/lowcmd_rl` 与 `rt/arm_sdk` 合成为最终 `rt/lowcmd`，是行走与手臂协同的关键仲裁器。

- **基础策略**：默认整段复制 RL（电机 0–34），保证下半身行走不被打断
- **手臂接管**：当 `rt/arm_sdk.motor_cmd[29].q > 1e-3` 且报文新鲜（< 250ms）时，用 arm_sdk 覆盖电机 12–29（腰+双臂+权重槽）
- **新鲜度保护**：RL 报文超 500ms 未更新则停止写 `rt/lowcmd`，arm_sdk 超 250ms 未更新则回退到 RL 上半身
- **CRC 校验**：每帧计算 CRC，确保电机板接受指令
- **线程模型**：DDS 回调线程 + 500Hz `RecurrentThread` 工作线程，通过 `threading.Lock` 交换快照

> 详细原理见 [机器人移动机制详解](#机器人移动机制详解)。

---

## 坐标系定义

### 躯干坐标系 (`torso_link`)

| 轴 | 方向 |
|----|------|
| X | 前（机器人正前方） |
| Y | 左 |
| Z | 上 |

### 手臂可达范围参考

| | X 范围 | Y 范围 | Z 范围 |
|---|--------|--------|--------|
| 左手 | [0.10, 0.55] | [0.00, 0.50] | [-0.10, 0.55] |
| 右手 | [0.10, 0.55] | [-0.50, 0.00] | [-0.10, 0.55] |

理想抓取距离：X = 0.40m（手臂半伸展，重心稳定）

### 相机外参（来自 URDF）

- 安装位置：`[0.0576, 0.0175, 0.4299]` m（相对 `torso_link`）
- 俯仰角：约 47.6°（0.8308 rad）

---

## DDS 话题说明

| 话题 | 方向 | 消息类型 | 发布者 | 说明 |
|------|------|----------|--------|------|
| `rt/lowcmd` | 发布 | `LowCmd_` | merge 脚本 | 最终合成电机指令（500Hz） |
| `rt/lowcmd_rl` | 订阅 | `LowCmd_` | RL Pipeline | RL 行走策略输出 |
| `rt/arm_sdk` | 订阅 | `LowCmd_` | box_demo | 手臂控制指令 |
| `rt/lowstate` | 订阅 | `LowState_` | 机器人 | 全身关节状态反馈 |

**arm_sdk 使能机制：** `motor_cmd[29].q > 0.5` 时，merge 脚本用 `rt/arm_sdk` 覆盖电机 12-29。

---

## 辅助工具

### 调试与测试

| 脚本 | 用途 |
|------|------|
| `zero_position_test.py` | 零位执行测试，验证手臂和腰部运动 |
| `test_arm_sdk_official.py` | 复现官方 C++ 例程的四阶段上肢控制流程 |
| `read_state.py` | 读取并打印全身关节角，可写入 `state.csv` |
| `read_lowcmd.py` | 监听 `rt/lowcmd` 和 `rt/arm_sdk` 话题消息 |
| `compare_cmd_state.py` | 对比指令关节角与实际关节角的误差 |

### 相机与图像

| 脚本 | 用途 |
|------|------|
| `capture_photo.py` | 快速拍照保存为 JPEG |
| `capture_head_rgbd.py` | 采集 RGB+深度图并保存到本地 |
| `snap.py` | 简易拍照保存到 `img/` 目录 |

### 分析与基准

| 脚本 | 用途 |
|------|------|
| `analyze_error.py` | 对比 `goal.csv`（IK 目标）与 `state.csv`（实际位置）的末端误差 |
| `benchmark_policy_freq.py` | 测量 ONNX 行走策略的推理频率 |

---

## 注意事项

### 启动顺序

必须按以下顺序启动，否则 DDS 话题可能冲突：

1. `merge_lowcmd_arm_sdk.py` — 指令合成
2. `run_pipeline.py` — RL 行走策略
3. `box_demo_main.py` — 主抓取流程

### 安全提醒

- 确保机器人周围无障碍物，手臂可自由活动
- 按 `Ctrl+C` 可随时中断抓取流程

### 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| SAM3 预测失败（只有一个面） | 离箱子太近或太远 | 系统会自动调整距离（前移3cm或后退5cm） |
| IK 求解失败 | 抓取点超出工作空间 | 系统会自动计算移动距离并调整位置（含衰减策略） |
| 侧移不准确 | RL 策略侧移有欠追踪 | 使用 `--walk-scale` 补偿（推荐 1.0~1.5） |
| 相机亮度不稳定 | RealSense 自动曝光收敛慢 | 系统自动 50 帧预热 + 2s 冷启动等待 |
| DDS 通信异常 | `libddsc` 版本冲突 | 脚本自动优先加载正确版本 |
| `videohub_pc4` 占用相机 | 宇树自带相机服务 | 脚本自动停止并重启该服务 |

### 数据文件

- `goal.csv` — IK 目标关节数据（转置格式：行=变量，列=采样）
- `state.csv` — 实际关节数据
- `img/` — 采集的图像（按 `attempt*N_color/depth.png` 和 `one.png` 命名）
