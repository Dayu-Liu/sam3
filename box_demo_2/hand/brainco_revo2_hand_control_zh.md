# BrainCo Revo2 灵巧手控制参考

这份文档整理了当前 G1 集成里用于控制 BrainCo/Revo2 灵巧手的本地代码。目标是给别人快速参考：不需要通读整个 SONIC deploy 栈，也能知道应该从哪里发指令、每个通道是什么意思、如何做低风险测试。

## 命令模型

BrainCo 官方 G1 手部服务为每只手暴露一个 DDS 命令 topic 和一个 DDS 状态 topic：

```text
rt/brainco/left/cmd
rt/brainco/left/state
rt/brainco/right/cmd
rt/brainco/right/state
```

每条命令和状态消息里包含 6 个归一化电机通道，顺序固定为：

```text
[thumb, thumb_aux, index, middle, ring, pinky]
```

其中：

```text
thumb     = 大拇指屈伸
thumb_aux = 大拇指辅助轴，通常对应外展/内收/根部方向调整
index     = 食指
middle    = 中指
ring      = 无名指
pinky     = 小指
```

命令值是绝对目标位置，不是相对于当前状态或初始状态的增量：

```text
0.0 = 张开
1.0 = 完全闭合/弯曲
```

例如 `[0.45, 0, 0, 0, 0, 0]` 表示让大拇指 `thumb` 屈伸轴移动到 45% 闭合位置，其余 5 个轴保持张开。`dq` 是速度字段，不是位移。

## SONIC 正式适配器

deploy 侧正式的 BrainCo/Revo2 适配器在：

```text
gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/brainco_hands.hpp
```

关键入口：

- `BrainCoHands::initialize()`：打开左右手的 DDS command/state topics。
- `setNormalizedCommand(is_left, q6)`：发送 6D Revo2 normalized 目标位姿。
- `setAllJointsCommand(is_left, q7)`：接收 SONIC 现有 7D hand buffer，并转换到 Revo2 6D。
- `open(is_left)`：发送 `[0, 0, 0, 0, 0, 0]`。
- `close(is_left)`：发送一个保守的闭合姿态。
- `writeOnce()`：以 50 Hz 发布平滑后的手部命令。

`BrainCoHands` 支持 3 种输入模式：

```text
dex3_proxy  - 将旧的 7D Dex3 命令映射成 Revo2 6D 命令。
normalized  - 直接把 q[0:6] 当作 [thumb, thumb_aux, index, middle, ring, pinky]。
auto        - 当第 7 维为 0 且 q[0:6] 都在 [0, 1] 时自动按 normalized 处理。
```

如果使用生成好的 GRAB/OMOMO Revo2 reference motion 在真机上测试，应该使用：

```bash
--hand-type brainco --brainco-input-mode normalized
```

deploy 主循环里把 hand action 写入当前选择的 hand backend，代码在：

```text
gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp
```

可以搜索这些关键词：

```text
--hand-type <dex3|brainco|none>
--brainco-input-mode <dex3_proxy|normalized|auto>
brainco_hands_.setAllJointsCommand(...)
```

## 使用 SONIC Deploy 控制 Revo2

在 G1 上先启动 BrainCo hand service：

```bash
tmux kill-session -t brainco_hand 2>/dev/null || true
tmux new-session -d -s brainco_hand \
  'cd /home/unitree/brainco_hand_service/bin && sudo ./brainco_hand_server --network_interface enP8p1s0 2>&1 | tee /tmp/brainco_hand_server.log'
sleep 2
tail -80 /tmp/brainco_hand_server.log
```

正常情况下会看到类似绑定信息：

```text
left hand bound to /dev/ttyUSB1 port
right hand bound to /dev/ttyUSB2 port
```

在 deploy 主机上，用 BrainCo backend 启动 SONIC：

```bash
cd /home/wjzh/lizhe/GR00T-WholeBodyControl/gear_sonic_deploy
bash deploy.sh \
  --input-type zmq_manager \
  --output-type zmq \
  --zmq-host localhost \
  --hand-type brainco \
  --brainco-input-mode normalized \
  enp130s0
```

仿真模式也可以使用同样的 hand mode：

```bash
cd /home/wjzh/lizhe/GR00T-WholeBodyControl/gear_sonic_deploy
bash deploy.sh \
  --input-type zmq_manager \
  --output-type zmq \
  --zmq-host localhost \
  --hand-type brainco \
  --brainco-input-mode normalized \
  sim
```

## 独立 DDS 控制示例

这些示例和 SONIC deploy 使用同一条 DDS 控制路径，因此需要 `brainco_hand_server` 已经在 G1 上运行。不要在 SONIC deploy 正在运行时同时运行这些独立 DDS 示例，否则两边都会向 `rt/brainco/*/cmd` 发布命令。

### 张开单只手

源码：

```text
debug_outputs/revo2_assets/send_brainco_open_hands.cpp
```

核心代码：

```cpp
msg.cmds().resize(6);
for (auto& finger : msg.cmds()) {
  finger.mode(0);
  finger.q(0.0f);
  finger.dq(1.0f);
  finger.kp(0.0f);
  finger.kd(0.0f);
  finger.tau(0.0f);
}
```

### 对单个轴做慢速脉冲

源码：

```text
debug_outputs/revo2_assets/send_brainco_thumb_pulse.cpp
```

虽然文件名里有 `thumb`，但这个程序可以测试任意一个轴：

```bash
/tmp/send_brainco_thumb_pulse \
  --side left \
  --axis thumb \
  --amplitude 0.45 \
  --ramp 4.0 \
  --hold 0.7 \
  --hz 10 \
  --dq 0.25 \
  --network_interface enP8p1s0
```

可选轴名：

```text
thumb, thumb_aux, index, middle, ring, pinky
```

在 G1 上编译：

```bash
c++ -std=c++17 -O2 \
  -I/usr/local/include/ddscxx \
  -I/home/unitree/brainco_hand_service/include \
  /tmp/send_brainco_thumb_pulse.cpp \
  -L/home/unitree/brainco_hand_service/lib/aarch64 \
  -Wl,-rpath,/home/unitree/brainco_hand_service/lib/aarch64 \
  -lunitree_sdk2 -lddsc -lddscxx -lrt -lpthread \
  -lboost_program_options -lyaml-cpp -lfmt -lbc_stark_sdk \
  -o /tmp/send_brainco_thumb_pulse
```

## 直连串口 SDK 示例

串口 SDK 路径只建议用于诊断或低层测试。它会绕过 DDS 服务，所以如果需要直接访问串口，先停止 `brainco_hand_server`：

```bash
tmux kill-session -t brainco_hand 2>/dev/null || true
sudo pkill -x brainco_hand_server 2>/dev/null || true
```

串口 SDK 的 6D 顺序同样是：

```text
[thumb, thumb_aux, index, middle, ring, pinky]
```

当前 G1 上常见的设备映射：

```text
left  hand: /dev/ttyUSB1, slave id 126
right hand: /dev/ttyUSB2, slave id 127
baud: 460800
```

### 读取状态

源码：

```text
debug_outputs/revo2_assets/read_brainco_serial_status.cpp
```

它会读取设备信息以及：

```text
positions, speeds, currents, states
```

这些信息适合用来判断某个手指是否进入堵转、保护、限位或异常状态。

### 张开双手

源码：

```text
debug_outputs/revo2_assets/send_brainco_serial_open.cpp
```

核心 SDK 调用：

```cpp
const uint16_t positions[6] = {0, 0, 0, 0, 0, 0};
const uint16_t durations[6] = {millis, millis, millis, millis, millis, millis};
stark_set_finger_positions_and_durations(
    handle, slave_id, positions, durations, 6);
```

### 大拇指双轴探测

源码：

```text
debug_outputs/revo2_assets/probe_left_thumb_pair.cpp
```

它会做三组低层测试：

```text
thumb only
thumb_aux only
thumb + thumb_aux
```

核心单指 SDK 调用：

```cpp
stark_set_finger_position(
    handle,
    slave_id,
    static_cast<StarkFingerId>(i + 1),
    target_position);
```

注意这里是 `i + 1`。这条 Stark SDK 路径里的 finger id 是 1-based。

在 G1 上编译串口诊断工具：

```bash
cd /home/unitree/brainco_hand_service
g++ -std=c++17 -O2 -Iinclude /tmp/read_brainco_serial_status.cpp \
  -Llib/aarch64 -Wl,-rpath,/home/unitree/brainco_hand_service/lib/aarch64 \
  -lbc_stark_sdk -lfmt -lpthread -o /tmp/read_brainco_serial_status
```

## Reference Motion 中的手部 Sidecar

生成好的 SONIC reference motion 会把 Revo2 手部命令存成：

```text
left_hand_joints.csv
right_hand_joints.csv
```

文件是 7 列，因为旧的 SONIC hand path 期待 7D hand buffer。对 Revo2 来说，前 6 列是 BrainCo normalized 命令，第 7 列只是补零：

```text
[thumb, thumb_aux, index, middle, ring, pinky, pad0]
```

GRAB retargeting 转 SONIC reference 的脚本是：

```text
tools/grab_retarget_to_reference_motion.py
```

其中 Revo2 11D 到 BrainCo 6D 的投影关系是：

```text
thumb     = average(thumb_proximal_joint, thumb_distal_joint)
thumb_aux = thumb_metacarpal_joint
index     = average(index_proximal_joint, index_distal_joint)
middle    = average(middle_proximal_joint, middle_distal_joint)
ring      = average(ring_proximal_joint, ring_distal_joint)
pinky     = average(pinky_proximal_joint, pinky_distal_joint)
```

因此，使用这些 reference motion 在真机上回放时，需要使用：

```bash
--brainco-input-mode normalized
```

## 安全检查

- 不要在 SONIC deploy 正在运行时，再单独运行 DDS 控制程序。
- 第一次测试从小幅度开始，比如 `0.2` 到 `0.45`。
- 用慢速 ramp 做人工观察测试。
- 每次测试结束都回到 `[0, 0, 0, 0, 0, 0]`。
- 如果 DDS 命令变化但实体手指不动，优先检查 BrainCo service、串口绑定和电机状态。
- 如果某个轴很早就饱和或卡住，先比较目标命令和 `MotorStates_` 状态，再用串口状态工具检查 current/state。
