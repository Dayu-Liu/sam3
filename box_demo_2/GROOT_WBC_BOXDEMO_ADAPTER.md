# GR00T-WBC Adapter for box_demo_2

这个目录在不改动原始 box demo 代码的前提下，新增了一个 GR00T-WBC 下半身适配器。

## 为什么需要这个 adapter

官方 GR00T-WBC 脚本：

```bash
python decoupled_wbc/control/main/teleop/run_g1_control_loop.py --interface real
```

会直接把最终电机命令发布到：

```text
rt/lowcmd
```

但 box demo 需要保留一层合流流程：

```text
rt/lowcmd_rl + rt/arm_sdk -> merge_lowcmd_arm_sdk.py -> rt/lowcmd
```

所以 `groot_wbc_boxdemo_adapter.py` 复用了 GR00T-WBC 的观测和 policy 逻辑，但只发布到：

```text
rt/lowcmd_rl
```

这样可以保持 `box_demo_main.py`、IK 和 `rt/arm_sdk` 的手臂覆盖逻辑不变。

## 新增文件

- `groot_wbc_boxdemo_adapter.py`
  - 读取 `/tmp/robojudo_ext_cmd.json`。
  - 调用 GR00T Decoupled WBC。
  - 发布 `rt/lowcmd_rl`。
  - 将 motor command slot 29 保持为 `q=0`，避免 RL 流误请求手臂控制权。

- `start_groot_wbc_manual.sh`
  - 启动 merger、GR00T-WBC adapter 和键盘 IPC 控制。

- `start_groot_wbc_box.sh`
  - 启动 merger、GR00T-WBC adapter、键盘 IPC 控制和 `box_demo_main.py`。

- `decision_ipc_locomotion_demo.py`
  - 只测试 locomotion 的决策层模拟器。
  - 可以输入类似 `1 1.0` 的数字命令，表示前进 1 米。
  - 会把距离/角度请求转换成速度和持续时间，并持续刷新 IPC。

- `start_groot_wbc_mujoco_locomotion_test.sh`
  - 只用于 MuJoCo 的 locomotion 测试启动器。
  - 会启动 merge、sim 模式下的 GR00T-WBC adapter，以及数字决策模拟器。

- `start_groot_wbc_real_locomotion_test.sh`
  - 只用于真机 locomotion 的数字输入测试启动器。
  - 不启动 `box_demo_main.py`，不启动相机、VLM、SAM3 或上半身规划。
  - 默认速度和 height 范围比 MuJoCo 测试更保守。

## MuJoCo locomotion-only test

这条路线不会运行真机，也不会启动 `box_demo_main.py`。

```bash
cd ~/agile_boxdeploy/box_demo_2
bash start_groot_wbc_mujoco_locomotion_test.sh \
  --groot-repo ~/GR00T-WholeBodyControl \
  --wbc-venv ~/GR00T-WholeBodyControl/.venv_wbc \
  --ros-distro humble
```

在数字决策输入窗口中：

```text
1 1.0       前进 1.0 m
2 0.5       后退 0.5 m
3 0.4       左平移 0.4 m
4 0.4       右平移 0.4 m
5 90        左转 90 deg
6 90        右转 90 deg
7 0.55      将 base height 设置为 0.55 m
8           恢复站立高度
9           下蹲预设
0           停止；运动执行过程中按 0 会立即中断当前速度输入并写入零速度
Ctrl+C      写入 LIMP，让 adapter 释放电机刚度，进入瘫软/低刚度退出状态
demo        运行一个短的预设测试序列
```

这个模拟器会同时写入：

```text
/tmp/robojudo_ext_cmd.json
/tmp/agile_sim2sim/command.json
```

对于 GR00T-WBC adapter 路线，真正被使用的 IPC 是 `/tmp/robojudo_ext_cmd.json`。
同时写入 sim command file，是为了让同一个模拟器也能复用于现有的 AGILE sim2sim 流程。

## Real locomotion-only numeric test

这条路线会运行真机底层控制，但不会启动 `box_demo_main.py`，也不会启动上半身大脑。

第一次在真机主机上建议先跑 dry-run：

```bash
cd ~/agile_boxdeploy/box_demo_2
bash start_groot_wbc_real_locomotion_test.sh \
  --iface enp130s0 \
  --groot-repo ~/GR00T-WholeBodyControl \
  --wbc-venv ~/GR00T-WholeBodyControl/.venv_wbc \
  --ros-distro humble \
  --dry-run
```

dry-run 通过后，再在有人保护、机器人吊带/安全空间准备好的情况下去掉 `--dry-run`：

```bash
bash start_groot_wbc_real_locomotion_test.sh \
  --iface enp130s0 \
  --groot-repo ~/GR00T-WholeBodyControl \
  --wbc-venv ~/GR00T-WholeBodyControl/.venv_wbc \
  --ros-distro humble
```

真机数字输入窗口建议先从很小的命令开始：

```text
1 0.2       前进 0.2 m
2 0.2       后退 0.2 m
3 0.1       左平移 0.1 m
4 0.1       右平移 0.1 m
5 15        左转 15 deg
6 15        右转 15 deg
7 0.30      将 base height 设置为 0.30 m
8           恢复站立高度
0           停止；运动执行过程中按 0 会立即中断当前速度输入并写入零速度
Ctrl+C      写入 LIMP，让 adapter 释放电机刚度，进入瘫软/低刚度退出状态
```

真机脚本默认参数更保守：

```text
FWD_MAX=0.30
LAT_MAX=0.15
YAW_MAX=0.30
FWD_SPEED=0.12
LAT_SPEED=0.08
YAW_RATE=0.15
MIN_DURATION=0.80
MIN_HEIGHT=0.20
MAX_HEIGHT=0.74
CROUCH_HEIGHT=0.40
```

如需修改，可以在命令行上传参，例如：

```bash
bash start_groot_wbc_real_locomotion_test.sh \
  --iface enp130s0 \
  --fwd-speed 0.10 \
  --fwd-max 0.25 \
  --min-height 0.55 \
  --crouch-height 0.65
```

## Manual bring-up

在完整 box demo 测试前，建议先运行这个手动测试：

```bash
cd ~/agile_boxdeploy/box_demo_2
bash start_groot_wbc_manual.sh \
  --iface enp130s0 \
  --groot-repo ~/GR00T-WholeBodyControl \
  --wbc-venv ~/GR00T-WholeBodyControl/.venv_wbc \
  --ros-distro humble
```

Dry-run 检查：

```bash
bash start_groot_wbc_manual.sh \
  --iface enp130s0 \
  --groot-repo ~/GR00T-WholeBodyControl \
  --wbc-venv ~/GR00T-WholeBodyControl/.venv_wbc \
  --ros-distro humble \
  --dry-run
```

在键盘控制窗口中：

```text
w/s: 前进/后退
a/d: 左移/右移
q/e: yaw 旋转
z/x: 降低/升高 base height
c: 下蹲预设
r: 恢复站立高度
space: 停止
o: 阻尼请求
```

## Full box demo

```bash
cd ~/agile_boxdeploy/box_demo_2
bash start_groot_wbc_box.sh \
  --iface enP8p1s0 \
  --groot-repo ~/GR00T-WholeBodyControl \
  --wbc-venv ~/GR00T-WholeBodyControl/.venv_wbc \
  --ros-distro humble \
  --box-conda-env hdmi \
  --vlm-endpoint https://your-vlm-endpoint/v1 \
  --sam3-host 127.0.0.1 \
  --sam3-port 5300
```

## Important notes

- 不要同时运行 `run_g1_control_loop.py` 和这个 adapter。否则会出现多个进程同时写最终 `rt/lowcmd` 的情况。

- 在非 dry-run 的实机测试中，需要保持 `merge_lowcmd_arm_sdk.py` 运行。这个 adapter 有意只发布 `rt/lowcmd_rl`，而不是最终的 `rt/lowcmd`。

- 现有 `RobotMover` 仍然会把 box demo 的位置请求转换成速度和持续时间，并写入 `/tmp/robojudo_ext_cmd.json`。这个 adapter 会消费这些不断变化的速度命令。

- 抓取阶段，`rt/arm_sdk` 会通过 merger 覆盖 motors 12..29。GR00T-WBC 会继续提供下半身平衡命令。

- 在新主机上第一次测试时，建议先跑 `--dry-run`，用于检查 import、ROS2 Humble 环境和 `rt/lowstate` 是否可用。dry-run 不会发布电机命令。

- `0` 和 `Ctrl+C` 的语义不同：`0` 只中断当前速度输入并写零速度，adapter 仍会继续站立/平衡；`Ctrl+C` 会写入 `LIMP`，adapter 收到后发布 `kp=0, kd=0, tau=0` 的低层命令，用于让机器人退出到瘫软/低刚度状态。
