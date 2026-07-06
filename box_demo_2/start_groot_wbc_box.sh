#!/bin/bash
# G1 box_demo_2 + GR00T-WBC lower-body launcher.
#
# Panes:
#   1) merge_lowcmd_arm_sdk.py        rt/lowcmd_rl + rt/arm_sdk -> rt/lowcmd
#   2) groot_wbc_boxdemo_adapter.py   GR00T-WBC -> rt/lowcmd_rl
#   3) agile_keyboard_control.py      optional WASD/QE/Z/X/C/R IPC teleop
#   4) box_demo_main.py               camera/VLM/SAM3/IK/arm_sdk grasp flow

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SIM2REAL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SESSION="${SESSION:-g1-groot-wbc-box}"

# VLM 默认开启（千问 Qwen，国内直连，不走代理；key 从 ~/.bashrc 的 QWEN_API_KEY 读）。
# 加 --no-vlm 显式关闭 → 仅用 SAM3 检测（箱子需大致在正前方）。
VLM_ENDPOINT="${VLM_ENDPOINT:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
NO_VLM=0
SAM3_HOST="${SAM3_HOST:-127.0.0.1}"
SAM3_PORT="${SAM3_PORT:-5300}"
IFACE="${IFACE:-${UNITREE_DDS_INTERFACE:-enP8p1s0}}"
GROOT_REPO="${GROOT_REPO:-$HOME/GR00T-WholeBodyControl}"
WBC_VENV="${WBC_VENV:-$GROOT_REPO/.venv_wbc}"
ROS_DISTRO="${ROS_DISTRO:-humble}"
BOX_CONDA_ENV="${BOX_CONDA_ENV:-hdmi}"
TORCH_THREADS="${TORCH_THREADS:-1}"
WALK_SCALE="${WALK_SCALE:-1.0}"
FWD_MAX="${FWD_MAX:-0.50}"
LAT_MAX="${LAT_MAX:-0.30}"
YAW_MAX="${YAW_MAX:-0.60}"
HEIGHT_RATE="${HEIGHT_RATE:-0.20}"
NO_CONFIRM="--no-confirm"
DRY_RUN=""
AUTO_POLICY_ARGS=""
WITH_KEYBOARD=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --vlm-endpoint) VLM_ENDPOINT="$2"; shift 2 ;;
        --no-vlm)       NO_VLM=1;          shift ;;
        --sam3-host)    SAM3_HOST="$2";    shift 2 ;;
        --sam3-port)    SAM3_PORT="$2";    shift 2 ;;
        --iface)        IFACE="$2";        shift 2 ;;
        --groot-repo)   GROOT_REPO="$2";   shift 2 ;;
        --wbc-venv)     WBC_VENV="$2";     shift 2 ;;
        --ros-distro)   ROS_DISTRO="$2";   shift 2 ;;
        --box-conda-env) BOX_CONDA_ENV="$2"; shift 2 ;;
        --torch-threads) TORCH_THREADS="$2"; shift 2 ;;
        --walk-scale)   WALK_SCALE="$2";   shift 2 ;;
        --fwd-max)      FWD_MAX="$2";      shift 2 ;;
        --lat-max)      LAT_MAX="$2";      shift 2 ;;
        --yaw-max)      YAW_MAX="$2";      shift 2 ;;
        --height-rate)  HEIGHT_RATE="$2";  shift 2 ;;
        --confirm)      NO_CONFIRM="";     shift ;;
        --dry-run)      DRY_RUN="--dry-run"; shift ;;
        --no-auto-activate-policy) AUTO_POLICY_ARGS="--no-auto-activate-policy"; shift ;;
        --no-keyboard)  WITH_KEYBOARD=0;   shift ;;
        *) echo "unknown argument: $1"; exit 1 ;;
    esac
done

# VLM on by default (full pipeline). --no-vlm => SAM3-only (box must be in front).
if [[ "$NO_VLM" == "1" ]]; then
    VLM_ARGS="--no-vlm"
    VLM_DESC="disabled (SAM3-only)"
else
    VLM_ARGS="--vlm-endpoint $VLM_ENDPOINT"
    VLM_DESC="$VLM_ENDPOINT"
fi
if [[ ! -d "$GROOT_REPO/decoupled_wbc" ]]; then
    echo "GR00T repo not found: $GROOT_REPO"
    exit 1
fi
if [[ ! -f "$WBC_VENV/bin/activate" ]]; then
    echo "WBC venv not found: $WBC_VENV"
    exit 1
fi

echo "========================================"
echo "  G1 box_demo_2 + GR00T-WBC lower body"
echo "  iface:      $IFACE"
echo "  VLM:        $VLM_DESC"
echo "  SAM3:       $SAM3_HOST:$SAM3_PORT"
echo "  GROOT_REPO: $GROOT_REPO"
echo "  WBC_VENV:   $WBC_VENV"
echo "  ROS_DISTRO: $ROS_DISTRO"
echo "  box env:    $BOX_CONDA_ENV"
echo "  torch threads: $TORCH_THREADS"
echo "  walk-scale: $WALK_SCALE"
echo "  fwd/lat/yaw max: $FWD_MAX / $LAT_MAX / $YAW_MAX"
echo "  height-rate: $HEIGHT_RATE"
echo "  dry-run:    ${DRY_RUN:-no}"
echo "========================================"

if pgrep -f "merge_lowcmd_arm_sdk.py|groot_wbc_boxdemo_adapter.py|run_g1_control_loop.py|agile_lowcmd_pipeline.py" >/dev/null 2>&1; then
    echo "[WARN] Existing low-level control process detected."
    echo "       Stop old control processes before live hardware control."
fi

tmux kill-session -t "$SESSION" 2>/dev/null || true

export OMP_NUM_THREADS="$TORCH_THREADS"
export MKL_NUM_THREADS="$TORCH_THREADS"
export OPENBLAS_NUM_THREADS="$TORCH_THREADS"
export NUMEXPR_NUM_THREADS="$TORCH_THREADS"
export VECLIB_MAXIMUM_THREADS="$TORCH_THREADS"

WBC_SETUP="
    source /opt/ros/$ROS_DISTRO/setup.bash
    source '$WBC_VENV/bin/activate'
    export GROOT_WBC_REPO='$GROOT_REPO'
    export UNITREE_DDS_INTERFACE='$IFACE'
    export TORCH_THREADS='$TORCH_THREADS'
"

# 注意: 用绝对路径的 hdmi python(BOX_PY)启动,并 unset 继承来的 venv —— 否则若从
# 已激活某个 python venv(如 g1_deploy)的 shell 里启动,tmux pane 会继承该 venv,
# conda activate 盖不掉它,导致 box_demo 用错 python 报 'No module named cv2'。
BOX_SETUP="
    unset VIRTUAL_ENV
    source '$HOME/miniconda3/etc/profile.d/conda.sh'
    conda activate '$BOX_CONDA_ENV'
    export UNITREE_DDS_INTERFACE='$IFACE'
    BOX_PY='$HOME/miniconda3/envs/$BOX_CONDA_ENV/bin/python'
"

if [[ -n "$DRY_RUN" ]]; then
    tmux new-session -d -s "$SESSION" -n dryrun "
        echo '=== pane1: dry-run ==='
        echo 'merger is not started, so this run does not publish rt/lowcmd'
        exec bash"
else
    tmux new-session -d -s "$SESSION" -n merge "
        $WBC_SETUP
        cd '$SIM2REAL_DIR'
        echo '=== pane1: merge rt/lowcmd_rl + rt/arm_sdk -> rt/lowcmd ==='
        echo \"python: \$(which python)\"
        python '$SCRIPT_DIR/merge_lowcmd_arm_sdk.py' --iface '$IFACE'
        echo '[merge exited]'
        exec bash"
fi

tmux split-window -h -t "$SESSION" "
    $WBC_SETUP
    cd '$SIM2REAL_DIR'
    sleep 1
    echo '=== pane2: GR00T-WBC adapter -> rt/lowcmd_rl ==='
    echo \"python: \$(which python)\"
    python '$SCRIPT_DIR/groot_wbc_boxdemo_adapter.py' \
        --groot-repo '$GROOT_REPO' \
        --interface '$IFACE' \
        --torch-threads '$TORCH_THREADS' \
        --fwd-max '$FWD_MAX' \
        --lat-max '$LAT_MAX' \
        --yaw-max '$YAW_MAX' \
        --height-rate '$HEIGHT_RATE' \
        $AUTO_POLICY_ARGS \
        $DRY_RUN
    echo '[GR00T-WBC adapter exited]'
    exec bash"

if [[ "$WITH_KEYBOARD" == "1" ]]; then
    tmux split-window -v -t "$SESSION:0.1" "
        $BOX_SETUP
        cd '$SIM2REAL_DIR'
        sleep 2
        echo '=== pane3: keyboard IPC control ==='
        echo \"python: \$BOX_PY\"
        echo 'focus this pane for WASD/QE, z/x height, c pick-height, r stand'
        \"\$BOX_PY\" '$SCRIPT_DIR/agile_keyboard_control.py'
        echo '[keyboard exited]'
        exec bash"
fi

tmux split-window -v -t "$SESSION:0.0" "
    $BOX_SETUP
    cd '$SIM2REAL_DIR'
    sleep 3
    echo '=== pane4: box_demo_main camera/VLM/SAM3/IK/arm_sdk ==='
    echo \"python: \$BOX_PY\"
    \"\$BOX_PY\" '$SCRIPT_DIR/box_demo_main.py' \
        $VLM_ARGS \
        --host '$SAM3_HOST' \
        --port '$SAM3_PORT' \
        --iface '$IFACE' \
        --walk-scale '$WALK_SCALE' \
        --locomotion groot \
        $NO_CONFIRM
    echo '[box_demo exited]'
    exec bash"

tmux select-layout -t "$SESSION" tiled

echo ""
echo "tmux session started: $SESSION"
echo "  Ctrl+B arrows: switch panes"
echo "  Ctrl+B D: detach"
echo "  tmux attach -t $SESSION: reattach"
echo "  keyboard pane: w/s/a/d/q/e, z/x, c, r, space, l/f"
echo ""

sleep 1
tmux attach -t "$SESSION"
