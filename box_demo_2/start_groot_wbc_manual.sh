#!/bin/bash
# Minimal GR00T-WBC real-robot manual keyboard bring-up for box_demo_2.
#
# Panes:
#   1) merge_lowcmd_arm_sdk.py        rt/lowcmd_rl + rt/arm_sdk -> rt/lowcmd
#   2) groot_wbc_boxdemo_adapter.py   GR00T-WBC -> rt/lowcmd_rl
#   3) agile_keyboard_control.py      WASD/QE/Z/X/C/R IPC control

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SIM2REAL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SESSION="${SESSION:-g1-groot-wbc-manual}"

IFACE="${IFACE:-${UNITREE_DDS_INTERFACE:-enp130s0}}"
GROOT_REPO="${GROOT_REPO:-$HOME/GR00T-WholeBodyControl}"
WBC_VENV="${WBC_VENV:-$GROOT_REPO/.venv_wbc}"
ROS_DISTRO="${ROS_DISTRO:-humble}"
TORCH_THREADS="${TORCH_THREADS:-1}"
FWD_MAX="${FWD_MAX:-0.30}"
LAT_MAX="${LAT_MAX:-0.15}"
YAW_MAX="${YAW_MAX:-0.30}"
HEIGHT_RATE="${HEIGHT_RATE:-0.10}"
KEY_VX="${KEY_VX:-0.12}"
KEY_VY="${KEY_VY:-0.08}"
KEY_WZ="${KEY_WZ:-0.10}"
KEY_TIMEOUT="${KEY_TIMEOUT:-0.25}"
HEIGHT_STEP="${HEIGHT_STEP:-0.02}"
DRY_RUN=""
AUTO_POLICY_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --iface)        IFACE="$2";        shift 2 ;;
        --groot-repo)   GROOT_REPO="$2";   shift 2 ;;
        --wbc-venv)     WBC_VENV="$2";     shift 2 ;;
        --ros-distro)   ROS_DISTRO="$2";   shift 2 ;;
        --torch-threads) TORCH_THREADS="$2"; shift 2 ;;
        --fwd-max)      FWD_MAX="$2";      shift 2 ;;
        --lat-max)      LAT_MAX="$2";      shift 2 ;;
        --yaw-max)      YAW_MAX="$2";      shift 2 ;;
        --height-rate)  HEIGHT_RATE="$2";  shift 2 ;;
        --key-vx)       KEY_VX="$2";       shift 2 ;;
        --key-vy)       KEY_VY="$2";       shift 2 ;;
        --key-wz)       KEY_WZ="$2";       shift 2 ;;
        --key-timeout)  KEY_TIMEOUT="$2";  shift 2 ;;
        --height-step)  HEIGHT_STEP="$2";  shift 2 ;;
        --dry-run)      DRY_RUN="--dry-run"; shift ;;
        --no-auto-activate-policy) AUTO_POLICY_ARGS="--no-auto-activate-policy"; shift ;;
        *) echo "unknown argument: $1"; exit 1 ;;
    esac
done

if [[ ! -d "$GROOT_REPO/decoupled_wbc" ]]; then
    echo "GR00T repo not found: $GROOT_REPO"
    exit 1
fi
if [[ ! -f "$WBC_VENV/bin/activate" ]]; then
    echo "WBC venv not found: $WBC_VENV"
    exit 1
fi

echo "========================================"
echo "  G1 GR00T-WBC manual keyboard control"
echo "  iface:        $IFACE"
echo "  GROOT_REPO:   $GROOT_REPO"
echo "  WBC_VENV:     $WBC_VENV"
echo "  ROS_DISTRO:   $ROS_DISTRO"
echo "  torch threads:$TORCH_THREADS"
echo "  limits:       fwd=$FWD_MAX lat=$LAT_MAX yaw=$YAW_MAX height_rate=$HEIGHT_RATE"
echo "  keyboard:     vx=$KEY_VX vy=$KEY_VY wz=$KEY_WZ key_timeout=$KEY_TIMEOUT"
echo "  dry-run:      ${DRY_RUN:-no}"
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

tmux split-window -v -t "$SESSION:0.1" "
    $WBC_SETUP
    cd '$SIM2REAL_DIR'
    sleep 2
    echo '=== pane3: keyboard IPC control ==='
    echo \"python: \$(which python)\"
    echo 'focus this pane: w/s/a/d/q/e, z/x height, c crouch, r stand, space stop, o damp'
    python '$SCRIPT_DIR/agile_keyboard_control.py' \
        --vx '$KEY_VX' \
        --vy '$KEY_VY' \
        --wz '$KEY_WZ' \
        --height-step '$HEIGHT_STEP' \
        --key-timeout '$KEY_TIMEOUT'
    echo '[keyboard exited]'
    exec bash"

tmux select-layout -t "$SESSION" tiled

echo ""
echo "tmux session started: $SESSION"
echo "  Ctrl+B arrows: switch panes"
echo "  Ctrl+B D: detach"
echo "  tmux attach -t $SESSION: reattach"
echo "  focus keyboard pane for manual control"
echo ""

sleep 1
tmux attach -t "$SESSION"
