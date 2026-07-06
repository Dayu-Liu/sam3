#!/bin/bash
# MuJoCo-only locomotion test for GR00T-WBC box_demo adapter.
#
# This does not touch the real robot. It starts:
#   1) merge_lowcmd_arm_sdk.py
#   2) groot_wbc_boxdemo_adapter.py --interface sim --onscreen
#   3) decision_ipc_locomotion_demo.py

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SIM2REAL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SESSION="${SESSION:-g1-groot-wbc-mujoco-loco}"

GROOT_REPO="${GROOT_REPO:-$HOME/GR00T-WholeBodyControl}"
WBC_VENV="${WBC_VENV:-$GROOT_REPO/.venv_wbc}"
ROS_DISTRO="${ROS_DISTRO:-humble}"
DDS_IFACE="${DDS_IFACE:-lo}"
DOMAIN="${DOMAIN:-0}"
TORCH_THREADS="${TORCH_THREADS:-1}"
FWD_MAX="${FWD_MAX:-0.50}"
LAT_MAX="${LAT_MAX:-0.30}"
YAW_MAX="${YAW_MAX:-0.60}"
HEIGHT_RATE="${HEIGHT_RATE:-0.20}"
CMD_FILE="${CMD_FILE:-/tmp/robojudo_ext_cmd.json}"
STATE_DIR="${STATE_DIR:-/tmp/agile_sim2sim}"
NO_ONSCREEN="${NO_ONSCREEN:-0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --groot-repo)   GROOT_REPO="$2";   shift 2 ;;
        --wbc-venv)     WBC_VENV="$2";     shift 2 ;;
        --ros-distro)   ROS_DISTRO="$2";   shift 2 ;;
        --dds-iface)    DDS_IFACE="$2";    shift 2 ;;
        --domain)       DOMAIN="$2";       shift 2 ;;
        --torch-threads) TORCH_THREADS="$2"; shift 2 ;;
        --fwd-max)      FWD_MAX="$2";      shift 2 ;;
        --lat-max)      LAT_MAX="$2";      shift 2 ;;
        --yaw-max)      YAW_MAX="$2";      shift 2 ;;
        --height-rate)  HEIGHT_RATE="$2";  shift 2 ;;
        --cmd-file)     CMD_FILE="$2";     shift 2 ;;
        --state-dir)    STATE_DIR="$2";    shift 2 ;;
        --no-onscreen)  NO_ONSCREEN=1;     shift ;;
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
echo "  GR00T-WBC MuJoCo locomotion-only test"
echo "  GROOT_REPO:    $GROOT_REPO"
echo "  WBC_VENV:      $WBC_VENV"
echo "  ROS_DISTRO:    $ROS_DISTRO"
echo "  DDS interface: $DDS_IFACE"
echo "  DOMAIN:        $DOMAIN"
echo "  IPC legacy:    $CMD_FILE"
echo "  IPC sim dir:   $STATE_DIR"
echo "  fwd/lat/yaw:   $FWD_MAX / $LAT_MAX / $YAW_MAX"
echo "  onscreen:      $([[ "$NO_ONSCREEN" == "1" ]] && echo no || echo yes)"
echo "========================================"

if pgrep -f "merge_lowcmd_arm_sdk.py|groot_wbc_boxdemo_adapter.py|run_g1_control_loop.py" >/dev/null 2>&1; then
    echo "[WARN] Existing GR00T/lowcmd process detected."
    echo "       Stop old sim/control processes if topics conflict."
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
    export TORCH_THREADS='$TORCH_THREADS'
"

ONSCREEN_ARGS=(--onscreen)
if [[ "$NO_ONSCREEN" == "1" ]]; then
    ONSCREEN_ARGS=()
fi

tmux new-session -d -s "$SESSION" -n merge "
    $WBC_SETUP
    cd '$SIM2REAL_DIR'
    echo '=== pane1: sim merge rt/lowcmd_rl + rt/arm_sdk -> rt/lowcmd ==='
    echo \"python: \$(which python)\"
    python '$SCRIPT_DIR/merge_lowcmd_arm_sdk.py' --iface '$DDS_IFACE'
    echo '[merge exited]'
    exec bash"

tmux split-window -h -t "$SESSION" "
    $WBC_SETUP
    cd '$SIM2REAL_DIR'
    sleep 1
    echo '=== pane2: GR00T-WBC adapter with MuJoCo viewer ==='
    echo \"python: \$(which python)\"
    python '$SCRIPT_DIR/groot_wbc_boxdemo_adapter.py' \
        --groot-repo '$GROOT_REPO' \
        --interface sim \
        --domain '$DOMAIN' \
        --cmd-file '$CMD_FILE' \
        --torch-threads '$TORCH_THREADS' \
        --fwd-max '$FWD_MAX' \
        --lat-max '$LAT_MAX' \
        --yaw-max '$YAW_MAX' \
        --height-rate '$HEIGHT_RATE' \
        --shutdown-action limp \
        ${ONSCREEN_ARGS[@]}
    echo '[GR00T-WBC adapter exited]'
    exec bash"

tmux split-window -v -t "$SESSION:0.1" "
    $WBC_SETUP
    cd '$SIM2REAL_DIR'
    sleep 4
    echo '=== pane3: numeric decision IPC locomotion demo ==='
    echo 'Examples: 1 1.0 | 3 0.4 | 5 90 | 7 0.55 | 0'
    echo 'During an active motion, press 0 to interrupt immediately and write zero velocity.'
    python '$SCRIPT_DIR/decision_ipc_locomotion_demo.py' \
        --cmd-file '$CMD_FILE' \
        --state-dir '$STATE_DIR' \
        --write both \
        --fwd-max '$FWD_MAX' \
        --lat-max '$LAT_MAX' \
        --yaw-max '$YAW_MAX' \
        --ctrl-c-action limp
    echo '[decision demo exited]'
    exec bash"

tmux select-layout -t "$SESSION" tiled

echo ""
echo "tmux session started: $SESSION"
echo "  pane2 should open the MuJoCo viewer"
echo "  focus pane3 and type numeric commands:"
echo "    1 1.0   forward 1m"
echo "    3 0.4   left 0.4m"
echo "    5 90    yaw left 90deg"
echo "    7 0.55  set height"
echo "    0       stop / interrupt active motion"
echo ""

sleep 1
tmux attach -t "$SESSION"
