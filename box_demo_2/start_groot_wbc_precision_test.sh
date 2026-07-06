#!/bin/bash
# MuJoCo open-loop PRECISION test for the GR00T-WBC box_demo adapter.
#
# Like start_groot_wbc_mujoco_locomotion_test.sh, but:
#   - the adapter logs sim ground-truth floating_base_pose (--log-pose)
#   - pane3 runs measure_precision.py (drives groot_mover through small/medium
#     forward/strafe/yaw targets, then prints a goal-vs-actual table)
#
# Does NOT touch the real robot.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SIM2REAL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SESSION="${SESSION:-g1-groot-wbc-precision}"

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
POSE_CSV="${POSE_CSV:-/tmp/groot_pose.csv}"
MOVES_JSON="${MOVES_JSON:-/tmp/groot_moves.json}"
WARMUP_S="${WARMUP_S:-6}"
SETTLE_S="${SETTLE_S:-1.5}"
NO_ONSCREEN="${NO_ONSCREEN:-0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --groot-repo)    GROOT_REPO="$2";    shift 2 ;;
        --wbc-venv)      WBC_VENV="$2";      shift 2 ;;
        --ros-distro)    ROS_DISTRO="$2";    shift 2 ;;
        --dds-iface)     DDS_IFACE="$2";     shift 2 ;;
        --domain)        DOMAIN="$2";        shift 2 ;;
        --torch-threads) TORCH_THREADS="$2"; shift 2 ;;
        --fwd-max)       FWD_MAX="$2";       shift 2 ;;
        --lat-max)       LAT_MAX="$2";       shift 2 ;;
        --yaw-max)       YAW_MAX="$2";       shift 2 ;;
        --pose-csv)      POSE_CSV="$2";      shift 2 ;;
        --warmup-s)      WARMUP_S="$2";      shift 2 ;;
        --settle-s)      SETTLE_S="$2";      shift 2 ;;
        --no-onscreen)   NO_ONSCREEN=1;      shift ;;
        *) echo "unknown argument: $1"; exit 1 ;;
    esac
done

if [[ ! -d "$GROOT_REPO/decoupled_wbc" ]]; then
    echo "GR00T repo not found: $GROOT_REPO"; exit 1
fi
if [[ ! -f "$WBC_VENV/bin/activate" ]]; then
    echo "WBC venv not found: $WBC_VENV"; exit 1
fi

echo "========================================"
echo "  GR00T-WBC MuJoCo PRECISION test"
echo "  GROOT_REPO: $GROOT_REPO"
echo "  pose CSV:   $POSE_CSV"
echo "  caps:       fwd=$FWD_MAX lat=$LAT_MAX yaw=$YAW_MAX"
echo "========================================"

tmux kill-session -t "$SESSION" 2>/dev/null || true
rm -f "$POSE_CSV"

export OMP_NUM_THREADS="$TORCH_THREADS" MKL_NUM_THREADS="$TORCH_THREADS"
export OPENBLAS_NUM_THREADS="$TORCH_THREADS" NUMEXPR_NUM_THREADS="$TORCH_THREADS"

WBC_SETUP="
    source /opt/ros/$ROS_DISTRO/setup.bash
    source '$WBC_VENV/bin/activate'
    export GROOT_WBC_REPO='$GROOT_REPO'
    export TORCH_THREADS='$TORCH_THREADS'
"

ONSCREEN_ARGS=(--onscreen)
[[ "$NO_ONSCREEN" == "1" ]] && ONSCREEN_ARGS=()

tmux new-session -d -s "$SESSION" -n merge "
    $WBC_SETUP
    cd '$SIM2REAL_DIR'
    echo '=== pane1: sim merge -> rt/lowcmd ==='
    python '$SCRIPT_DIR/merge_lowcmd_arm_sdk.py' --iface '$DDS_IFACE'
    exec bash"

tmux split-window -h -t "$SESSION" "
    $WBC_SETUP
    cd '$SIM2REAL_DIR'
    sleep 1
    echo '=== pane2: GR00T-WBC adapter (sim) + pose logging ==='
    python '$SCRIPT_DIR/groot_wbc_boxdemo_adapter.py' \
        --groot-repo '$GROOT_REPO' \
        --interface sim --domain '$DOMAIN' --cmd-file '$CMD_FILE' \
        --torch-threads '$TORCH_THREADS' \
        --fwd-max '$FWD_MAX' --lat-max '$LAT_MAX' --yaw-max '$YAW_MAX' \
        --height-rate '$HEIGHT_RATE' --shutdown-action limp \
        --log-pose '$POSE_CSV' \
        ${ONSCREEN_ARGS[@]}
    exec bash"

tmux split-window -v -t "$SESSION:0.1" "
    $WBC_SETUP
    cd '$SCRIPT_DIR'
    sleep 5
    echo '=== pane3: precision harness (drives groot_mover, then prints table) ==='
    python '$SCRIPT_DIR/measure_precision.py' \
        --cmd-file '$CMD_FILE' \
        --pose-csv '$POSE_CSV' \
        --save-moves '$MOVES_JSON' \
        --warmup-s '$WARMUP_S' \
        --settle-s '$SETTLE_S'
    echo '[precision harness done; table above]'
    exec bash"

tmux select-layout -t "$SESSION" tiled
echo "tmux session started: $SESSION (pane2 = MuJoCo viewer, pane3 = precision table)"
sleep 1
tmux attach -t "$SESSION"
