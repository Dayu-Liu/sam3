#!/bin/bash
# Real-robot locomotion-only numeric IPC test for GR00T-WBC box_demo adapter.
#
# No box_demo_main.py, no camera/VLM/SAM3, no upper-body planner.
# Starts:
#   1) merge_lowcmd_arm_sdk.py        rt/lowcmd_rl + rt/arm_sdk -> rt/lowcmd
#   2) groot_wbc_boxdemo_adapter.py   GR00T-WBC -> rt/lowcmd_rl
#   3) decision_ipc_locomotion_demo.py numeric distance/angle/height commands

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SIM2REAL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SESSION="${SESSION:-g1-groot-wbc-real-loco}"

IFACE="${IFACE:-${UNITREE_DDS_INTERFACE:-enp130s0}}"
DOMAIN="${DOMAIN:-0}"
GROOT_REPO="${GROOT_REPO:-$HOME/GR00T-WholeBodyControl}"
WBC_VENV="${WBC_VENV:-$GROOT_REPO/.venv_wbc}"
ROS_DISTRO="${ROS_DISTRO:-humble}"
TORCH_THREADS="${TORCH_THREADS:-1}"

# Conservative live defaults. Increase only after supervised small-step tests.
FWD_MAX="${FWD_MAX:-2.00}"
LAT_MAX="${LAT_MAX:-0.5}"
YAW_MAX="${YAW_MAX:-0.90}"
FWD_SPEED="${FWD_SPEED:-0.40}"
LAT_SPEED="${LAT_SPEED:-0.15}"
YAW_RATE="${YAW_RATE:-0.15}"
MIN_DURATION="${MIN_DURATION:-0.80}"
HEIGHT_RATE="${HEIGHT_RATE:-0.08}"
STAND_HEIGHT="${STAND_HEIGHT:-0.74}"
MIN_HEIGHT="${MIN_HEIGHT:-0.20}"
MAX_HEIGHT="${MAX_HEIGHT:-0.74}"
CROUCH_HEIGHT="${CROUCH_HEIGHT:-0.40}"
CMD_FILE="${CMD_FILE:-/tmp/robojudo_ext_cmd.json}"
DRY_RUN=""
AUTO_POLICY_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --iface)        IFACE="$2";        shift 2 ;;
        --domain)       DOMAIN="$2";       shift 2 ;;
        --groot-repo)   GROOT_REPO="$2";   shift 2 ;;
        --wbc-venv)     WBC_VENV="$2";     shift 2 ;;
        --ros-distro)   ROS_DISTRO="$2";   shift 2 ;;
        --torch-threads) TORCH_THREADS="$2"; shift 2 ;;
        --fwd-max)      FWD_MAX="$2";      shift 2 ;;
        --lat-max)      LAT_MAX="$2";      shift 2 ;;
        --yaw-max)      YAW_MAX="$2";      shift 2 ;;
        --fwd-speed)    FWD_SPEED="$2";    shift 2 ;;
        --lat-speed)    LAT_SPEED="$2";    shift 2 ;;
        --yaw-rate)     YAW_RATE="$2";     shift 2 ;;
        --min-duration) MIN_DURATION="$2"; shift 2 ;;
        --height-rate)  HEIGHT_RATE="$2";  shift 2 ;;
        --stand-height) STAND_HEIGHT="$2"; shift 2 ;;
        --min-height)   MIN_HEIGHT="$2";   shift 2 ;;
        --max-height)   MAX_HEIGHT="$2";   shift 2 ;;
        --crouch-height) CROUCH_HEIGHT="$2"; shift 2 ;;
        --cmd-file)     CMD_FILE="$2";     shift 2 ;;
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
echo "  GR00T-WBC real locomotion-only numeric test"
echo "  iface/domain: $IFACE / $DOMAIN"
echo "  GROOT_REPO:   $GROOT_REPO"
echo "  WBC_VENV:     $WBC_VENV"
echo "  ROS_DISTRO:   $ROS_DISTRO"
echo "  IPC:          $CMD_FILE"
echo "  limits:       fwd=$FWD_MAX lat=$LAT_MAX yaw=$YAW_MAX"
echo "  default cmd:  fwd=$FWD_SPEED lat=$LAT_SPEED yaw=$YAW_RATE min_duration=$MIN_DURATION"
echo "  height:       stand=$STAND_HEIGHT min=$MIN_HEIGHT max=$MAX_HEIGHT crouch=$CROUCH_HEIGHT rate=$HEIGHT_RATE"
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
        --domain '$DOMAIN' \
        --cmd-file '$CMD_FILE' \
        --torch-threads '$TORCH_THREADS' \
        --fwd-max '$FWD_MAX' \
        --lat-max '$LAT_MAX' \
        --yaw-max '$YAW_MAX' \
        --height-rate '$HEIGHT_RATE' \
        --stand-height '$STAND_HEIGHT' \
        --min-height '$MIN_HEIGHT' \
        --max-height '$MAX_HEIGHT' \
        --shutdown-action limp \
        $AUTO_POLICY_ARGS \
        $DRY_RUN
    echo '[GR00T-WBC adapter exited]'
    exec bash"

tmux split-window -v -t "$SESSION:0.1" "
    $WBC_SETUP
    cd '$SIM2REAL_DIR'
    sleep 4
    echo '=== pane3: numeric decision IPC locomotion demo ==='
    echo 'Start with small commands: 1 0.2 | 3 0.1 | 5 15 | 0'
    echo 'During an active motion, press 0 to interrupt immediately and write zero velocity.'
    python '$SCRIPT_DIR/decision_ipc_locomotion_demo.py' \
        --cmd-file '$CMD_FILE' \
        --write legacy \
        --fwd-speed '$FWD_SPEED' \
        --lat-speed '$LAT_SPEED' \
        --yaw-rate '$YAW_RATE' \
        --fwd-max '$FWD_MAX' \
        --lat-max '$LAT_MAX' \
        --yaw-max '$YAW_MAX' \
        --min-duration '$MIN_DURATION' \
        --height '$STAND_HEIGHT' \
        --stand-height '$STAND_HEIGHT' \
        --min-height '$MIN_HEIGHT' \
        --max-height '$MAX_HEIGHT' \
        --crouch-height '$CROUCH_HEIGHT' \
        --ctrl-c-action limp
    echo '[decision demo exited]'
    exec bash"

tmux select-layout -t "$SESSION" tiled

echo ""
echo "tmux session started: $SESSION"
echo "  Ctrl+B arrows: switch panes"
echo "  Ctrl+B D: detach"
echo "  tmux attach -t $SESSION: reattach"
echo "  focus pane3 and begin with supervised small commands:"
echo "    1 0.2   forward 0.2m"
echo "    3 0.1   left 0.1m"
echo "    5 15    yaw left 15deg"
echo "    0       stop / interrupt active motion"
echo ""

sleep 1
tmux attach -t "$SESSION"
