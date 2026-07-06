#!/bin/bash
# G1 box_demo_2 + AGILE lower-body launcher.
#
# Panes:
#   1) merge_lowcmd_arm_sdk.py        rt/lowcmd_rl + rt/arm_sdk -> rt/lowcmd
#   2) agile_lowcmd_pipeline.py       AGILE lower-body -> rt/lowcmd_rl
#   3) agile_keyboard_control.py      optional WASD/QE/Z/X/C/R IPC teleop
#   4) box_demo_main.py               camera/VLM/SAM3/IK/arm_sdk grasp flow

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SIM2REAL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SESSION="${SESSION:-g1-agile-box}"

VLM_ENDPOINT="${VLM_ENDPOINT:-}"
SAM3_HOST="${SAM3_HOST:-127.0.0.1}"
SAM3_PORT="${SAM3_PORT:-5300}"
IFACE="${IFACE:-enP8p1s0}"
AGILE_REPO="${AGILE_REPO:-$SIM2REAL_DIR/cc/experiments/repos/WBC-AGILE}"
DEVICE="${DEVICE:-cpu}"
CONDA_ENV="${CONDA_ENV:-hdmi}"
TORCH_THREADS="${TORCH_THREADS:-1}"
WALK_SCALE="${WALK_SCALE:-1.0}"
FWD_MAX="${FWD_MAX:-0.50}"
LAT_MAX="${LAT_MAX:-0.30}"
YAW_MAX="${YAW_MAX:-0.60}"
HEIGHT_RATE="${HEIGHT_RATE:-0.20}"
NO_CONFIRM="--no-confirm"
DRY_RUN=""
MOTION_RELEASE_ARGS=""
WITH_KEYBOARD=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --vlm-endpoint) VLM_ENDPOINT="$2"; shift 2 ;;
        --sam3-host)    SAM3_HOST="$2";    shift 2 ;;
        --sam3-port)    SAM3_PORT="$2";    shift 2 ;;
        --iface)        IFACE="$2";        shift 2 ;;
        --agile-repo)   AGILE_REPO="$2";   shift 2 ;;
        --device)       DEVICE="$2";       shift 2 ;;
        --conda-env)    CONDA_ENV="$2";    shift 2 ;;
        --torch-threads) TORCH_THREADS="$2"; shift 2 ;;
        --walk-scale)   WALK_SCALE="$2";   shift 2 ;;
        --fwd-max)      FWD_MAX="$2";      shift 2 ;;
        --lat-max)      LAT_MAX="$2";      shift 2 ;;
        --yaw-max)      YAW_MAX="$2";      shift 2 ;;
        --height-rate)  HEIGHT_RATE="$2";  shift 2 ;;
        --confirm)      NO_CONFIRM="";     shift ;;
        --dry-run)      DRY_RUN="--dry-run"; shift ;;
        --no-motion-release) MOTION_RELEASE_ARGS="$MOTION_RELEASE_ARGS --no-motion-release"; shift ;;
        --motion-release-in-dry-run) MOTION_RELEASE_ARGS="$MOTION_RELEASE_ARGS --motion-release-in-dry-run"; shift ;;
        --motion-release-timeout-s) MOTION_RELEASE_ARGS="$MOTION_RELEASE_ARGS --motion-release-timeout-s $2"; shift 2 ;;
        --allow-motion-release-failure) MOTION_RELEASE_ARGS="$MOTION_RELEASE_ARGS --allow-motion-release-failure"; shift ;;
        --no-keyboard)  WITH_KEYBOARD=0;   shift ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

if [[ -z "$VLM_ENDPOINT" ]]; then
    echo "错误: 需要 --vlm-endpoint URL"
    echo "用法: $0 --vlm-endpoint https://your-api/v1 [--sam3-host 127.0.0.1] [--sam3-port 5300] [--iface enP8p1s0] [--confirm]"
    exit 1
fi

echo "========================================"
echo "  G1 box_demo_2 + AGILE lower body"
echo "  iface:      $IFACE"
echo "  VLM:        $VLM_ENDPOINT"
echo "  SAM3:       $SAM3_HOST:$SAM3_PORT"
echo "  AGILE_REPO: $AGILE_REPO"
echo "  device:     $DEVICE"
echo "  conda env:  $CONDA_ENV"
echo "  torch threads: $TORCH_THREADS"
echo "  walk-scale: $WALK_SCALE"
echo "  fwd/lat/yaw max: $FWD_MAX / $LAT_MAX / $YAW_MAX"
echo "  height-rate: $HEIGHT_RATE"
echo "  dry-run:    ${DRY_RUN:-no}"
echo "  motion release: ${MOTION_RELEASE_ARGS:-default live ReleaseMode}"
echo "========================================"

tmux kill-session -t "$SESSION" 2>/dev/null || true

export OMP_NUM_THREADS="$TORCH_THREADS"
export MKL_NUM_THREADS="$TORCH_THREADS"
export OPENBLAS_NUM_THREADS="$TORCH_THREADS"
export NUMEXPR_NUM_THREADS="$TORCH_THREADS"
export VECLIB_MAXIMUM_THREADS="$TORCH_THREADS"

tmux new-session -d -s "$SESSION" -n merge "
    source '$HOME/miniconda3/etc/profile.d/conda.sh'
    conda activate '$CONDA_ENV'
    cd '$SIM2REAL_DIR'
    echo '=== pane1: merge rt/lowcmd_rl + rt/arm_sdk -> rt/lowcmd ==='
    echo \"python: \$(which python)\"
    python '$SCRIPT_DIR/merge_lowcmd_arm_sdk.py' --iface '$IFACE'
    echo '[merge exited]'
    exec bash"

tmux split-window -h -t "$SESSION" "
    source '$HOME/miniconda3/etc/profile.d/conda.sh'
    conda activate '$CONDA_ENV'
    cd '$SIM2REAL_DIR'
    sleep 1
    echo '=== pane2: AGILE lower-body pipeline -> rt/lowcmd_rl ==='
    echo \"python: \$(which python)\"
    python '$SCRIPT_DIR/agile_lowcmd_pipeline.py' \
        --iface '$IFACE' \
        --agile-repo '$AGILE_REPO' \
        --device '$DEVICE' \
        --torch-threads '$TORCH_THREADS' \
        --fwd-max '$FWD_MAX' \
        --lat-max '$LAT_MAX' \
        --yaw-max '$YAW_MAX' \
        --height-rate '$HEIGHT_RATE' \
        $MOTION_RELEASE_ARGS \
        $DRY_RUN
    echo '[AGILE pipeline exited]'
    exec bash"

if [[ "$WITH_KEYBOARD" == "1" ]]; then
    tmux split-window -v -t "$SESSION:0.1" "
        source '$HOME/miniconda3/etc/profile.d/conda.sh'
        conda activate '$CONDA_ENV'
        cd '$SIM2REAL_DIR'
        sleep 2
        echo '=== pane3: keyboard IPC control ==='
        echo \"python: \$(which python)\"
        echo 'focus this pane for WASD/QE, z/x height, c pick-height, r stand'
        python '$SCRIPT_DIR/agile_keyboard_control.py'
        echo '[keyboard exited]'
        exec bash"
fi

tmux split-window -v -t "$SESSION:0.0" "
    source '$HOME/miniconda3/etc/profile.d/conda.sh'
    conda activate '$CONDA_ENV'
    cd '$SIM2REAL_DIR'
    sleep 3
    echo '=== pane4: box_demo_main camera/VLM/SAM3/IK/arm_sdk ==='
    echo \"python: \$(which python)\"
    python '$SCRIPT_DIR/box_demo_main.py' \
        --vlm-endpoint '$VLM_ENDPOINT' \
        --host '$SAM3_HOST' \
        --port '$SAM3_PORT' \
        --iface '$IFACE' \
        --walk-scale '$WALK_SCALE' \
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
