#!/bin/bash
# G1 视觉抓取一键启动 —— tmux 三窗格，支持 SSH
# 用法: ./start.sh --vlm-endpoint URL [--iface enP8p1s0] [--walk-scale 1.0] [--confirm]
# 快捷键: Ctrl+B 方向键 切窗格, Ctrl+B D 退出(后台运行), tmux attach -t g1-grasp 重新连入

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SESSION="g1-grasp"

# ── 参数 ─────────────────────────────────────────────────────────
VLM_ENDPOINT="${VLM_ENDPOINT:-}"
IFACE="${IFACE:-enP8p1s0}"
WALK_SCALE="${WALK_SCALE:-1.0}"
NO_CONFIRM="--no-confirm"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --vlm-endpoint) VLM_ENDPOINT="$2"; shift 2 ;;
        --iface)        IFACE="$2";        shift 2 ;;
        --walk-scale)   WALK_SCALE="$2";   shift 2 ;;
        --confirm)      NO_CONFIRM="";     shift ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

if [[ -z "$VLM_ENDPOINT" ]]; then
    echo "错误: 需要 --vlm-endpoint URL"
    echo "用法: $0 --vlm-endpoint https://your-api/v1 [--iface enP8p1s0] [--walk-scale 1.0] [--confirm]"
    exit 1
fi

echo "========================================"
echo "  G1 视觉抓取 一键启动 (tmux)"
echo "  iface:      $IFACE"
echo "  VLM:        $VLM_ENDPOINT"
echo "  walk-scale: $WALK_SCALE"
echo "========================================"

cd "$PROJECT_DIR"

# 干掉旧 session（如果存在）
tmux kill-session -t "$SESSION" 2>/dev/null || true

# ── 创建 session，第一个窗格跑 merge ──────────────────────────────
tmux new-session -d -s "$SESSION" -n merge "
    echo '=== 窗格1: merge_lowcmd_arm_sdk ==='
    python zihou/box_demo_2/merge_lowcmd_arm_sdk.py --iface $IFACE
    echo '[merge 已退出]'
    exec bash"

# ── 第二个窗格（右侧）：RL pipeline ───────────────────────────────
tmux split-window -h -t "$SESSION" "
    sleep 1
    echo '=== 窗格2: run_pipeline (RL) ==='
    cd locomotion/RoboJuDo_zihou2
    python scripts/run_pipeline.py -c g1_mjlab_loco_real_merge
    echo '[pipeline 已退出]'
    exec bash"

# ── 第三个窗格（下方）：box_demo ──────────────────────────────────
tmux split-window -v -t "$SESSION" "
    sleep 2
    echo '=== 窗格3: box_demo_main ==='
    python zihou/box_demo_2/box_demo_main.py \
        --vlm-endpoint '$VLM_ENDPOINT' \
        --iface $IFACE \
        --walk-scale $WALK_SCALE \
        $NO_CONFIRM
    echo '[box_demo 已退出]'
    exec bash"

# ── 调整布局：左侧 merge+box_demo 上下分，右侧 pipeline ──────────
tmux select-layout -t "$SESSION" tiled

echo ""
echo "三个窗格已启动，正在进入 tmux..."
echo ""
echo "  快捷键:"
echo "    Ctrl+B ↑↓←→  切换窗格"
echo "    Ctrl+B D      退出（后台继续运行）"
echo "    tmux attach -t $SESSION  重新连入"
echo "    在 box_demo 窗格按 Ctrl+C 可停止抓取流程"
echo ""

sleep 1
tmux attach -t "$SESSION"
