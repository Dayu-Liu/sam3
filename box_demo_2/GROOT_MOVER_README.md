# GR00T-WBC ← box_demo precise-locomotion interface

The cm/deg → velocity interface that lets the box-hugging manipulation pipeline
drive the **GR00T-WBC** locomotion base for precise small moves. Local source of
truth: `cc/box_demo_groot/`; deploy target: `5080-laptop:/home/wjzh/agile_boxdeploy/box_demo_2/`.

## What's here

| File | Role | State |
| --- | --- | --- |
| `groot_mover.py` | **NEW** — drop-in for `RobotMover`. cm/deg/m/rad → velocity+duration, writes `units:"agile"` (physical m/s), with a **velocity floor above the 0.05 Walk threshold**. | ✅ 34 tests pass; deployed |
| `test_groot_mover.py` | table-driven tests for the conversion math + the walk-threshold invariant | ✅ |
| `groot_wbc_boxdemo_adapter.py` | adapter (theirs) + 2 **additive** edits: sub-0.05 dead-band guard, and opt-in `--log-pose` (sim ground-truth) | ✅ compiles in `.venv_wbc`; original at `.orig` |
| `measure_precision.py` | **NEW** — drives `groot_mover` through small/medium targets, prints a goal-vs-actual table from the pose CSV | ✅ analysis verified |
| `start_groot_wbc_precision_test.sh` | **NEW** — MuJoCo launcher: merger + adapter(`--log-pose`) + the harness | ready to run on the laptop |

## Two things this fixes for GR00T-WBC

1. **Units double-scale bug** — the old `RobotMover` wrote *normalized* velocity with
   no `units`, so the adapter re-multiplied by its caps (0.50/0.30/0.60) while
   RobotMover assumed AGILE caps (1.0/0.5/0.5) → every distance came out wrong.
   `groot_mover.py` always emits `units:"agile"` physical m/s (no rescale).
2. **Balance/Walk 0.05 threshold** — GR00T balances (no step) when
   `‖[vx,vy,wz]‖ < 0.05`. So **precise small distance = above-floor velocity ×
   short time, never a tiny velocity.** `groot_mover` floors velocity at 0.08 m/s
   / 0.10 rad/s; the smallest reliable increment is ~one step (≈5 cm / ≈3.4°).
   Sub-step precision is left to the perceive→move→re-perceive loop.

## Run the precision test (on the laptop, MuJoCo — no real robot)

```bash
cd ~/agile_boxdeploy/box_demo_2
bash start_groot_wbc_precision_test.sh \
  --groot-repo ~/GR00T-WholeBodyControl \
  --wbc-venv  ~/GR00T-WholeBodyControl/.venv_wbc \
  --ros-distro humble
# pane2 opens the MuJoCo viewer; pane3 drives the sequence and prints:
#   move   target  expected  actual   err  flr
# Re-analyze later: python measure_precision.py --analyze-only \
#   --save-moves /tmp/groot_moves.json --pose-csv /tmp/groot_pose.csv
```

## Base-rotation decision (operator-confirmed)

`rotate(deg)` on the base is used **only in the approach phase**; during the grasp
the base does not turn (waist/`WaistRotator` only), to avoid desyncing the arm IK
target. Wiring this into `box_demo_main.py` is Step 4 below.

## Status / next steps

- ✅ Step 1: `groot_mover.py` (interface + bug fix + floor) — done, tested, deployed.
- ✅ Step 2: adapter dead-band guard + `--log-pose` — done, additive, compiles.
- 🟡 Step 3: run `start_groot_wbc_precision_test.sh` in MuJoCo → first goal-vs-actual
  precision table (needs the laptop display; harness is ready).
- ⬜ Step 4: swap the mover in `box_demo_main.py` behind a flag
  (`from groot_mover import RobotMover`), wire `rotate(deg)` into the approach phase only.
- ⬜ Step 5: tethered real-robot bring-up (small presets, conservative caps,
  DAMP/estop verified). Keep `merge_lowcmd_arm_sdk.py` unchanged.

Do **not** modify `merge_lowcmd_arm_sdk.py` (sole `rt/lowcmd` publisher) or the GR00T policy.
