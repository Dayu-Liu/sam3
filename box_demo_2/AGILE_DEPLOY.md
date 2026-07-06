# box_demo_2 + AGILE Deploy Notes

## Current Routing

The original `start.sh` still launches the old RoboJuDo locomotion pipeline:

```text
merge_lowcmd_arm_sdk.py
run_pipeline.py -c g1_mjlab_loco_real_merge
box_demo_main.py
```

Use `start_agile_box.sh` for the AGILE lower-body base:

```text
merge_lowcmd_arm_sdk.py
agile_lowcmd_pipeline.py
agile_keyboard_control.py
box_demo_main.py
```

`box_demo_main.py` is intentionally unchanged at the manipulation level. It
still runs:

```text
D435 camera -> VLM -> SAM3 3D grasp points -> cam_to_torso -> ArmKinematics IK
-> DualArmController -> rt/arm_sdk
```

The replacement is only the lower-body source. `agile_lowcmd_pipeline.py` reads
`/tmp/robojudo_ext_cmd.json`, feeds AGILE's
`[vx, vy, wz, absolute_height]` command, and publishes `rt/lowcmd_rl`.
`merge_lowcmd_arm_sdk.py` still performs the final arbitration:

```text
rt/lowcmd_rl from AGILE + rt/arm_sdk from box_demo -> rt/lowcmd
```

## Keyboard Interface

`agile_keyboard_control.py` writes the same IPC file used by `RobotMover`.
It stays passive when idle, so it does not overwrite autonomous box-demo motion.

Keys:

- `w/s`: forward/backward
- `a/d`: left/right strafe
- `q/e`: yaw left/right
- `z/x`: lower/raise AGILE absolute base-height command
- `c`: set pick-height preset, default `0.55m`
- `r`: return to stand height `0.72m`
- `space`: stop velocity, keep current height
- `f`: `RL_FULL`
- `l`: `RL_LOWER`, legs balance while upper body is available for `arm_sdk`
- `o`: damping request

Height is clamped to `[0.40, 0.72]m` and slewed inside
`agile_lowcmd_pipeline.py` at `0.20m/s` by default. For hardware bring-up, use
`z/x` in 2cm increments and treat `c=0.55m` as the first grasp-height preset.

## Startup

For manual keyboard bring-up on the 5080, use the minimal launcher. It starts
only the merger, AGILE lower-body pipeline, and keyboard IPC control:

```bash
cd ~/agile_boxdeploy/box_demo_2
./start_agile_manual.sh
```

Current defaults are chosen for the 5080 + Unitree wired link:

```text
iface=enp130s0
AGILE_REPO=~/agile_boxdeploy/WBC-AGILE
device=cpu, torch_threads=1
fwd/lat/yaw limits=0.30 / 0.15 / 0.30
keyboard vx/vy/wz=0.12 / 0.08 / 0.10
```

When `box_demo_main.py` is not running, the AGILE pipeline now holds waist and
arms at AGILE's default upper-body pose (`q=0` for motors `12..28`, default
`kp=40,kd=1`) instead of holding the gantry/suspended startup pose. When
`rt/arm_sdk` is active, the merger still lets box-demo upper-body commands
override motors `12..29`.

For a non-invasive environment check, use:

```bash
./start_agile_manual.sh --dry-run
```

In this manual launcher, `--dry-run` does not start the merger, so it does not
publish final `rt/lowcmd`.

```bash
cd /path/to/sim2real/box_demo_2
./start_agile_box.sh \
  --vlm-endpoint https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --iface enP8p1s0 \
  --agile-repo /path/to/WBC-AGILE \
  --device cpu \
  --confirm
```

If the autonomous `RobotMover` alignment under-tracks because the AGILE velocity
limits are intentionally conservative, tune without editing code:

```bash
./start_agile_box.sh ... --fwd-max 0.60 --lat-max 0.30 --yaw-max 0.60 --walk-scale 1.2
```

Dry-run policy/IPC bring-up without publishing `rt/lowcmd_rl`:

```bash
./start_agile_box.sh \
  --vlm-endpoint https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --iface enP8p1s0 \
  --agile-repo /path/to/WBC-AGILE \
  --dry-run
```

## Safety Notes

- Before live publishing, `agile_lowcmd_pipeline.py` mirrors the Unitree and
  SONIC low-level examples: after `ChannelFactoryInitialize()`, it checks the
  active Unitree motion service with `MotionSwitcherClient.CheckMode()` and calls
  `ReleaseMode()` until no high-level sport/AI/advanced motion mode is active.
  `--dry-run` skips this by default; pass `--motion-release-in-dry-run` only when
  you intentionally want to test the mode switch. Use `--no-motion-release` only
  if another supervisor has already released the Unitree motion service.
- `agile_lowcmd_pipeline.py` does not publish `rt/lowcmd` directly; it publishes
  `rt/lowcmd_rl`, so the existing merger remains the only final motor command
  publisher.
- AGILE controls lower motors `0..11`. Upper motors `12..28` are held at their
  measured positions in `rt/lowcmd_rl`; any real upper-body motion must come
  from `rt/arm_sdk`.
- During grasp, `box_demo_main.py` calls `RobotMover.shutdown()`, which writes
  `RL_LOWER` while preserving the current height command. AGILE keeps balancing
  the legs and `rt/arm_sdk` overrides the waist/arms through the merger.
- The IMU gyroscope is used as body-frame angular velocity. Do not add another
  world-to-body rotation in this hardware path.

## Safety Guards (hardened 2026-06-17)

`agile_lowcmd_pipeline.py` now runs hot-path guards on every tick (the merger is
intentionally **not** modified — that is a separate change pending sign-off).
On any trip it **latches DAMP** (kd=8) and keeps sending damping frames until the
process is restarted:

- **Non-finite guard** — checks `isfinite` on the obs, the policy action, and the
  joint target. A NaN/Inf latches DAMP. (The position clamp does *not* sanitize
  NaN — `clamp(NaN)=NaN` — so the finite-check runs first, by design.)
- **Tilt / fall abort** — body tilt from gravity-in-body-frame (AGILE's own
  `quat_rotate_inverse`); `--tilt-limit-deg` default `50` (AGILE trains to 40°).
- **Joint overspeed abort** — `--overspeed-rad-s` default `30`, over the 12 legs.
- **q_target clamp + rate-limit** — legs clamped to the YAML
  `default_joint_pos_limits` and bounded to `--max-delta-rad` (default `0.6`) per
  tick. Generous backstop, not a smoother.
- **State-freshness watchdog** — `--state-timeout-s` default `0.5`. On stale
  LowState the pipeline itself emits damping (does not just stop publishing), and
  the prepare ramp aborts to DAMP if state goes stale mid-ramp.
- **Hardware e-stop** — decodes the `wireless_remote` **select** button into the
  DAMP latch (disable with `--no-remote-estop`).
- **Mode handshake** — refuses to start if first-frame `mode_machine==0`
  (`--allow-zero-mode-machine` for sim/loopback); warns if firmware `mode_pr`
  differs from the published `mode_pr` (the merger forwards the RL frame's mode).
- **Startup self-test** — proves obs→policy→action on the first real LowState and
  asserts the action dim before any motion.

> ⚠️ Still hardware-gated and **not yet ported**: the merger damping fail-safe
> (`merge_lowcmd_arm_sdk.py` goes silent on stale `rt/lowcmd_rl`) is a separate
> reviewed change. See `cc/AGILE_BOX_DEMO2_SIM2REAL.md` §2 for the full must-fix
> list and the incremental bring-up plan.

## Smoke Check

On `4090` with the AGILE environment:

```text
OK 29 80 12 MLPPolicyWrapper 2
```

This verifies AGILE config loading, 29-joint policy order, 80-D observation
processor, 12-D action processor, TorchScript policy wrapper, and two recurrent
buffers reset. It does not exercise DDS or real motors.
