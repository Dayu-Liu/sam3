#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GrootMover — precise small-move locomotion interface for GR00T-WBC.

Drop-in replacement for box_demo_2/robot_move.py::RobotMover that drives the
GR00T-WBC decoupled-WBC base via the file IPC `/tmp/robojudo_ext_cmd.json`,
consumed by groot_wbc_boxdemo_adapter.py. The manipulation pipeline keeps calling
in meters/radians (or the cm/deg convenience wrappers); this module converts a
distance/angle request into a *velocity + duration* and refreshes the IPC while
the move is active (open-loop timed velocity — there is no odometry feedback).

Two things it fixes vs the AGILE RobotMover, both required for GR00T-WBC:

1. UNITS — it always writes `"units": "agile"` with PHYSICAL m/s + rad/s. The old
   RobotMover wrote NORMALIZED values with no `units`, so the GR00T adapter
   re-multiplied them by its own caps (0.50/0.30/0.60) while RobotMover assumed
   the AGILE caps (1.0/0.5/0.5) — every commanded distance came out wrong.

2. WALK THRESHOLD — GR00T auto-switches Balance<->Walk at
   ||[vx,vy,wz]|| < 0.05 (g1_gear_wbc_policy.py:223). A too-small velocity never
   steps. So precise SMALL distance = a velocity ABOVE the floor held for a SHORT
   time, never a tiny velocity. We enforce a velocity floor (V_FLOOR / W_FLOOR)
   comfortably above 0.05; the smallest reliable increment is therefore
   ~V_FLOOR*MIN_DURATION (≈ one step). Sub-step precision is left to the
   perceive->move->re-perceive outer loop, not commanded.

MuJoCo characterization (5080 scene_29dof + real Balance/Walk onnx, 2026-07-01;
scratchpad sweep_step_limit.py). Findings that set the defaults below:
  * STARTUP RAMP: from a stand the Walk gait needs ~0.6-0.8 s to reach cruise
    displacement rate. So realized forward = ~35-45% of commanded at T<=1.0 s,
    rising to ~55-70% at T>=2 s (the fixed ramp cost gets amortized). The robot
    NEVER falls on a short command — it just fails to make net progress.
  * SIGN INSTABILITY (this is the "腰晃脚不走/反而后退" bug): the NET displacement
    depends on the residual standing-sway phase at command onset. From an
    UNSETTLED start the same command can net BACKWARD — measured across sway
    phases: 0.08x0.6s spans -7.4..+2.2 cm; even 0.12x1.2s -> -2.3 and 0.15x1.2s
    -> -1.0 at the worst phase. From a SETTLED start every case is strongly
    net-positive (+7..+13 cm). So the backward tail is an UNSETTLED-start effect.
    => fix (validated end-to-end in sim2sim_groot_mover.py, driving THIS module
       through the policy sim): a WARM-UP pre-step (WARMUP_TIME 0.6 s @ 0.15 m/s)
       is the primary robustifier — it establishes the gait deterministically so
       the move rides an already-walking base, making the worst-case net across
       ALL sway phases strictly positive (+3.9 cm; never backward). A fixed
       SETTLE_BEFORE_S does NOT help (periodic sway -> a fixed delay can land on a
       bad phase; OFF by default). Combine with MIN_DURATION 1.5 s + MIN_DISTANCE
       ~8 cm + V_FLOOR 0.12 (a floored small move nets ~10 cm). In the settled
       pipeline case (>=2.5 s stand before a move) even no-warm-up is reliable.
  * WARM-UP is the key knob (see above); a 0.6 s pre-roll turns the worst-case
    from -4.9 cm to +3.9 cm across sway phases. Cost: every move becomes a
    reliable ~10 cm+ step (the warm-up travels ~0.15*0.6 m too).
  * WALK-HEIGHT FLOOR: the Walk policy collapses fast below stand height —
    h0.72 walks (~10 cm), h0.70 is the last good rung, h0.68 already collapses
    (~3 cm, missing steps), h<=0.66 = zero steps. You CANNOT walk while squatting;
    navigate at stand height, squat only when stationary (WALK_MIN_HEIGHT=0.72
    guard keeps one rung above the cliff).
  * YAW is well-behaved (65-89% efficient, reliable even at T=0.6 s) — no change.
Sim is idealized (instant command, no comms latency, ideal friction) so it is an
OPTIMISTIC bound; keep a margin on real hardware and let the camera loop correct.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from dataclasses import dataclass

CMD_FILE = "/tmp/robojudo_ext_cmd.json"

# Absolute pelvis-height command (meters). Nominal 0.74 matches the adapter
# (DEFAULT_BASE_HEIGHT) / keyboard / bundled controller, gives ~2cm margin above
# the 0.72 walk-floor guard, and walks better than 0.72 (mujoco: 0.74->13cm vs
# 0.72->10.6cm for the same command).
STAND_HEIGHT = 0.74
MIN_HEIGHT = 0.40
MAX_HEIGHT = 0.74

# GR00T-WBC Balance<->Walk switch (raw physical norm). Below this -> no stepping.
WALK_THRESHOLD = 0.05

# Velocity floors. Above WALK_THRESHOLD AND high enough that a floored small move
# nets a USEFUL distance: mujoco shows 0.08 m/s held 1.5 s nets only ~5.7 cm
# (<the 8 cm min increment), whereas 0.12 m/s x 1.5 s nets ~10 cm. So the floor is
# 0.12, not 0.08 — a floored move actually clears MIN_DISTANCE.
V_FLOOR = 0.12   # m/s   linear (forward / lateral)
W_FLOOR = 0.10   # rad/s yaw (~5.7 deg/s)

# Physical caps — keep inside the adapter's conservative box-demo caps. With
# units="agile" the adapter does NOT rescale; it only clamps to its own caps.
FWD_MAX = 0.50   # m/s
LAT_MAX = 0.30   # m/s
YAW_MAX = 0.60   # rad/s

# Cruise speeds for precise moves (well below the caps; smaller = less overshoot).
# Bumped from 0.12/0.10 — mujoco: higher cruise = higher realized efficiency.
FWD_CRUISE = 0.15  # m/s
LAT_CRUISE = 0.12  # m/s
YAW_CRUISE = 0.15  # rad/s (~8.6 deg/s)

# --- reliability knobs (mujoco-tuned; see the module docstring) ---
# (A) min execution time. Sensitivity sweep across sway phases: T<=1.2 s still
# nets BACKWARD at the worst phase (vx0.12/T1.2 -> -2.3 cm; vx0.15/T1.2 -> -1.0);
# T>=1.5 s at vx>=0.15 (or T>=2.0 at vx0.12) is the first reliably net-positive
# floor. 1.5 s chosen; pair with SETTLE_BEFORE_S so the worst phase never occurs.
MIN_DURATION = 1.5   # s
# (B) min NET forward/lateral increment. Below ~8 cm the move is ramp/sway
# dominated and sign-unstable. 0 disables; recommend 0.08-0.10. The
# perceive->move->re-perceive loop converges the sub-increment remainder.
MIN_DISTANCE = 0.08  # m
# (C) warm-up pre-step: a brief roll to spin up the gait before the measured
# move. This is the PRIMARY robustifier for the "反而后退" bug — the sim2sim test
# (sim2sim_groot_mover.py, driving THIS module through the policy sim) shows that
# a 0.6 s @ 0.15 m/s warm-up makes the worst-case net across ALL sway phases
# strictly positive (+3.9 cm; never backward), vs -4.9 cm without it. It works by
# establishing the gait deterministically so the measured move rides an
# already-walking base instead of a swaying stand. Adds ~warmup_speed*warmup_time
# of travel (so every move is a reliable ~10 cm+ step). ON by default.
WARMUP_SPEED = 0.15  # m/s
WARMUP_TIME = 0.6    # s
# settle guard: hold Balance this long before a move. COUNTERINTUITIVELY this can
# HURT: residual standing sway is ~periodic, so a FIXED settle delay just shifts
# the phase and can land ON a bad phase (sim2sim: settle=0.5 is worse than 0.0 in
# every warm-up config). Left as a knob but OFF by default — prefer the warm-up.
SETTLE_BEFORE_S = 0.0  # s
# (E) walk-height floor: GR00T-WBC Walk collapses fast below stand height —
# h0.72 walks (~10 cm), h0.70 still walks but is the last good rung, h0.68 already
# collapses (~3 cm, 2 missing steps), h<=0.66 = zero steps. Floor at 0.72 to keep
# one rung of margin above the cliff (height estimate noise / sag tolerance).
WALK_MIN_HEIGHT = 0.72  # m
# (D) open-loop distance calibration: realized ~= dist/DIST_GAIN of commanded.
# 1.0 = command the raw target (closed loop corrects). ~1.7 compensates the
# ~55-60% realization so a single open-loop move lands closer. Opt-in.
DIST_GAIN = 1.0

REFRESH_HZ = 20.0    # Hz  must refresh inside the adapter stale window (0.4 s)
STOP_HOLD_S = 0.4    # s   write zero velocity after a move so the base settles


@dataclass
class MotionPlan:
    """Resolved velocity + duration for one axis move (open-loop)."""

    speed: float       # signed physical speed (m/s or rad/s)
    duration: float    # seconds
    expected: float    # signed expected displacement (m or rad)
    floored: bool      # True if the floor bound -> the small target will overshoot


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def solve_linear(distance_m: float, cruise: float, v_floor: float,
                 v_max: float, min_duration: float) -> MotionPlan:
    """distance (m) -> (speed m/s, duration s), never below the walk floor.

    Normal distances ride at ``cruise``. Small distances stretch to
    ``min_duration`` at a reduced speed, but the speed is never allowed below
    ``v_floor`` — if preserving the distance would need a sub-floor speed, we
    hold the floor and accept a small overshoot (flagged), so the robot always
    actually steps instead of silently balancing.
    """
    d = float(distance_m)
    if abs(d) < 1e-6:
        return MotionPlan(0.0, 0.0, 0.0, False)
    v_floor = min(abs(v_floor), abs(v_max))
    direction = 1.0 if d >= 0.0 else -1.0
    speed = _clamp(abs(cruise), v_floor, abs(v_max))
    duration = abs(d) / speed
    floored = False
    if duration < min_duration:
        duration = min_duration
        need = abs(d) / duration
        if need < v_floor:
            speed = v_floor          # accept overshoot to stay above threshold
            floored = True
        else:
            speed = need
    expected = math.copysign(speed * duration, d)
    return MotionPlan(direction * speed, duration, expected, floored)


def solve_yaw(angle_rad: float, cruise: float, w_floor: float,
              w_max: float, min_duration: float) -> MotionPlan:
    """angle (rad) -> (yaw_rate rad/s, duration s), never below the walk floor."""
    a = float(angle_rad)
    if abs(a) < 1e-6:
        return MotionPlan(0.0, 0.0, 0.0, False)
    w_floor = min(abs(w_floor), abs(w_max))
    direction = 1.0 if a >= 0.0 else -1.0
    rate = _clamp(abs(cruise), w_floor, abs(w_max))
    duration = abs(a) / rate
    floored = False
    if duration < min_duration:
        duration = min_duration
        need = abs(a) / duration
        if need < w_floor:
            rate = w_floor
            floored = True
        else:
            rate = need
    expected = math.copysign(rate * duration, a)
    return MotionPlan(direction * rate, duration, expected, floored)


def write_command(cmd_file: str, fsm: str | None, forward: float = 0.0,
                  lateral: float = 0.0, yaw: float = 0.0,
                  height: float | None = None, duration: float | None = None,
                  estop: bool = False, limp: bool = False) -> None:
    """Atomically write the GR00T IPC command with units='agile' (physical)."""
    cmd: dict = {
        "fsm": fsm,
        "velocity": {"forward": float(forward), "lateral": float(lateral),
                     "yaw": float(yaw)},
        "units": "agile",
        "timestamp": time.time(),
        "source": "groot_mover",
    }
    if height is not None:
        cmd["height"] = float(_clamp(height, MIN_HEIGHT, MAX_HEIGHT))
    if duration is not None and duration > 0:
        cmd["duration"] = float(duration)
    if estop:
        cmd["estop"] = True
    if limp:
        cmd["limp"] = True
    dir_name = os.path.dirname(cmd_file) or "/tmp"
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(cmd, f)
        os.replace(tmp_path, cmd_file)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


class GrootMover:
    """File-IPC mover for the GR00T-WBC base. Drop-in for RobotMover."""

    def __init__(self, velocity: float | None = None, *, cmd_file: str = CMD_FILE,
                 stand_height: float = STAND_HEIGHT,
                 fwd_cruise: float = FWD_CRUISE, lat_cruise: float = LAT_CRUISE,
                 yaw_cruise: float = YAW_CRUISE, fwd_max: float = FWD_MAX,
                 lat_max: float = LAT_MAX, yaw_max: float = YAW_MAX,
                 v_floor: float = V_FLOOR, w_floor: float = W_FLOOR,
                 min_duration: float = MIN_DURATION, refresh_hz: float = REFRESH_HZ,
                 stop_hold_s: float = STOP_HOLD_S, min_distance: float = MIN_DISTANCE,
                 warmup_speed: float = WARMUP_SPEED, warmup_time: float = WARMUP_TIME,
                 settle_before_s: float = SETTLE_BEFORE_S,
                 walk_min_height: float = WALK_MIN_HEIGHT,
                 auto_raise_for_walk: bool = False, dist_gain: float = DIST_GAIN,
                 verbose: bool = True):
        self.cmd_file = cmd_file
        self._height = stand_height
        # `velocity` (RobotMover-compatible positional) overrides the fwd cruise.
        self.fwd_cruise = float(velocity) if velocity else fwd_cruise
        self.lat_cruise = lat_cruise
        self.yaw_cruise = yaw_cruise
        self.fwd_max, self.lat_max, self.yaw_max = fwd_max, lat_max, yaw_max
        self.v_floor, self.w_floor = v_floor, w_floor
        # Reliability knobs (mujoco-tuned; see module docstring). All are plain
        # attributes so box_demo can tweak per-test:
        #  (A) min_duration  — stretch a move to >=1.2 s so it reliably net-steps.
        #  (B) min_distance  — snap a sub-8cm target up to a reliable increment.
        #  (C) warmup_time   — pre-roll to spin up the gait (0=off).
        #  settle_before_s   — Balance hold before a move to kill sway (0=off).
        #  (E) walk_min_height — refuse/raise below the ~0.70 m walk floor.
        #  (D) dist_gain     — open-loop distance compensation (1=off).
        self.min_duration = min_duration
        self.min_distance = min_distance
        self.warmup_speed = warmup_speed
        self.warmup_time = warmup_time
        self.settle_before_s = settle_before_s
        self.walk_min_height = walk_min_height
        self.auto_raise_for_walk = auto_raise_for_walk
        self.dist_gain = max(1.0, float(dist_gain))
        self.refresh_hz = refresh_hz
        self.stop_hold_s = stop_hold_s
        self.verbose = verbose

    # ----------------------------------------------------------------- helpers
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"  GrootMover: {msg}")

    def _hold(self, forward: float, lateral: float, yaw: float,
              duration: float) -> None:
        """Hold a velocity for `duration` (blocking). NO trailing settle, so it
        can be chained (settle -> warm-up -> move) without a Balance gap that
        would let the gait spin down between phases."""
        if duration <= 0:
            return
        period = 1.0 / self.refresh_hz
        deadline = time.time() + duration
        while time.time() < deadline:
            remaining = max(0.0, deadline - time.time())
            write_command(self.cmd_file, "RL_FULL", forward, lateral, yaw,
                          height=self._height, duration=remaining)
            time.sleep(period)

    def _settle(self) -> None:
        """Write zero velocity (Balance) for stop_hold_s so the base comes to rest."""
        period = 1.0 / self.refresh_hz
        end = time.time() + self.stop_hold_s
        while time.time() < end:
            write_command(self.cmd_file, "RL_FULL", 0.0, 0.0, 0.0,
                          height=self._height)
            time.sleep(period)

    def _refresh_for(self, forward: float, lateral: float, yaw: float,
                     duration: float) -> None:
        """Hold then settle (kept for back-compat; move_* uses _execute_move)."""
        self._hold(forward, lateral, yaw, duration)
        self._settle()

    def _guard_walk_height(self) -> bool:
        """GR00T-WBC Walk stops stepping below ~0.70 m (mujoco sweep). Returns
        True if it is OK to walk. Warns (and optionally raises) otherwise."""
        if self._height >= self.walk_min_height - 1e-6:
            return True
        if self.auto_raise_for_walk:
            prev = self._height  # capture BEFORE set_height mutates it
            self._log(f"height {prev:.2f}m < walk floor "
                      f"{self.walk_min_height:.2f}m -> raising before walking")
            self.set_height(self.walk_min_height)
            # wait for the height slew (adapter height_rate ~0.20 m/s) + margin,
            # so the base is actually above the walk floor before it steps.
            time.sleep(max(0.0, self.walk_min_height - prev) / 0.20 + 0.3)
            return True
        self._log(f"[WARN] height {self._height:.2f}m < walk floor "
                  f"{self.walk_min_height:.2f}m: GR00T-WBC won't step while "
                  f"squatting. Raise height (or set auto_raise_for_walk) to walk.")
        return False

    def _execute_move(self, forward: float, lateral: float, yaw: float,
                      duration: float, warm: bool = True) -> None:
        """settle-before -> warm-up pre-step -> main move -> settle."""
        if self.settle_before_s > 0:
            self._hold(0.0, 0.0, 0.0, self.settle_before_s)  # damp residual sway
        if warm and self.warmup_time > 0:
            wf = math.copysign(self.warmup_speed, forward) if abs(forward) > 1e-9 else 0.0
            wl = math.copysign(self.warmup_speed, lateral) if abs(lateral) > 1e-9 else 0.0
            if wf or wl:
                self._hold(wf, wl, 0.0, self.warmup_time)  # spin up the gait
        self._hold(forward, lateral, yaw, duration)
        self._settle()

    def _snap_min_distance(self, distance_m: float, tag: str) -> float:
        """选项B: 把过小的线性目标顶到 min_distance。太短的前进只会让步态做一个
        原地踏步/略后退的启停瞬态、没有净位移;顶到最小增量(如 5/10cm)保证真走出去。
        min_distance=0 时关闭。box_demo 的 perceive->move->re-perceive 会用更粗的步收敛。"""
        d = float(distance_m)
        if self.min_distance > 0.0 and 1e-6 <= abs(d) < self.min_distance:
            self._log(f"{tag} {abs(d)*100:.0f}cm < 最小增量 {self.min_distance*100:.0f}cm -> 顶到最小")
            return math.copysign(self.min_distance, d)
        return d

    # ----------------------------------------------------- RobotMover surface
    def initialize(self) -> None:
        write_command(self.cmd_file, "RL_FULL", height=self._height)
        time.sleep(1.5)
        self._log("RL_FULL ready.")

    def move_forward(self, distance_m: float) -> MotionPlan:
        """Forward (+) / backward (-) by distance in METERS. Blocks until done."""
        desired = self._snap_min_distance(distance_m, "forward")
        commanded = desired * self.dist_gain
        plan = solve_linear(commanded, self.fwd_cruise, self.v_floor,
                            self.fwd_max, self.min_duration)
        if plan.duration <= 0:
            return plan
        self._guard_walk_height()
        tag = "forward" if desired > 0 else "backward"
        flo = " [floored->overshoot]" if plan.floored else ""
        gain = f" gain->{abs(commanded)*100:.1f}cm" if self.dist_gain > 1.0 else ""
        self._log(f"{tag} {abs(desired)*100:.1f}cm{gain} "
                  f"(v={plan.speed:+.3f}m/s, {plan.duration:.2f}s){flo}")
        self._execute_move(plan.speed, 0.0, 0.0, plan.duration)
        return plan

    def move_left(self, distance_m: float) -> MotionPlan:
        """Left (+) / right (-) strafe by distance in METERS. Blocks until done."""
        desired = self._snap_min_distance(distance_m, "strafe")
        commanded = desired * self.dist_gain
        plan = solve_linear(commanded, self.lat_cruise, self.v_floor,
                            self.lat_max, self.min_duration)
        if plan.duration <= 0:
            return plan
        self._guard_walk_height()
        tag = "left" if desired > 0 else "right"
        flo = " [floored->overshoot]" if plan.floored else ""
        gain = f" gain->{abs(commanded)*100:.1f}cm" if self.dist_gain > 1.0 else ""
        self._log(f"{tag} {abs(desired)*100:.1f}cm{gain} "
                  f"(v={plan.speed:+.3f}m/s, {plan.duration:.2f}s){flo}")
        self._execute_move(0.0, plan.speed, 0.0, plan.duration)
        return plan

    def rotate(self, angle_rad: float, yaw_speed: float | None = None) -> MotionPlan:
        """Turn left (+) / right (-) by angle in RADIANS. Blocks until done."""
        cruise = self.yaw_cruise if yaw_speed is None else yaw_speed
        plan = solve_yaw(angle_rad, cruise, self.w_floor, self.yaw_max,
                         self.min_duration)
        if plan.duration <= 0:
            return plan
        self._guard_walk_height()
        tag = "left" if angle_rad > 0 else "right"
        flo = " [floored->overshoot]" if plan.floored else ""
        self._log(f"turn {tag} {abs(math.degrees(angle_rad)):.1f}deg "
                  f"(w={plan.speed:+.3f}rad/s, {plan.duration:.2f}s, "
                  f"exp={math.degrees(plan.expected):+.1f}deg){flo}")
        self._execute_move(0.0, 0.0, plan.speed, plan.duration, warm=False)
        return plan

    def stop(self) -> None:
        write_command(self.cmd_file, "RL_FULL", height=self._height)

    def set_height(self, height: float) -> None:
        self._height = float(_clamp(height, MIN_HEIGHT, MAX_HEIGHT))
        write_command(self.cmd_file, "RL_FULL", height=self._height)
        self._log(f"base height = {self._height:.2f}m")

    def adjust_height(self, delta_m: float) -> None:
        self.set_height(self._height + float(delta_m))

    def shutdown(self) -> None:
        """Hand the upper body to arm_sdk: legs keep balancing (RL_LOWER)."""
        write_command(self.cmd_file, "RL_LOWER", height=self._height)
        time.sleep(0.5)
        self._log("RL_LOWER (legs balance, arms free). IPC file kept.")

    handoff_to_arms = shutdown  # explicit alias

    def damp(self) -> None:
        write_command(self.cmd_file, "DAMP", height=self._height, estop=True)
        self._log("DAMP (kd-only damping).")

    estop = damp

    def limp(self) -> None:
        write_command(self.cmd_file, "LIMP", height=self._height, limp=True)
        self._log("LIMP (release stiffness).")

    def release(self) -> None:
        try:
            if os.path.exists(self.cmd_file):
                os.remove(self.cmd_file)
        except OSError:
            pass
        self._log("IPC file cleared.")

    # ----------------------------------------------------- cm/deg convenience
    def move_forward_cm(self, cm: float) -> MotionPlan:
        """Manipulation-pipeline entry: forward (+) / back (-) in CENTIMETERS."""
        return self.move_forward(float(cm) / 100.0)

    def strafe_cm(self, cm: float) -> MotionPlan:
        """Left (+) / right (-) strafe in CENTIMETERS."""
        return self.move_left(float(cm) / 100.0)

    def rotate_deg(self, deg: float) -> MotionPlan:
        """Turn left (+) / right (-) in DEGREES."""
        return self.rotate(math.radians(float(deg)))


# Drop-in alias so `from groot_mover import RobotMover` swaps the base 1:1.
RobotMover = GrootMover
