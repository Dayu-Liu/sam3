#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for groot_mover conversion math (pure, no DDS/robot needed)."""

import json
import math
import os
import tempfile

import pytest

import groot_mover as gm
from groot_mover import MotionPlan, solve_linear, solve_yaw, write_command


# ----------------------------------------------------------------- solve_linear
def test_linear_normal_distance_rides_cruise():
    # 0.20 m at cruise 0.12 -> duration 1.667s, speed 0.12, no floor
    p = solve_linear(0.20, cruise=0.12, v_floor=0.08, v_max=0.50, min_duration=0.6)
    assert p.speed == pytest.approx(0.12, abs=1e-6)
    assert p.duration == pytest.approx(0.20 / 0.12, abs=1e-6)
    assert p.expected == pytest.approx(0.20, abs=1e-6)
    assert not p.floored


def test_linear_medium_distance_preserves_via_min_duration():
    # 0.08 m: 0.08/0.12 = 0.667s > 0.6 -> still cruise, distance preserved
    p = solve_linear(0.08, cruise=0.12, v_floor=0.08, v_max=0.50, min_duration=0.6)
    assert p.expected == pytest.approx(0.08, abs=1e-6)
    assert not p.floored


def test_linear_tiny_distance_floors_and_overshoots():
    # 0.03 m: preserving over 0.6s needs 0.05 m/s < floor -> hold floor, overshoot
    p = solve_linear(0.03, cruise=0.12, v_floor=0.08, v_max=0.50, min_duration=0.6)
    assert p.floored
    assert abs(p.speed) == pytest.approx(0.08, abs=1e-6)
    assert p.duration == pytest.approx(0.6, abs=1e-6)
    assert abs(p.expected) > 0.03  # overshoots the tiny target
    assert abs(p.expected) == pytest.approx(0.08 * 0.6, abs=1e-6)


def test_linear_negative_distance_signs_speed():
    p = solve_linear(-0.20, cruise=0.12, v_floor=0.08, v_max=0.50, min_duration=0.6)
    assert p.speed < 0
    assert p.expected < 0


def test_linear_zero_is_noop():
    p = solve_linear(0.0, cruise=0.12, v_floor=0.08, v_max=0.50, min_duration=0.6)
    assert p == MotionPlan(0.0, 0.0, 0.0, False)


def test_linear_clamps_to_v_max():
    # request a cruise above the cap -> clamp to cap
    p = solve_linear(2.0, cruise=0.90, v_floor=0.08, v_max=0.50, min_duration=0.6)
    assert abs(p.speed) == pytest.approx(0.50, abs=1e-6)


@pytest.mark.parametrize("dist_cm", [0.5, 1, 2, 3, 5, 8, 12, 20, 50, -3, -20])
def test_linear_always_clears_walk_threshold(dist_cm):
    # CRITICAL invariant: any nonzero move commands |v| above the Balance/Walk
    # switch (0.05) with margin, so the robot actually steps.
    p = solve_linear(dist_cm / 100.0, cruise=0.12, v_floor=0.08, v_max=0.50,
                     min_duration=0.6)
    assert abs(p.speed) >= gm.WALK_THRESHOLD + 0.02


# -------------------------------------------------------------------- solve_yaw
def test_yaw_normal_angle():
    # 90 deg at 0.15 rad/s
    p = solve_yaw(math.radians(90), cruise=0.15, w_floor=0.10, w_max=0.60,
                  min_duration=0.6)
    assert p.speed == pytest.approx(0.15, abs=1e-6)
    assert math.degrees(p.expected) == pytest.approx(90.0, abs=1e-3)
    assert not p.floored


def test_yaw_tiny_angle_floors():
    # 2 deg: needs 0.058 rad/s < floor -> hold floor, overshoot
    p = solve_yaw(math.radians(2), cruise=0.15, w_floor=0.10, w_max=0.60,
                  min_duration=0.6)
    assert p.floored
    assert abs(p.speed) == pytest.approx(0.10, abs=1e-6)
    assert abs(math.degrees(p.expected)) > 2.0


def test_yaw_negative_signs_rate():
    p = solve_yaw(math.radians(-30), cruise=0.15, w_floor=0.10, w_max=0.60,
                  min_duration=0.6)
    assert p.speed < 0
    assert p.expected < 0


@pytest.mark.parametrize("deg", [1, 2, 5, 15, 45, 90, -15, -90])
def test_yaw_always_clears_walk_threshold(deg):
    p = solve_yaw(math.radians(deg), cruise=0.15, w_floor=0.10, w_max=0.60,
                  min_duration=0.6)
    assert abs(p.speed) >= gm.WALK_THRESHOLD + 0.02


# ----------------------------------------------------------------- write_command
def test_write_command_emits_agile_units_and_physical_values():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "cmd.json")
        write_command(path, "RL_FULL", forward=0.12, lateral=0.0, yaw=0.0,
                      height=0.55, duration=1.0)
        payload = json.loads(open(path).read())
        assert payload["units"] == "agile"          # the bug fix: physical units
        assert payload["fsm"] == "RL_FULL"
        assert payload["velocity"]["forward"] == pytest.approx(0.12)
        assert payload["height"] == pytest.approx(0.55)
        assert payload["duration"] == pytest.approx(1.0)
        assert "timestamp" in payload


def test_write_command_clamps_height():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "cmd.json")
        write_command(path, "RL_FULL", height=10.0)   # above MAX_HEIGHT
        payload = json.loads(open(path).read())
        assert payload["height"] == pytest.approx(gm.MAX_HEIGHT)


def test_write_command_estop_and_limp_flags():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "cmd.json")
        write_command(path, "DAMP", estop=True)
        assert json.loads(open(path).read())["estop"] is True
        write_command(path, "LIMP", limp=True)
        assert json.loads(open(path).read())["limp"] is True


# ------------------------------------------------------------- mover integration
def test_mover_cm_deg_wrappers_write_agile(monkeypatch, tmp_path):
    path = str(tmp_path / "cmd.json")
    mover = gm.GrootMover(cmd_file=path, refresh_hz=1000.0, stop_hold_s=0.0,
                          min_duration=0.01, verbose=False)
    # very short min_duration + fast refresh so the blocking move returns quickly
    plan = mover.move_forward_cm(20.0)           # 20 cm forward
    assert plan.expected == pytest.approx(0.20, abs=0.02)
    payload = json.loads(open(path).read())      # last write = the settle zero
    assert payload["units"] == "agile"


def test_mover_velocity_positional_overrides_cruise(tmp_path):
    mover = gm.GrootMover(velocity=0.20, cmd_file=str(tmp_path / "c.json"))
    assert mover.fwd_cruise == pytest.approx(0.20)


def test_robotmover_alias_is_grootmover():
    assert gm.RobotMover is gm.GrootMover
