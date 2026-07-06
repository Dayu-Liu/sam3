"""Compare goal.csv (IK targets) vs state.csv (actual positions) for end-effector error analysis."""
import csv
import math

def read_transposed_csv(path):
    """Read transposed CSV where rows=variables, columns=samples."""
    data = {}
    with open(path) as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            key = row[0]
            values = []
            for v in row[1:]:
                try:
                    values.append(float(v))
                except ValueError:
                    values.append(v)  # keep strings (timestamps)
            data[key] = values
    return data

def main():
    goal = read_transposed_csv("goal.csv")
    state = read_transposed_csv("state.csv")

    # Get timestamps
    goal_ts = goal.get("timestamp", [])
    state_ts = state.get("timestamp", [])

    print(f"goal.csv: {len(goal_ts)} samples")
    print(f"state.csv: {len(state_ts)} samples")

    # Find today's data (2026-06-09) in state.csv
    today_idx = [i for i, t in enumerate(state_ts) if isinstance(t, str) and "2026-06-09" in t]
    if not today_idx:
        print("No 2026-06-09 data in state.csv")
        return
    print(f"\nToday's state data: indices {today_idx[0]} to {today_idx[-1]}")
    print(f"  From {state_ts[today_idx[0]]} to {state_ts[today_idx[-1]]}")

    # Find today's data in goal.csv
    goal_today_idx = [i for i, t in enumerate(goal_ts) if isinstance(t, str) and "2026-06-09" in t]
    if not goal_today_idx:
        print("No 2026-06-09 data in goal.csv")
        # Use the last entry
        goal_today_idx = [len(goal_ts) - 1]
        print(f"  Using last goal entry instead: {goal_ts[goal_today_idx[0]]}")
    else:
        print(f"\nToday's goal data: {len(goal_today_idx)} entries")
        for idx in goal_today_idx:
            print(f"  [{idx}] {goal_ts[idx]}")

    # === ANALYSIS 1: IK accuracy (goal FK vs target) ===
    print("\n" + "=" * 60)
    print("分析 1: IK 精度 (FK计算位置 vs 目标位置)")
    print("=" * 60)

    for idx in goal_today_idx:
        ts = goal_ts[idx]
        print(f"\n  Goal at {ts}:")

        # Left hand
        lt = goal.get("left_target_x", [None])[idx], goal.get("left_target_y", [None])[idx], goal.get("left_target_z", [None])[idx]
        lf = goal.get("left_fk_x", [None])[idx], goal.get("left_fk_y", [None])[idx], goal.get("left_fk_z", [None])[idx]

        if all(v is not None for v in lt) and all(v is not None for v in lf):
            err = math.sqrt(sum((a - b) ** 2 for a, b in zip(lt, lf)))
            print(f"  Left target:  ({lt[0]:.4f}, {lt[1]:.4f}, {lt[2]:.4f})")
            print(f"  Left FK:      ({lf[0]:.4f}, {lf[1]:.4f}, {lf[2]:.4f})")
            print(f"  Left IK error: {err * 100:.2f} cm")

        # Right hand
        rt = goal.get("right_target_x", [None])[idx], goal.get("right_target_y", [None])[idx], goal.get("right_target_z", [None])[idx]
        rf = goal.get("right_fk_x", [None])[idx], goal.get("right_fk_y", [None])[idx], goal.get("right_fk_z", [None])[idx]

        if all(v is not None for v in rt) and all(v is not None for v in rf):
            err = math.sqrt(sum((a - b) ** 2 for a, b in zip(rt, rf)))
            print(f"  Right target: ({rt[0]:.4f}, {rt[1]:.4f}, {rt[2]:.4f})")
            print(f"  Right FK:     ({rf[0]:.4f}, {rf[1]:.4f}, {rf[2]:.4f})")
            print(f"  Right IK error: {err * 100:.2f} cm")

    # === ANALYSIS 2: Tracking error (actual FK vs goal FK) ===
    print("\n" + "=" * 60)
    print("分析 2: 跟踪误差 (实际到达位置 vs IK计算位置)")
    print("=" * 60)

    for g_idx in goal_today_idx:
        g_ts = goal_ts[g_idx]
        print(f"\n  Goal at {g_ts}:")

        # Get goal FK positions
        g_lfk = [goal.get(f"left_fk_{ax}", [None])[g_idx] for ax in ["x", "y", "z"]]
        g_rfk = [goal.get(f"right_fk_{ax}", [None])[g_idx] for ax in ["x", "y", "z"]]

        if any(v is None for v in g_lfk) or any(v is None for v in g_rfk):
            print("  No FK data in goal")
            continue

        # Find the closest state entries (within ±5 seconds of goal timestamp)
        # Parse goal timestamp
        from datetime import datetime
        try:
            g_time = datetime.fromisoformat(g_ts)
        except:
            continue

        # Find state entries near the goal time
        best_state_idx = None
        min_dt = float('inf')
        for si in today_idx:
            try:
                s_time = datetime.fromisoformat(state_ts[si])
                dt = abs((s_time - g_time).total_seconds())
                if dt < min_dt:
                    min_dt = dt
                    best_state_idx = si
            except:
                continue

        if best_state_idx is None:
            print("  No matching state entry found")
            continue

        print(f"  Closest state at {state_ts[best_state_idx]} (Δt={min_dt:.1f}s)")

        # Get actual FK from state
        s_lfk = [state.get(f"left_fk_{ax}", [None])[best_state_idx] for ax in ["x", "y", "z"]]
        s_rfk = [state.get(f"right_fk_{ax}", [None])[best_state_idx] for ax in ["x", "y", "z"]]

        if any(v is None for v in s_lfk):
            print("  No FK in state.csv, using last few stable entries instead")
            # Use the last few stable entries (robot should be settled)
            stable_indices = today_idx[-5:]  # last 5 readings
            s_lfk_avg = []
            s_rfk_avg = []
            for ax in ["x", "y", "z"]:
                vals_l = [state.get(f"left_fk_{ax}", [None])[i] for i in stable_indices if state.get(f"left_fk_{ax}", [None])[i] is not None]
                vals_r = [state.get(f"right_fk_{ax}", [None])[i] for i in stable_indices if state.get(f"right_fk_{ax}", [None])[i] is not None]
                s_lfk_avg.append(sum(vals_l) / len(vals_l) if vals_l else None)
                s_rfk_avg.append(sum(vals_r) / len(vals_r) if vals_r else None)
            s_lfk = s_lfk_avg
            s_rfk = s_rfk_avg
            print(f"  Using average of last {len(stable_indices)} stable readings")

        if all(v is not None for v in s_lfk) and all(v is not None for v in g_lfk):
            err_l = math.sqrt(sum((a - b) ** 2 for a, b in zip(s_lfk, g_lfk)))
            print(f"  Left goal FK:  ({g_lfk[0]:.4f}, {g_lfk[1]:.4f}, {g_lfk[2]:.4f})")
            print(f"  Left actual:   ({s_lfk[0]:.4f}, {s_lfk[1]:.4f}, {s_lfk[2]:.4f})")
            print(f"  Left tracking error: {err_l * 100:.2f} cm")
            for i, ax in enumerate(["x", "y", "z"]):
                d = (s_lfk[i] - g_lfk[i]) * 100
                print(f"    Δ{ax}: {d:+.2f} cm")

        if all(v is not None for v in s_rfk) and all(v is not None for v in g_rfk):
            err_r = math.sqrt(sum((a - b) ** 2 for a, b in zip(s_rfk, g_rfk)))
            print(f"  Right goal FK: ({g_rfk[0]:.4f}, {g_rfk[1]:.4f}, {g_rfk[2]:.4f})")
            print(f"  Right actual:  ({s_rfk[0]:.4f}, {s_rfk[1]:.4f}, {s_rfk[2]:.4f})")
            print(f"  Right tracking error: {err_r * 100:.2f} cm")
            for i, ax in enumerate(["x", "y", "z"]):
                d = (s_rfk[i] - g_rfk[i]) * 100
                print(f"    Δ{ax}: {d:+.2f} cm")

    # === ANALYSIS 3: Joint-level comparison for recent stable data ===
    print("\n" + "=" * 60)
    print("分析 3: 上体关节角对比 (目标 vs 实际, 取最近稳定时刻)")
    print("=" * 60)

    # Goal joint names (from goal.csv)
    goal_joint_names = [
        "LeftShoulderPitch", "LeftShoulderRoll", "LeftShoulderYaw", "LeftElbow",
        "LeftWristRoll", "LeftWristPitch", "LeftWristYaw",
        "RightShoulderPitch", "RightShoulderRoll", "RightShoulderYaw", "RightElbow",
        "RightWristRoll", "RightWristPitch", "RightWristYaw",
        "WaistYaw", "WaistRoll", "WaistPitch",
    ]

    # State joint names (indices 15-28, 12-14)
    state_joint_names = [
        "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
        "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
        "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
        "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
        "waist_yaw", "waist_roll", "waist_pitch",
    ]

    for g_idx in goal_today_idx[-1:]:  # just the latest goal
        g_ts = goal_ts[g_idx]
        print(f"\n  Goal at {g_ts}:")

        # Use the last stable state reading
        stable_idx = today_idx[-1]
        print(f"  State at {state_ts[stable_idx]}:")

        print(f"\n  {'Joint':<25} {'Goal (rad)':>10} {'Actual (rad)':>12} {'Error (rad)':>11} {'Error (°)':>9}")
        print("  " + "-" * 70)

        max_err = 0
        max_err_joint = ""
        for gj, sj in zip(goal_joint_names, state_joint_names):
            g_key = f"{gj}_q"
            s_key = f"{sj}_q"
            g_val = goal.get(g_key, [None])[g_idx]
            s_val = state.get(s_key, [None])[stable_idx]

            if g_val is not None and s_val is not None:
                err = s_val - g_val
                err_deg = math.degrees(err)
                abs_err = abs(err_deg)
                flag = " <<<" if abs_err > 2.0 else ""
                if abs_err > max_err:
                    max_err = abs_err
                    max_err_joint = sj
                print(f"  {sj:<25} {g_val:>+10.4f} {s_val:>+12.4f} {err:>+11.4f} {err_deg:>+8.2f}°{flag}")

        print(f"\n  最大误差关节: {max_err_joint} = {max_err:.2f}°")

    # === ANALYSIS 4: Historical FK error trends ===
    print("\n" + "=" * 60)
    print("分析 4: 历史 IK 误差趋势 (目标 vs FK, 最近10次)")
    print("=" * 60)

    recent_goals = list(range(max(0, len(goal_ts) - 10), len(goal_ts)))
    print(f"\n  {'Timestamp':<28} {'Left IK err':>12} {'Right IK err':>13}")
    print("  " + "-" * 55)

    for g_idx in recent_goals:
        ts = goal_ts[g_idx]
        lt = [goal.get(f"left_target_{ax}", [None])[g_idx] for ax in ["x", "y", "z"]]
        lf = [goal.get(f"left_fk_{ax}", [None])[g_idx] for ax in ["x", "y", "z"]]
        rt = [goal.get(f"right_target_{ax}", [None])[g_idx] for ax in ["x", "y", "z"]]
        rf = [goal.get(f"right_fk_{ax}", [None])[g_idx] for ax in ["x", "y", "z"]]

        left_err = right_err = None
        if all(v is not None for v in lt) and all(v is not None for v in lf):
            left_err = math.sqrt(sum((a - b) ** 2 for a, b in zip(lt, lf))) * 100
        if all(v is not None for v in rt) and all(v is not None for v in rf):
            right_err = math.sqrt(sum((a - b) ** 2 for a, b in zip(rt, rf))) * 100

        ts_str = str(ts)[:23] if ts else "N/A"
        le_str = f"{left_err:.2f} cm" if left_err is not None else "N/A"
        re_str = f"{right_err:.2f} cm" if right_err is not None else "N/A"
        print(f"  {ts_str:<28} {le_str:>12} {re_str:>13}")


if __name__ == "__main__":
    main()
