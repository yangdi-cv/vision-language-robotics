"""SO-ARM101 teleoperation with a compact live terminal table.

Run from the lerobot-hiwonder Conda environment. COM12 is the white follower;
COM13 is the black leader. Keep the follower supported before pressing Ctrl+C.
"""

import csv
import os
import sys
import time
from pathlib import Path

from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
from lerobot.teleoperators.so_leader import SOLeader, SOLeaderTeleopConfig


FPS = 10
MAX_STEP = 5.0  # degrees for joints; percentage points for gripper
JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


def enable_ansi() -> bool:
    if os.name != "nt":
        return True
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.GetStdHandle.restype = ctypes.c_void_p
        kernel32.GetConsoleMode.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
        kernel32.SetConsoleMode.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        handle = kernel32.GetStdHandle(-11)
        mode = ctypes.c_ulong()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
    except (AttributeError, OSError, ValueError):
        return False


def draw(target: dict, actual: dict, sent: dict, health: dict, ansi: bool) -> None:
    lines = [
        "SO-ARM101  COM13 black -> COM12 white   10 Hz   max step 5 deg",
        "Joint            Black target   White actual   Sent now     Difference",
        "---------------------------------------------------------------------",
    ]
    for joint in JOINTS:
        key = f"{joint}.pos"
        lines.append(
            f"{joint:<16} {target[key]:>11.1f}   {actual[key]:>11.1f}   "
            f"{sent[key]:>8.1f}   {target[key] - actual[key]:>11.1f}"
        )
    lines.extend(
        [
            "",
            "Shoulder motor: "
            f"torque={health['torque']} status={health['status']} "
            f"goal(raw)={health['goal']} current(raw)={health['current']} voltage(raw)={health['voltage']} "
            f"temp(C)={health['temperature']}",
            "Angles are degrees; gripper is percent. Difference = target - actual.",
            "Before Ctrl+C, support the white arm on the table; it will lose torque.",
        ]
    )
    table = "\n".join(lines) + "\n"
    sys.stdout.write(("\x1b[H\x1b[J" if ansi else "\n") + table)
    sys.stdout.flush()


def main() -> None:
    leader = SOLeader(SOLeaderTeleopConfig(port="COM13", id="my_awesome_leader_arm"))
    follower = SOFollower(
        SOFollowerRobotConfig(
            port="COM12",
            id="my_awesome_follower_arm",
            max_relative_target=None,
        )
    )
    ansi = enable_ansi()
    trace_path = Path(__file__).with_name("teleop_trace.csv")
    trace = None
    follower_ready = False
    print("Connecting COM13 (black) and COM12 (white)...", flush=True)
    try:
        print("Step 1/4: connecting black leader (COM13)...", flush=True)
        leader.connect()
        print("Step 1/4 complete: black leader connected.", flush=True)
        # An old goal may remain in a servo register from a previous run.
        # Replace it with the current pose before follower.connect() enables torque.
        print("Step 2/4: checking white motors (COM12)...", flush=True)
        follower.bus.connect()
        print("Step 2/4 complete: all white motors responded.", flush=True)
        try:
            print("Step 3/4: matching white motor goals to current positions...", flush=True)
            current_raw = follower.bus.sync_read("Present_Position", normalize=False)
            follower.bus.sync_write("Goal_Position", current_raw, normalize=False)
            print("Step 3/4 complete: current positions saved as goals.", flush=True)
        finally:
            follower.bus.disconnect(disable_torque=False)
        print("Step 4/4: loading white calibration and motor settings...", flush=True)
        follower.connect()
        follower_ready = True
        print("Step 4/4 complete: white follower ready.", flush=True)
        trace = trace_path.open("w", newline="", encoding="utf-8")
        writer = csv.writer(trace)
        writer.writerow(
            ["time_s"]
            + [f"{joint}_black" for joint in JOINTS]
            + [f"{joint}_white" for joint in JOINTS]
            + [f"{joint}_sent" for joint in JOINTS]
            + ["shoulder_torque", "shoulder_status", "shoulder_goal_raw", "shoulder_current_raw", "shoulder_voltage_raw", "shoulder_temp_c"]
        )
        if ansi:
            sys.stdout.write("\x1b[2J\x1b[H")
        last_draw = 0.0
        last_health = 0.0
        health = dict.fromkeys(("torque", "status", "goal", "current", "voltage", "temperature"), "-")
        trace_start = time.perf_counter()
        count = 0
        while True:
            cycle_start = time.perf_counter()
            actual = follower.get_observation()
            target = leader.get_action()
            sent = {}
            for joint in JOINTS:
                key = f"{joint}.pos"
                current = float(actual[key])
                goal = float(target[key])
                sent[key] = max(current - MAX_STEP, min(current + MAX_STEP, goal))
            follower.send_action(sent)
            if cycle_start - last_health >= 0.5:
                try:
                    health = {
                        "torque": follower.bus.read("Torque_Enable", "shoulder_lift"),
                        "status": follower.bus.read("Status", "shoulder_lift"),
                        "goal": follower.bus.read("Goal_Position", "shoulder_lift", normalize=False),
                        "current": follower.bus.read("Present_Current", "shoulder_lift"),
                        "voltage": follower.bus.read("Present_Voltage", "shoulder_lift"),
                        "temperature": follower.bus.read("Present_Temperature", "shoulder_lift"),
                    }
                except RuntimeError as exc:
                    health = dict.fromkeys(health, "read error")
                    print(f"Shoulder telemetry error: {exc}", file=sys.stderr, flush=True)
                last_health = cycle_start
            writer.writerow(
                [round(cycle_start - trace_start, 3)]
                + [target[f"{joint}.pos"] for joint in JOINTS]
                + [actual[f"{joint}.pos"] for joint in JOINTS]
                + [sent[f"{joint}.pos"] for joint in JOINTS]
                + [health[key] for key in ("torque", "status", "goal", "current", "voltage", "temperature")]
            )
            count += 1
            if count % 10 == 0:
                trace.flush()
            if cycle_start - last_draw >= 0.2:
                draw(target, actual, sent, health, ansi)
                last_draw = cycle_start
            time.sleep(max(0.0, 1.0 / FPS - (time.perf_counter() - cycle_start)))
    except KeyboardInterrupt:
        print("\nStopping teleoperation...", flush=True)
    finally:
        if trace is not None:
            trace.close()
        try:
            if follower.bus.is_connected:
                if follower_ready:
                    try:
                        follower.disconnect()
                    except Exception as exc:
                        print(f"Could not disable every white motor: {exc}", file=sys.stderr, flush=True)
                        if follower.bus.is_connected:
                            follower.bus.disconnect(disable_torque=False)
                        print("White arm torque state is unknown; support the arm.", file=sys.stderr, flush=True)
                else:
                    # The connection failed before this script enabled follower torque.
                    follower.bus.disconnect(disable_torque=False)
        finally:
            if leader.is_connected:
                leader.disconnect()
        if follower_ready:
            print(f"Disconnected. Trace: {trace_path}", flush=True)


if __name__ == "__main__":
    main()
