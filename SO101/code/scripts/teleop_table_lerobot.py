"""Original lerobot.zip SO-ARM101 teleoperation with a live terminal table.

Run from the lerobot Conda environment. On Windows, COM12 is the white
follower and COM13 is the black leader. Support the follower before Ctrl+C.
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig


FPS = 10
MAX_STEP = 5.0  # degrees for joints; percentage points for gripper
BASELINE_P = 16
P_BY_JOINT = {"shoulder_lift": 28, "elbow_flex": 24}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_ROOT = PROJECT_ROOT / "calibration"
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


def draw(target: dict, actual: dict, sent: dict, health: dict, ansi: bool, ports: str) -> None:
    lines = [
        f"Original lerobot.zip  {ports}   10 Hz   max step 5 deg   shoulder P=28 elbow P=24",
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
    parser = argparse.ArgumentParser(description="SO-ARM101 leader/follower teleoperation")
    parser.add_argument("--leader-port", default="COM13" if os.name == "nt" else None)
    parser.add_argument("--follower-port", default="COM12" if os.name == "nt" else None)
    parser.add_argument("--calibration-root", type=Path, default=CALIBRATION_ROOT)
    parser.add_argument("--record-root", type=Path, help="New local LeRobot dataset directory (one episode)")
    parser.add_argument("--task", help="Task description required for recording")
    parser.add_argument("--camera-path", default="/dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0")
    parser.add_argument("--rotation", type=int, choices=(0, 90, 180, 270), default=270)
    parser.add_argument("--record-preview", action="store_true", help="Display the same frames saved by the recorder")
    parser.add_argument("--require-calibration", action="store_true", help="Fail before configuration if saved calibration mismatches")
    parser.add_argument("--wrist-roll-offset", type=float, default=0.0, help="Follower wrist roll offset in calibrated motor degrees")
    parser.add_argument("--max-step", type=float, default=MAX_STEP)
    parser.add_argument("--keep-gripper-position", action="store_true")
    parser.add_argument("--record-format", choices=("lerobot", "jsonl"), default="lerobot")
    parser.add_argument("--quiet", action="store_true", help="Suppress the live terminal table")
    parser.add_argument("--status-file", type=Path, help="Publish controller lifecycle and PID for the integrated UI")
    args = parser.parse_args()

    def publish_status(state):
        if args.status_file:
            temporary = args.status_file.with_suffix(".tmp")
            temporary.write_text(json.dumps({"state": state, "pid": os.getpid()}))
            temporary.replace(args.status_file)

    publish_status("starting")
    if args.record_root and (not args.task or not args.task.strip()):
        parser.error("--record-root requires --task")
    if not args.leader_port or not args.follower_port:
        parser.error("On Linux, specify both --leader-port and --follower-port.")
    calibration_root = args.calibration_root.expanduser().resolve()
    ports = f"{args.leader_port} black -> {args.follower_port} white"
    leader = SO101Leader(
        SO101LeaderConfig(
            port=args.leader_port,
            id="my_awesome_leader_arm",
            calibration_dir=calibration_root / "teleoperators" / "so_leader",
            use_degrees=True,
        )
    )
    follower = SO101Follower(
        SO101FollowerConfig(
            port=args.follower_port,
            id="my_awesome_follower_arm",
            calibration_dir=calibration_root / "robots" / "so_follower",
            use_degrees=True,
            max_relative_target=None,
        )
    )
    ansi = enable_ansi()
    trace_dir = PROJECT_ROOT / "records"
    trace_dir.mkdir(exist_ok=True)
    trace_path = trace_dir / f"teleop_trace_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    trace = None
    follower_ready = False
    follower_configuration_started = False
    tuned_p_modified = set()
    recorder = None
    print(f"Connecting {ports}...", flush=True)
    try:
        if args.record_root:
            if args.record_format == "jsonl":
                from trajectory_recorder import DemonstrationRecorder
            else:
                from demonstration_recorder import DemonstrationRecorder
            recorder = DemonstrationRecorder(
                args.record_root, args.task, args.camera_path, JOINTS,
                calibration_root, {"leader": args.leader_port, "follower": args.follower_port}, FPS,
                rotation=args.rotation,
                preview=args.record_preview,
            )
            recorder.manifest.update(
                wrist_roll_offset_degrees=args.wrist_roll_offset,
                max_step_per_cycle=args.max_step,
                gripper_locked=args.keep_gripper_position,
                action_offset_already_applied=True,
                replay_note="Recorded absolute follower goals already include wrist offset; do not add it again.",
            )
            if args.record_format == "jsonl":
                recorder._manifest()
            else:
                recorder._write_manifest()
            input("Confirm arms fixed, workspace clear, power cutoff reachable, white arm supported. ENTER enables teleoperation and recording; Ctrl+C cancels: ")
        if args.record_root or args.require_calibration:
            for device in (leader, follower):
                try:
                    device.bus.connect()
                    if not device.bus.is_calibrated:
                        raise RuntimeError(f"Calibration mismatch for {device.id}; recording cancelled")
                finally:
                    if device.bus.is_connected:
                        device.bus.disconnect(disable_torque=False)
        print(f"Step 1/4: connecting black leader ({args.leader_port})...", flush=True)
        leader.connect()
        print("Step 1/4 complete: black leader connected.", flush=True)
        # An old goal may remain in a servo register from a previous run.
        # Replace it with the current pose before follower.connect() enables torque.
        print(f"Step 2/4: checking white motors ({args.follower_port})...", flush=True)
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
        follower_configuration_started = True
        follower.connect()
        follower_ready = True
        print("Step 4/4 complete: white follower ready.", flush=True)
        # Keep each joint's goal at its resting pose before torque is restored.
        for motor, desired_p in P_BY_JOINT.items():
            with follower.bus.torque_disabled(motor):
                current_raw = follower.bus.read("Present_Position", motor, normalize=False)
                follower.bus.write("Goal_Position", motor, current_raw, normalize=False)
                tuned_p_modified.add(motor)
                follower.bus.write("P_Coefficient", motor, desired_p)
            actual_p = follower.bus.read("P_Coefficient", motor)
            if actual_p != desired_p:
                raise RuntimeError(f"{motor} P readback was {actual_p}, expected {desired_p}")
            print(f"White {motor} P={actual_p}.", flush=True)
        trace = trace_path.open("w", newline="", encoding="utf-8")
        writer = csv.writer(trace)
        writer.writerow(
            ["time_s"]
            + [f"{joint}_black" for joint in JOINTS]
            + [f"{joint}_white" for joint in JOINTS]
            + [f"{joint}_sent" for joint in JOINTS]
            + ["shoulder_torque", "shoulder_status", "shoulder_goal_raw", "shoulder_current_raw", "shoulder_voltage_raw", "shoulder_temp_c"]
        )
        if ansi and not args.quiet:
            sys.stdout.write("\x1b[2J\x1b[H")
        last_draw = 0.0
        last_health = 0.0
        health = dict.fromkeys(("torque", "status", "goal", "current", "voltage", "temperature"), "-")
        locked_gripper = float(follower.get_observation()["gripper.pos"]) if args.keep_gripper_position else None
        trace_start = time.perf_counter()
        publish_status("running")
        count = 0
        while True:
            cycle_start = time.perf_counter()
            actual = follower.get_observation()
            state_ns = time.monotonic_ns()
            if recorder:
                image, image_ns = recorder.capture()
            target = leader.get_action()
            sent = {}
            for joint in JOINTS:
                key = f"{joint}.pos"
                current = float(actual[key])
                goal = locked_gripper if joint == "gripper" and locked_gripper is not None else float(target[key])
                if joint == "wrist_roll":
                    goal += args.wrist_roll_offset
                    if not -180 <= goal <= 180:
                        raise RuntimeError("Wrist offset target exceeds calibrated single-turn range")
                sent[key] = max(current - args.max_step, min(current + args.max_step, goal))
            follower.send_action(sent)
            if recorder:
                recorder.add(actual, sent, image, state_ns, image_ns, time.monotonic_ns())
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
            if not args.quiet and cycle_start - last_draw >= 0.2:
                draw(target, actual, sent, health, ansi, ports)
                last_draw = cycle_start
            time.sleep(max(0.0, 1.0 / FPS - (time.perf_counter() - cycle_start)))
    except KeyboardInterrupt:
        print("\nStopping teleoperation...", flush=True)
    finally:
        if trace is not None:
            trace.close()
        try:
            if follower.bus.is_connected:
                if follower_ready or follower_configuration_started:
                    if tuned_p_modified:
                        try:
                            follower.bus.disable_torque(list(tuned_p_modified))
                            for motor in P_BY_JOINT:
                                if motor in tuned_p_modified:
                                    follower.bus.write("P_Coefficient", motor, BASELINE_P)
                            print(f"White {', '.join(sorted(tuned_p_modified))} P restored to 16.", flush=True)
                        except Exception as exc:
                            print(f"Could not restore white joint P: {exc}", file=sys.stderr, flush=True)
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
            try:
                if leader.is_connected:
                    leader.disconnect()
            finally:
                if recorder is not None:
                    recorder.close()
        publish_status("stopped")
        if follower_ready:
            print(f"Disconnected. Trace: {trace_path}", flush=True)


if __name__ == "__main__":
    main()
