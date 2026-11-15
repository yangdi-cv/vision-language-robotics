"""Replay recorded absolute actions, with read-only preflight by default."""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
from teleop_table_lerobot import JOINTS, P_BY_JOINT, BASELINE_P


def load_actions(root, calibration_root):
    session = json.loads((root / "session.json").read_text())
    if session["status"] not in ("review_required", "reviewed"):
        raise ValueError("Dataset has not finished saving")
    if session["joint_order"] != list(JOINTS) or session["fps_nominal"] != 10:
        raise ValueError("Unexpected joint order or recording rate")
    for relative, expected in session["calibration_sha256"].items():
        if hashlib.sha256((calibration_root / relative).read_bytes()).hexdigest() != expected:
            raise ValueError("Calibration files changed since recording")
    tables = [pq.read_table(p, columns=["action", "observation.state", "frame_index", "episode_index"])
              for p in sorted((root / "data").rglob("*.parquet"))]
    if len(tables) != 1:
        raise ValueError("Pilot replay requires one episode data file")
    rows = tables[0].to_pylist()
    actions = np.array([r["action"] for r in rows], dtype=float)
    initial = np.array(rows[0]["observation.state"], dtype=float)
    if actions.shape != (session["frames"], 6) or not np.isfinite(actions).all():
        raise ValueError("Invalid action array")
    if any(r["frame_index"] != i or r["episode_index"] != 0 for i, r in enumerate(rows)):
        raise ValueError("Frame ordering mismatch")
    return actions, initial


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--execute", action="store_true", help="Enable motion after explicit confirmation")
    parser.add_argument("--validate-only", action="store_true", help="Check data without connecting hardware")
    parser.add_argument("--return-to-start", action="store_true", help="Slowly return to recorded initial state; do not replay")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[2]
    calibration_root = project / "calibration"
    actions, initial = load_actions(args.root.resolve(), calibration_root)
    print(f"Validated {len(actions)} action frames, nominal duration {len(actions)/10:.1f}s.", flush=True)
    if args.validate_only:
        return
    follower = SO101Follower(SO101FollowerConfig(
        port="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5CE5122996-if00",
        id="my_awesome_follower_arm",
        calibration_dir=calibration_root / "robots/so_follower", use_degrees=True,
    ))
    enabled = False
    modified = set()
    log = None
    try:
        follower.bus.connect()
        if not follower.bus.is_calibrated:
            raise ValueError("Motor calibration mismatch; replay cancelled")
        actual = follower.bus.sync_read("Present_Position")
        current = np.array([actual[j] for j in JOINTS])
        delta = initial - current
        for joint, now, start, difference in zip(JOINTS, current, initial, delta):
            print(f"{joint}: current={now:.2f}, recorded start={start:.2f}, difference={difference:.2f}", flush=True)
        if not args.execute and not args.return_to_start:
            print("READ-ONLY CHECK COMPLETE; no goal or torque writes.", flush=True)
            return
        if args.return_to_start and np.max(np.abs(delta)) > 20:
            raise ValueError("Return exceeds 20 units; manual reposition required")
        if not args.return_to_start and np.max(np.abs(delta)) > 5:
            raise ValueError("Start differs by more than 5 units. Reposition manually; no automatic move to start.")
        expected = "RETURN" if args.return_to_start else "REPLAY"
        confirmation = input(f"Clear workspace and prepare power cutoff. Type {expected} to enable white arm motion: ")
        if confirmation != expected:
            print("Cancelled")
            return
        # Check again after the operator's preparation.
        actual = follower.bus.sync_read("Present_Position")
        current = np.array([actual[j] for j in JOINTS])
        limit = 20 if args.return_to_start else 5
        if np.max(np.abs(current - initial)) > limit:
            raise ValueError("Arm moved during confirmation; start check failed")
        raw = follower.bus.sync_read("Present_Position", normalize=False)
        follower.bus.sync_write("Goal_Position", raw, normalize=False)
        follower.bus.disconnect(disable_torque=False)
        enabled = True  # cleanup must disable torque even if configure raises
        follower.connect(calibrate=False)
        for joint, value in P_BY_JOINT.items():
            with follower.bus.torque_disabled(joint):
                follower.bus.write("Goal_Position", joint, follower.bus.read("Present_Position", joint, normalize=False), normalize=False)
                modified.add(joint)
                follower.bus.write("P_Coefficient", joint, value)
        log_path = project / "records" / f"replay_{time.time_ns()}.csv"
        log = log_path.open("w", newline="")
        writer = csv.writer(log)
        writer.writerow(["frame_index", "monotonic_ns"] + [f"{j}_actual" for j in JOINTS] + [f"{j}_sent" for j in JOINTS])
        if args.return_to_start:
            command = current.copy()
            settled = 0
            for i in range(200):
                start = time.monotonic()
                obs = follower.get_observation()
                measured = np.array([obs[f"{j}.pos"] for j in JOINTS])
                if np.max(np.abs(command - measured)) > 10:
                    raise RuntimeError("Return tracking error exceeds 10 units; stopped")
                command = command + np.clip(initial - command, -.5, .5)
                follower.send_action({f"{j}.pos": float(v) for j, v in zip(JOINTS, command)})
                writer.writerow([i, time.monotonic_ns(), *measured, *command])
                settled = settled + 1 if np.max(np.abs(initial - measured)) < 2 else 0
                if i % 10 == 0:
                    print(f"Return error max={np.max(np.abs(initial-measured)):.2f}", flush=True)
                    log.flush()
                if settled >= 10:
                    break
                time.sleep(max(0, .1 - (time.monotonic() - start)))
            else:
                raise RuntimeError("Return did not settle within 20 seconds")
            print(f"Returned to start. Holding position. Log: {log_path}", flush=True)
            input("Support white arm, then ENTER to disable torque: ")
            return
        for i, target in enumerate(actions):
            start = time.monotonic()
            obs = follower.get_observation()
            current = np.array([obs[f"{j}.pos"] for j in JOINTS])
            command = np.clip(target, current - 5, current + 5)
            follower.send_action({f"{j}.pos": float(v) for j, v in zip(JOINTS, command)})
            writer.writerow([i, time.monotonic_ns(), *current, *command])
            if i % 10 == 0:
                log.flush()
                print(f"Replay {i+1}/{len(actions)}", flush=True)
            time.sleep(max(0, .1 - (time.monotonic() - start)))
        print(f"Replay complete. Log: {log_path}", flush=True)
        input("Arm is holding final pose. Support white arm, then ENTER to disable torque: ")
    except KeyboardInterrupt:
        print("Stopping; white arm will lose torque.", flush=True)
    finally:
        if log:
            log.close()
        if follower.bus.is_connected:
            if enabled:
                try:
                    follower.bus.disable_torque()
                    for joint in modified:
                        follower.bus.write("P_Coefficient", joint, BASELINE_P)
                finally:
                    follower.bus.disconnect(disable_torque=True)
            else:
                follower.bus.disconnect(disable_torque=False)


if __name__ == "__main__":
    main()
