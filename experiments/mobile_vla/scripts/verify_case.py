"""Verify a completed same-world case using files, not a success sentence alone."""

from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import imageio_ffmpeg


def verify(out):
    nav = json.loads((out / "navigation.json").read_text())
    result = json.loads((out / "result.json").read_text())
    trace = json.loads((out / "trace.json").read_text())
    protocol = json.loads((out / "protocol.json").read_text())
    times = np.asarray([x["time"] for x in trace])
    assert np.allclose(np.diff(times), 0.05, atol=1e-7), (
        "Discontinuous simulation clock"
    )
    phases = {x["stage"] for x in trace}
    assert {"navigation", "unfold", "manipulation"} <= phases
    pose = np.asarray([x["base"] for x in trace])
    assert np.linalg.norm(pose[-1, :2] - pose[0, :2]) > 0.5, (
        "No substantial navigation motion"
    )
    assert np.max(np.linalg.norm(np.diff(pose[:, :2], axis=0), axis=1)) < 0.04, (
        "Pose jump incompatible with bounded base velocity"
    )
    a = np.load(out / "policy_actions.npz")
    pred = a["predictions"]
    applied = a["applied"]
    assert np.array_equal(pred[:, :5].reshape(-1, 7)[: len(applied)], applied), (
        "Applied action is not the model queue"
    )
    inputs = sorted((out / "policy_inputs").glob("*.npz"))
    assert len(inputs) == len(pred)
    hashes = []
    for p in inputs:
        x = np.load(p)
        assert x["image"].shape == x["wrist"].shape == (224, 224, 3)
        assert x["state"].shape == (8,) and np.isfinite(x["state"]).all()
        hashes.append(hashlib.sha256(x["image"].tobytes()).hexdigest())
    assert len(set(hashes)) > 1, "No refreshed policy RGB"
    frames, _ = imageio_ffmpeg.count_frames_and_secs(str(out / "rollout.mp4"))
    assert frames == len(trace)
    manipulation = [x for x in trace if x["stage"] == "manipulation"]
    xy = np.asarray([x["base"][:2] for x in manipulation])
    drift = float(np.max(np.linalg.norm(xy - xy[0], axis=1)))
    r = {
        "navigation_success": nav["navigation_success"],
        "official_task_success": result["success"],
        "collision_watchdog_events": result["collision_count"],
        "continuous_clock": True,
        "no_pose_teleport": True,
        "model_action_queue_equal": True,
        "refreshed_rgb_queries": len(set(hashes)),
        "policy_queries": len(pred),
        "policy_steps": len(applied),
        "video_frames_decoded": frames,
        "parking_drift_m": drift,
        "grasp_seen": any(x.get("object_grasped") is True for x in trace),
        "grasp_at_end": trace[-1].get("object_grasped"),
        "initial_object_z": trace[0].get("object_z"),
        "max_object_z": max(
            (x["object_z"] for x in trace if x.get("object_z") is not None),
            default=None,
        ),
        "dock_error_m": protocol.get("navigation_arrival_error_m"),
        "world_reset_after_navigation": False,
    }
    (out / "verification.json").write_text(json.dumps(r, indent=2))
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    a = p.parse_args()
    print(json.dumps(verify(a.out.resolve()), indent=2))


if __name__ == "__main__":
    main()
