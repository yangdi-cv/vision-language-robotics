"""Offline dataset validation and observation video; never sends robot commands."""
import argparse
import csv
import json
from pathlib import Path

import av
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    session = json.loads((root / "session.json").read_text())
    if session["status"] not in ("review_required", "reviewed"):
        raise ValueError(f"Dataset is not ready: {session['status']}")
    dataset = LeRobotDataset("local/desktop_pick_place", root=root)
    with (root / "timing.csv").open() as f:
        rows = list(csv.DictReader(f))
    if len(rows) != len(dataset) or not rows:
        raise ValueError("Frame/timing count mismatch or empty episode")
    times = np.array([int(r["state_received_ns"]) for r in rows], dtype=np.int64)
    gaps_ms = np.diff(times) / 1e6
    offsets_ms = np.array([int(r["image_received_ns"]) - int(r["state_received_ns"]) for r in rows]) / 1e6
    if any(int(r["frame_index"]) != i for i, r in enumerate(rows)):
        raise ValueError("Timing frame indices are not contiguous")
    if len(gaps_ms) and np.any(gaps_ms <= 0):
        raise ValueError("Non-monotonic host timestamps")
    states, actions = [], []
    video_path = root / "observation_preview.mp4"
    with av.open(str(video_path), "w") as output:
        stream = output.add_stream("libx264", rate=dataset.fps)
        stream.width = session["image_shape"][1]
        stream.height = session["image_shape"][0]
        stream.pix_fmt = "yuv420p"
        for i in range(len(dataset)):
            frame = dataset[i]
            state = frame["observation.state"].numpy()
            action = frame["action"].numpy()
            if not np.isfinite(state).all() or not np.isfinite(action).all():
                raise ValueError(f"Non-finite joint values at frame {i}")
            states.append(state)
            actions.append(action)
            image = (frame["observation.images.front"].permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    state = np.stack(states)
    action = np.stack(actions)
    quality = {
        "frames": len(dataset), "episodes": dataset.num_episodes,
        "task": session["task"], "nominal_fps": dataset.fps,
        "observed_fps": float(1000 / gaps_ms.mean()) if len(gaps_ms) else None,
        "cycle_ms_p50": float(np.percentile(gaps_ms, 50)) if len(gaps_ms) else None,
        "cycle_ms_p95": float(np.percentile(gaps_ms, 95)) if len(gaps_ms) else None,
        "cycles_above_1_5_nominal": int(np.sum(gaps_ms > 1500 / dataset.fps)),
        "state_to_image_receipt_ms_p95": float(np.percentile(offsets_ms, 95)),
        "state_range_by_joint": dict(zip(session["joint_order"], np.ptp(state, axis=0).tolist())),
        "max_action_minus_state": float(np.abs(action - state).max()),
        "task_success": "unreviewed",
        "notes": "Receipt timestamps do not establish exposure synchronization or camera drop count. Video plays at nominal FPS. Action/state differences are command steps, not tracking errors.",
    }
    (root / "quality.json").write_text(json.dumps(quality, indent=2, ensure_ascii=False))
    print(json.dumps(quality, indent=2, ensure_ascii=False))
    print(f"Observation replay: {video_path}")


if __name__ == "__main__":
    main()
