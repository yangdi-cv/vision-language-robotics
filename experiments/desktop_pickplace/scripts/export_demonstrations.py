"""Convert successful scripted RGB-D episodes to task-conditioned Cartesian demonstrations."""

from pathlib import Path
from types import SimpleNamespace
import numpy as np, av, json, hashlib
from PIL import Image
from task import waypoints, action_at

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "runs/batch002"
out = ROOT / "data/demonstrations"
out.mkdir(parents=True, exist_ok=True)
manifest = []
rows = json.loads((source / "results.json").read_text())
for index, summary in enumerate(rows):
    if summary["perception"] != "rgbd" or not summary["success"]:
        continue
    directory = source / f"case_{index:03d}"
    destination = out / f"episode_{index:03d}.npz"
    data = np.genfromtxt(directory / "trajectory.csv", delimiter=",", names=True)
    steps = np.arange(0, len(data), 2)
    dummy = SimpleNamespace(
        data=SimpleNamespace(
            mocap_pos=np.array(
                [[-0.31718881, 0.5, 0.29525084], [0.31718881, 0.49999888, 0.29525084]]
            )
        )
    )
    points = waypoints(
        dummy,
        summary["target"],
        summary["bin"],
        summary["offset_m"],
        summary["estimated_xyz"],
    )
    labels = np.array(
        [action_at(points, min(int(t) + 2, 800)) for t in steps], dtype=np.float32
    )
    goal = np.array(
        [summary["target"] == c for c in ("red", "blue", "green")]
        + [summary["bin"] == b for b in ("left", "right")],
        dtype=np.float32,
    )
    q = np.column_stack(
        [data[f"robot_q{i}"][steps] for i in range(8, 14)]
        + [(data["robot_q14"][steps] - 0.021) / (0.057 - 0.021)]
    )
    states = np.c_[q, np.repeat(goal[None], len(steps), axis=0)].astype(np.float32)
    if not destination.exists():
        frames = []
        with av.open(str(directory / "rollout.mp4")) as video:
            for frame in video.decode(video=0):
                frames.append(
                    np.array(
                        Image.fromarray(frame.to_ndarray(format="rgb24")).resize(
                            (320, 240)
                        )
                    )
                )
        assert len(frames) == len(steps)
        np.savez_compressed(
            destination,
            images=np.array(frames, dtype=np.uint8),
            states=states,
            actions=labels,
            sim_time_s=data["sim_time_s"][steps],
        )
    seed = summary["seed"]
    split = "validation" if seed == 103 else "test" if seed == 104 else "train"
    manifest.append(
        {
            "file": destination.name,
            "source_case": index,
            "seed": seed,
            "target": summary["target"],
            "destination": summary["bin"],
            "frames": len(steps),
            "split": split,
            "group": summary["group"],
        }
    )
    print(destination.name, split, flush=True)
metadata = {
    "episodes": len(manifest),
    "frames": sum(x["frames"] for x in manifest),
    "fps": 25,
    "image_shape": [240, 320, 3],
    "state_dim": 12,
    "state_fields": "right6joint angles+normalized actual gripper+3target onehots+2bin onehots",
    "action_dim": 8,
    "action_fields": "next25Hz end-effector targetxyz(m), quaternionwxyz, desired normalized gripper; interpolated from exact scripted path",
    "source": "scripted physical demonstration, not human teleoperation",
    "split": "by layout seed, paired oracle episodes excluded; stress layout200 in train; validation103 and test104",
    "frame_source": "H264 rendered video decoded; observation after frame step, action labels target next sampled step; control simulator originally50Hz",
    "limitations": "Custom dataset, not converted to LeRobot format; no ACT trained on these data yet; retained initial warmup omitted",
    "items": manifest,
}
(out / "manifest.json").write_text(json.dumps(metadata, indent=2))
print("Export complete", len(manifest))
