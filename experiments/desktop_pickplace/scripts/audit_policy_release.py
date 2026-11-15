"""Replay recorded commands to verify release/floor contacts independently of inference."""

import argparse, json
from pathlib import Path
import numpy as np
from task import make, BINS, position, image, locate_rgbd

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--run", required=True)
args = parser.parse_args()
folder = ROOT / "runs" / args.run
rows = []
for path in sorted(folder.glob("*/trajectory.json")):
    data = json.loads(path.read_text())
    result = data["summary"]
    env, p = make(result["seed"])
    positions = []
    contact = []
    for _ in range(50):
        env.step(np.r_[p.data.mocap_pos[1].copy(), [1, 0, 0, 0], 1])
    if (
        "rgbd" in args.run
        or "qwen_act" in args.run
        or json.loads((folder / "metrics.json").read_text()).get(
            "target_position_input", False
        )
    ):
        locate_rgbd(p, result["target"])
    for action in data["commands"]:
        image(p)
        for _ in range(2):
            env.step(np.asarray(action, dtype=np.float32))
        positions.append(position(p, result["target"]).tolist())
        pairs = []
        for c in p.data.contact[: p.data.ncon]:
            names = [p.model.id2name(int(g), "geom") for g in (c.geom1, c.geom2)]
            if result["target"] + "_cube" in names:
                pairs.extend(names)
        contact.append(pairs)
    difference = float(
        np.max(np.abs(np.asarray(positions) - np.asarray(data["positions"])))
    )
    assert difference < 1e-5, (path, difference)
    released = all(
        not any("vx300s_right/10_" in n for n in row) for row in contact[-25:]
    )
    floor = any(result["bin"] + "_floor" in row for row in contact[-25:])
    strict = bool(result["success"] and released and floor)
    rows.append(
        {
            "case": path.parent.name,
            "position_replay_error_max_m": difference,
            "original_success": result["success"],
            "released_last_second": released,
            "bin_floor_contact": floor,
            "strict_success": strict,
        }
    )
    p.free()
summary = {
    "cases": len(rows),
    "original_successes": sum(r["original_success"] for r in rows),
    "strict_successes": sum(r["strict_success"] for r in rows),
    "max_position_replay_error_m": max(r["position_replay_error_max_m"] for r in rows),
    "rows": rows,
}
(folder / "release_audit.json").write_text(json.dumps(summary, indent=2))
print({k: v for k, v in summary.items() if k != "rows"})
