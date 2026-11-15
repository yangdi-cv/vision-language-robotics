"""Add initial RGB-D target coordinates while retaining the same episode split."""

import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
cache = ROOT / "data/act_cache"
out = ROOT / "data/act_cache_rgbd"
out.mkdir(exist_ok=True)
manifest = json.loads((ROOT / "data/demonstrations/manifest.json").read_text())
features = []
for item in manifest["items"]:
    summary = json.loads(
        (
            ROOT / "runs/batch002" / f"case_{item['source_case']:03d}/summary.json"
        ).read_text()
    )
    features.append(
        np.repeat(np.asarray(summary["estimated_xyz"])[None], item["frames"], axis=0)
    )
states = np.column_stack(
    [np.load(cache / "states.npy"), np.concatenate(features)]
).astype(np.float32)
np.save(out / "states.npy", states)
for name in ("images.npy", "actions.npy", "episodes.npy", "manifest.json"):
    destination = out / name
    if not destination.exists():
        destination.symlink_to(Path("../act_cache") / name)
split = json.loads((cache / "manifest.json").read_text())
train = np.isin(np.load(cache / "episodes.npy"), split["train_episodes"])
stats = json.loads((cache / "train_stats.json").read_text())
stats["observation.state"] = {
    "mean": states[train].mean(axis=0).tolist(),
    "std": np.maximum(states[train].std(axis=0), 1e-6).tolist(),
}
(out / "train_stats.json").write_text(json.dumps(stats, indent=2))
print("Coordinate-conditioned states:", states.shape)
