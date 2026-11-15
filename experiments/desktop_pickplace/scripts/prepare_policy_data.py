from pathlib import Path
import json, numpy as np

ROOT = Path(__file__).resolve().parents[1]
data = ROOT / "data/demonstrations"
manifest = json.loads((data / "manifest.json").read_text())
out = ROOT / "data/act_cache"
out.mkdir(exist_ok=True)
total = manifest["frames"]
images = np.lib.format.open_memmap(
    out / "images.npy", mode="w+", dtype=np.uint8, shape=(total, 240, 320, 3)
)
states = []
actions = []
episodes = []
bounds = []
offset = 0
splits = {"train": [], "validation": [], "test": []}
for i, item in enumerate(manifest["items"]):
    with np.load(data / item["file"]) as d:
        n = len(d["images"])
        images[offset : offset + n] = d["images"]
        states.append(d["states"])
        actions.append(d["actions"])
        episodes.extend([i] * n)
    bounds.append([offset, offset + n])
    offset += n
    splits[item["split"]].append(i)
images.flush()
states = np.concatenate(states)
actions = np.concatenate(actions)
episodes = np.array(episodes)
np.save(out / "states.npy", states)
np.save(out / "actions.npy", actions)
np.save(out / "episodes.npy", episodes)
train = np.isin(episodes, splits["train"])
pixel = images[np.flatnonzero(train)[::100]].astype(float) / 255
stats = {
    "observation.state": {
        "mean": states[train].mean(0).tolist(),
        "std": np.maximum(states[train].std(0), 1e-6).tolist(),
    },
    "action": {
        "mean": actions[train].mean(0).tolist(),
        "std": np.maximum(actions[train].std(0), 1e-6).tolist(),
    },
    "observation.images.top": {
        "mean": pixel.mean((0, 1, 2)).reshape(3, 1, 1).tolist(),
        "std": np.maximum(pixel.std((0, 1, 2)), 1e-6).reshape(3, 1, 1).tolist(),
    },
}
(out / "train_stats.json").write_text(json.dumps(stats))
(out / "manifest.json").write_text(
    json.dumps(
        {
            "source": "scripted desktop sorting demonstrations",
            "revision": "2026-10-15",
            "boundaries": bounds,
            "frames": total,
            "train_episodes": splits["train"],
            "validation_episodes": splits["validation"],
            "test_episodes": splits["test"],
        },
        indent=2,
    )
)
print("Policy cache ready", total)
