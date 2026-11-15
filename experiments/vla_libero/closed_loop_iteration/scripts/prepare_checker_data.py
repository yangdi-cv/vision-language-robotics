"""CPU-only extraction with episode/init-disjoint splits and step-aligned official labels."""

from pathlib import Path
import concurrent.futures
import hashlib
import json
import numpy as np
from PIL import Image
import imageio.v2 as imageio

R = Path(__file__).resolve().parents[1]
V = R.parent
D = R / "data"
(D / "images").mkdir(exist_ok=True, parents=True)


def extract(path):
    p = Path(path)
    r = json.loads(p.read_text())
    a = np.load(p.parent / "trace.npz")
    n = len(a["predicate"])
    wanted = sorted(
        set([1, max(1, n // 2), max(1, n - 20), max(1, n - 5), max(1, n - 1), n])
    )
    eid = f"{r['suite']}_t{r['task_id']:02d}_i{r['initial_state_id']:02d}"
    frames = {}
    reader = imageio.get_reader(p.parent / "rollout.mp4")
    try:
        for i, frame in enumerate(reader):
            if i == 0 or i in wanted:
                im = Image.fromarray(frame).convert("RGB")
                assert im.size == (256, 256), im.size
                name = f"images/{eid}_{i:04d}.png"
                im.save(D / name)
                frames[i] = {
                    "path": name,
                    "sha256": hashlib.sha256(np.asarray(im).tobytes()).hexdigest(),
                }
            if i >= n:
                break
    finally:
        reader.close()
    assert len(frames) == len(wanted) + 1
    split = {0: "train", 1: "validation", 2: "test"}[r["initial_state_id"]]
    cases = []
    for i in wanted:
        cases.append(
            {
                "id": f"{eid}_s{i:04d}",
                "episode": eid,
                "initial_state_sha256": r["initial_state_sha256"],
                "split": split,
                "suite": r["suite"],
                "task": r["task_id"],
                "init": r["initial_state_id"],
                "step": i,
                "episode_steps": n,
                "instruction": r["original_prompt"],
                "images": [frames[0]["path"], frames[i]["path"]],
                "image_hashes": [frames[0]["sha256"], frames[i]["sha256"]],
                "done": bool(a["predicate"][i - 1]),
                "stratum": "terminal"
                if i == n
                else ("boundary" if i >= n - 5 else "earlier"),
                "source": str(p.relative_to(V)),
                "source_result_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
        )
    return cases


def main():
    target = D / "cases.json"
    if target.exists():
        raise FileExistsError(target)
    paths = sorted((V / "results/baseline").glob("*/*/result.json"))
    assert len(paths) == 120
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        rows = [r for group in ex.map(extract, paths) for r in group]
    episodes = {
        s: {r["episode"] for r in rows if r["split"] == s}
        for s in ["train", "validation", "test"]
    }
    hashes = {
        s: {r["initial_state_sha256"] for r in rows if r["split"] == s}
        for s in episodes
    }
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        assert not episodes[a] & episodes[b]
        assert not hashes[a] & hashes[b]
    summary = {
        "cases": len(rows),
        "splits": {
            s: {
                "cases": sum(r["split"] == s for r in rows),
                "positive": sum(r["split"] == s and r["done"] for r in rows),
                "episodes": len(episodes[s]),
            }
            for s in episodes
        },
        "labels": "official full-task predicate after the exact control step, aligned with frame index; never Qwen answers",
        "image_size": [256, 256],
        "frame_source": "existing baseline videos, same resolution and compression for both labels",
        "limitation": "Retrospective initial-state holdout, tasks and scenes shared; near-threshold truth may not be visually identifiable. This is full-task/goal completion, not arbitrary subgoal labels.",
    }
    target.write_text(json.dumps(rows, indent=2))
    (D / "audit.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
