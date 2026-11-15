"""Index actual decision frames; official relation labels are not visibility judgments."""

import json
import hashlib
from pathlib import Path
import sys
from PIL import Image
import numpy as np
from analyze import S, V, truth

sys.path.insert(0, str(V / "scripts"))
from episode_journal import atomic_json


def pixels(path):
    with Image.open(path) as image:
        return hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()


def main():
    root = S / "episodes/primary"
    samples = {}
    counts = dict(calls=0, label_conflicts=0)
    for path in sorted(root.glob("*/libero_10/task*")):
        if (
            not (path / "monitor_calls.jsonl").exists()
            or not (path / "trace.npz").exists()
        ):
            continue
        result = json.loads((path / "result.json").read_text())
        if result["status"] == "running" or "protocol_sha256" not in result:
            continue
        with np.load(path / "trace.npz") as f:
            atoms = f["goal_atoms"]
        for line in (path / "monitor_calls.jsonl").read_text().splitlines():
            row = json.loads(line)
            counts["calls"] += 1
            reference, current = [Path(p) for p in row["original_images"]]
            pair = [pixels(reference), pixels(current)]
            key = hashlib.sha256(
                (row["subgoal"] + "|" + ":".join(pair)).encode()
            ).hexdigest()
            label = truth(result["task_id"], atoms[row["step"] - 1])
            observation = dict(
                mode=result["study_mode"],
                task=result["task_id"],
                initial=result["initial_state_id"],
                step=row["step"],
                model_called=row["actual_model_call"],
                prediction=row["reply"].get("done"),
                valid=row["reply"].get("ok"),
                checker=row["reply"].get("checker"),
                sent_images=row["actual_images"],
            )
            if key not in samples:
                samples[key] = dict(
                    id=key,
                    group=f"task{result['task_id']:02d}_init{result['initial_state_id']:02d}",
                    instruction=row["subgoal"],
                    images=[str(p.relative_to(V)) for p in [reference, current]],
                    pixel_sha256=pair,
                    official_relation_label=label,
                    label_source="recorded official final-task atom for first subgoal; task5 standalone pickup unscorable",
                    physical_support_label=None,
                    observability="not_reviewed",
                    review_status="pending",
                    split="development_collection",
                    observations=[],
                )
            elif samples[key]["official_relation_label"] != label:
                samples[key]["label_conflict"] = True
            samples[key]["observations"].append(observation)
    rows = list(samples.values())
    counts.update(
        unique_pairs=len(rows),
        positive=sum(r["official_relation_label"] is True for r in rows),
        negative=sum(r["official_relation_label"] is False for r in rows),
        unscorable=sum(r["official_relation_label"] is None for r in rows),
        label_conflicts=sum(r.get("label_conflict", False) for r in rows),
        independent_test=False,
    )
    directory = S / "decision_data"
    directory.mkdir(exist_ok=True)
    atomic_json(directory / "cases.json", rows)
    atomic_json(directory / "summary.json", counts)
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
