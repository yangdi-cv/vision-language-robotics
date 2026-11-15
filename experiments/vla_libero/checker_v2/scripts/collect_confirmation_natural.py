# ruff: noqa: E402
"""Twelve predeclared new-initial-state policy episodes, then physical held/release labels."""

from pathlib import Path
import json
import sys
import numpy as np
from PIL import Image

R = Path(__file__).resolve().parents[1]
V = R.parent
sys.path[:0] = [str(V / "scripts"), str(R / "scripts")]
import evaluate as ev
from generate import env_for, reset, grasp
from run_hierarchical import monitor_view
from libero.libero import benchmark


def main():
    target = R / "confirmation/data/natural.json"
    if target.exists():
        raise FileExistsError(target)
    from openpi.training import config
    from openpi.policies import policy_config

    policy = policy_config.create_trained_policy(
        config.get_config("pi05_libero"),
        V / "checkpoints/pi05_libero",
        sample_kwargs={"num_steps": 10},
    )
    cases = [
        (s, t, i)
        for s in ["libero_object", "libero_spatial"]
        for t in [0, 3, 6]
        for i in [11, 12]
    ]
    for s, t, i in cases:
        out = R / f"confirmation/rollouts/{s}/task{t:02d}_init{i:02d}"
        if not (out / "result.json").exists():
            ev.trial(policy, s, t, i, "baseline", out, record_video=False)
    rows = []
    for s, t, i in cases:
        suite = benchmark.get_benchmark_dict()[s]()
        e = env_for(suite, t)
        try:
            reset(e, suite, t, i)
            atoms = e.env.parsed_problem["goal_state"]
            if (
                len(atoms) != 1
                or atoms[0][0] not in ["on", "in"]
                or atoms[0][1] not in e.env.objects_dict
            ):
                continue
            name = atoms[0][1]
            group = f"confirm_natural_{s}_t{t:02d}_i{i:02d}"
            images = R / "confirmation/data/images"
            images.mkdir(parents=True, exist_ok=True)

            def save(im, suffix):
                p = images / (group + suffix + ".png")
                Image.fromarray(im).save(p)
                return str(p.relative_to(R / "confirmation/data"))

            reference = save(monitor_view(e, 512), "_ref")
            src = R / f"confirmation/rollouts/{s}/task{t:02d}_init{i:02d}"
            result = json.loads((src / "result.json").read_text())
            tr = np.load(src / "trace.npz")
            held = []
            for step, a in enumerate(tr["applied"]):
                e.step(a.tolist())
                if step % 5 == 0 and grasp(e, name):
                    held.append((step, monitor_view(e, 512), bool(e.check_success())))
            selected = [held[0], held[-1]] if len(held) > 1 else held
            for j, (step, im, rel) in enumerate(selected):
                rows.append(
                    {
                        "id": group + f"_held{j}",
                        "group": group,
                        "split": "confirmation",
                        "source": "natural",
                        "instruction": result["original_prompt"],
                        "images": [reference, save(im, f"_held{j}")],
                        "status": "no",
                        "done": False,
                        "relation_predicate": rel,
                        "grasp": True,
                        "kind": "held_unreleased",
                    }
                )
            for _ in range(20):
                e.step(ev.DUMMY.tolist())
            if e.check_success() and not grasp(e, name):
                rows.append(
                    {
                        "id": group + "_released",
                        "group": group,
                        "split": "confirmation",
                        "source": "natural",
                        "instruction": result["original_prompt"],
                        "images": [reference, save(monitor_view(e, 512), "_released")],
                        "status": "yes",
                        "done": True,
                        "grasp": False,
                        "kind": "natural_released",
                    }
                )
        finally:
            e.close()
    target.write_text(json.dumps(rows, indent=2))
    print(
        json.dumps(
            {"new_policy_episodes": 12, "natural_confirmation_cases": len(rows)}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
