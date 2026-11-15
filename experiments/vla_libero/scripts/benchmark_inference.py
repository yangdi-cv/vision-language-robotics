"""Interleaved 5/10-step inference benchmark on identical recorded observations."""

from pathlib import Path
import json, time, copy
import numpy as np
from openpi.training import config
from openpi.policies import policy_config

R = Path(__file__).resolve().parents[1]


def main():
    base = policy_config.create_trained_policy(
        config.get_config("pi05_libero"),
        R / "checkpoints/pi05_libero",
        sample_kwargs={"num_steps": 10},
    )
    policies = {10: base, 5: copy.copy(base)}
    policies[5]._sample_kwargs = {"num_steps": 5}
    contexts = []
    for suite in ["libero_spatial", "libero_goal"]:
        for tid in [0, 3, 6]:
            for init in [0, 1]:
                p = R / "results/baseline" / suite / f"task{tid:02d}_init{init:02d}"
                inputs = np.load(p / "query_inputs.npz")
                r = json.loads((p / "result.json").read_text())
                contexts.append(
                    {
                        "suite": suite,
                        "task_id": tid,
                        "init": init,
                        "observation": {
                            "observation/image": inputs["base_rgb"][0],
                            "observation/wrist_image": inputs["wrist_rgb"][0],
                            "observation/state": inputs["state"][0],
                            "prompt": r["original_prompt"],
                        },
                    }
                )
    records = []
    warmup = []
    for n in [10, 5]:
        start = time.monotonic()
        policies[n].infer(
            contexts[0]["observation"], noise=np.zeros((10, 32), np.float32)
        )
        warmup.append({"steps": n, "wall_s": time.monotonic() - start})
    for i, c in enumerate(contexts):
        noise = (
            np.random.default_rng(90000 + i)
            .standard_normal((10, 32))
            .astype(np.float32)
        )
        predictions = {}
        for repeat in range(3):
            order = [10, 5] if (repeat + i) % 2 == 0 else [5, 10]
            for n in order:
                before = time.monotonic()
                result = policies[n].infer(c["observation"], noise=noise)
                duration = time.monotonic() - before
                predictions[n] = result["actions"].copy()
                records.append(
                    {
                        "suite": c["suite"],
                        "task_id": c["task_id"],
                        "initial_state_id": c["init"],
                        "repeat": repeat,
                        "denoising_steps": n,
                        "wall_s": duration,
                    }
                )
        print(
            c["suite"],
            c["task_id"],
            c["init"],
            "chunk MAE",
            float(np.abs(predictions[5] - predictions[10]).mean()),
            flush=True,
        )
    summary = {
        "contexts": len(contexts),
        "repeats": 3,
        "warmup": warmup,
        "timings": {},
        "records": records,
        "note": "Same observations and explicit action noise; interleaved order, no simulator or other policy process. Does not measure real-time robot control.",
    }
    for n in [10, 5]:
        times = [v["wall_s"] for v in records if v["denoising_steps"] == n]
        summary["timings"][str(n)] = {
            "n": len(times),
            "median_s": float(np.median(times)),
            "p95_s": float(np.percentile(times, 95)),
        }
    summary["median_speedup_5_vs_10"] = (
        summary["timings"]["10"]["median_s"] / summary["timings"]["5"]["median_s"]
    )
    (R / "inference_benchmark.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["timings"], indent=2))


if __name__ == "__main__":
    main()
