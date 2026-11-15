"""Validate all three acceptance points and exported inference artifacts."""

from pathlib import Path
import hashlib
import json
import subprocess
import numpy as np
import imageio_ffmpeg


def main():
    R = Path(__file__).resolve().parents[1]
    rows = json.loads((R / "injection_results.json").read_text())
    assert len(rows) == 27
    for t in [0, 3, 4]:
        for init in [0, 1, 2]:
            g = [x for x in rows if x["task"] == t and x["init"] == init]
            assert len(g) == 3 and len({x["perturbed_state_sha256"] for x in g}) == 1
    for r in rows:
        p = R / f"results/injection/task{r['task']:02d}_init{r['init']:02d}" / r["mode"]
        state = np.load(p / "perturbed_state.npz")["state"]
        assert (
            hashlib.sha256(state.tobytes()).hexdigest() == r["perturbed_state_sha256"]
        )
        trace = np.load(p / "trace.npz")
        assert (
            trace["applied"].shape == (r["steps"], 7)
            and np.isfinite(trace["applied"]).all()
        )
        assert bool(trace["official_success"][-1]) == r["success"]
        subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-v",
                "error",
                "-i",
                str(p / "rollout.mp4"),
                "-f",
                "null",
                "-",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    assert len(json.loads((R / "t9_results.json").read_text())) == 4
    out = R / "results/checker_lora"
    protocol = json.loads((out / "protocol.json").read_text())
    assert not set(protocol["train_ids"]) & set(protocol["test_ids"]) and not set(
        protocol["validation_ids"]
    ) & set(protocol["test_ids"])
    for n in ["base_test", "adapter_test"]:
        a = json.loads((out / (n + ".json")).read_text())
        assert len(a) == 240 and len({r["id"] for r in a}) == 240
    assert (out / "best_adapter/adapter_model.safetensors").stat().st_size > 1_000_000
    assert json.loads((out / "reload_check.json").read_text())["fresh_process_load"]
    result = {
        "status": "passed",
        "t9_cases": 4,
        "paired_injection_states": 9,
        "injection_episodes": 27,
        "videos_decoded": 27,
        "independent_test_cases_per_model": 240,
        "lora_weights_reload": True,
    }
    (R / "verification.json").write_text(json.dumps(result, indent=2))
    print(result)


if __name__ == "__main__":
    main()
