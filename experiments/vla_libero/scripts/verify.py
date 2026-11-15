"""Re-read simulator records and check control timing, action scheduling and scoring."""

from pathlib import Path
import json, numpy as np, hashlib

R = Path(__file__).resolve().parents[1]


def main():
    count = 0
    samples = 0
    paired = {}
    errors = []
    for p in sorted((R / "results").rglob("result.json")):
        out = p.parent
        r = json.loads(p.read_text())
        c = json.loads((out / "config.json").read_text())
        assert (
            c["runner_sha256"]
            == hashlib.sha256((R / "scripts/evaluate.py").read_bytes()).hexdigest()
        ), "Runner changed during evaluation"
        if r["status"] == "error":
            errors.append(str(out.relative_to(R)))
            continue
        x = np.load(out / "trace.npz")
        q = np.load(out / "action_chunks.npz")["actions"]
        queries = json.loads((out / "queries.json").read_text())
        inputs = np.load(out / "query_inputs.npz")
        n = len(x["command"])
        assert n == r["steps"] and all(np.isfinite(x[k]).all() for k in x.files)
        assert not c["initial_success_predicate"]
        limited = np.asarray(c["actuator_control_limited"], dtype=bool)
        bounds = np.asarray(c["actuator_control_range"])
        assert np.all(x["ctrl"][:, limited] >= bounds[limited, 0] - 1e-6)
        assert np.all(x["ctrl"][:, limited] <= bounds[limited, 1] + 1e-6)
        assert (
            c["mocap_count"] == 0 and c["actuator_count"] > 0 and c["control_hz"] == 20
        )
        np.testing.assert_allclose(np.diff(x["sim_time"]), 0.05, atol=1e-9, rtol=0)
        assert bool(x["predicate"][-1]) == r["success"]
        expected = np.concatenate([chunk[: c["replan_steps"]] for chunk in q])[:n]
        np.testing.assert_allclose(expected, x["command"], atol=0, rtol=0)
        if r["mode"] == "action_delay":
            np.testing.assert_array_equal(
                x["applied"][:2], np.tile([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], (2, 1))
            )
            np.testing.assert_array_equal(x["applied"][2:], x["command"][:-2])
        else:
            np.testing.assert_array_equal(x["command"], x["applied"])
        assert [v["step"] for v in queries] == list(range(0, n, c["replan_steps"]))
        assert q.shape == (len(queries), 10, 7)
        if r["mode"] == "blank_images":
            assert not inputs["base_rgb"].any() and not inputs["wrist_rgb"].any()
        if r["mode"] == "frozen_images":
            assert np.all(inputs["base_rgb"] == inputs["base_rgb"][0])
            assert np.all(inputs["wrist_rgb"] == inputs["wrist_rgb"][0])
        key = (r["suite"], r["task_id"], r["initial_state_id"])
        h = c["post_wait_qpos_sha256"]
        if key in paired:
            assert paired[key] == h, "Paired trial starts differ"
        paired[key] = h
        count += 1
        samples += n
    result = {
        "checked_episodes": count,
        "control_samples": samples,
        "error_directories": errors,
        "control_timing_checked": True,
        "chunk_execution_checked": True,
        "paired_initial_states_checked": True,
        "official_predicate_checked": True,
        "mocap_absent": True,
        "note": "Checks numerical records and interfaces, not real robot safety or benchmark equivalence.",
    }
    (R / "verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
