"""Audit all completed interventions, model inputs, paired noise and execution prefixes."""

import hashlib, json
from pathlib import Path
import sys
import numpy as np
import imageio_ffmpeg
from PIL import Image
from analyze import S, V, prefix_checks, episode

sys.path.insert(0, str(V / "scripts"))
from episode_journal import atomic_json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    protocol = json.loads((S / "PROTOCOL.json").read_text())
    root = S / "episodes/primary"
    workspace = V.parents[1]
    assert sha(V / "checkpoints/pi05_libero/download_verified.json") == protocol["model_manifest_sha256"]
    for name, value in protocol["sources"].items():
        assert sha(workspace / name) == value, name
        assert sha(S / "archive/frozen_sources" / name) == value, name
    adapter = workspace / protocol["adapter"]
    for name, value in protocol["adapter_sha256"].items():
        assert sha(adapter / name) == value
    rows = []
    frames = queries = decisions = 0
    replaced = 0
    for path in sorted(root.glob("*/libero_10/task*")):
        r = json.loads((path / "result.json").read_text())
        if "protocol_sha256" not in r:
            continue
        assert r["protocol_sha256"] == sha(S / "PROTOCOL.json")
        assert r["status"] not in ["error", "interrupted", "running"], path
        expected = protocol["plans"][f"{r['task_id']:02d}_{r['initial_state_id']:02d}"][
            "plan"
        ]
        assert (
            r["subgoals"] == expected
            and r["max_steps"] == 520
            and r["confirm"] == 2
            and r["gripper_gate"]
        )
        with np.load(path / "trace.npz") as f:
            trace = {key: f[key] for key in f.files}
        assert len(trace["applied"]) == r["steps"]
        rng = np.random.default_rng(r["noise_seed"])
        executed = []
        qrows = [
            json.loads(x)
            for x in (path / "policy_queries.jsonl").read_text().splitlines()
        ]
        assert len(qrows) == r["policy_queries"]
        for q in qrows:
            with np.load(path / f"policy_inputs/step{q['step']:04d}.npz") as f:
                input = {key: f[key] for key in f.files}
            assert input["image"].shape == input["wrist"].shape == (224, 224, 3)
            assert input["image"].dtype == input["wrist"].dtype == np.uint8
            assert input["state"].shape == (8,) and input["chunk"].shape == (10, 7)
            assert str(input["instruction"]) == expected[q["index"]]
            noise = rng.standard_normal((10, 32)).astype(np.float32)
            assert np.array_equal(noise, input["noise"])
            assert hashlib.sha256(noise.tobytes()).hexdigest() == q["noise_sha256"]
            for key, value in [
                ("observation/image", input["image"]),
                ("observation/wrist_image", input["wrist"]),
                ("observation/state", input["state"]),
            ]:
                assert (
                    hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
                    == q["hashes"][key]
                )
            if q["step"]:
                np.testing.assert_array_equal(
                    input["state"][:3], trace["eef"][q["step"] - 1]
                )
                np.testing.assert_array_equal(
                    input["state"][-2:], trace["gripper"][q["step"] - 1]
                )
            assert np.isfinite(input["chunk"]).all()
            executed.extend(input["chunk"][:5])
            queries += 1
        np.testing.assert_array_equal(
            np.asarray(executed)[: r["steps"]], trace["applied"]
        )
        calls = (
            [
                json.loads(x)
                for x in (path / "monitor_calls.jsonl").read_text().splitlines()
            ]
            if (path / "monitor_calls.jsonl").exists()
            else []
        )
        assert len(calls) == r["monitor_opportunities"]
        assert sum(x["actual_model_call"] for x in calls) == r["monitor_model_calls"]
        streak = 0
        check_events = {e["step"]: e for e in r["events"] if e["kind"] == "check"}
        assert len(check_events) == len(calls)
        for call in calls:
            decisions += 1
            step = call["step"]
            reply = call["reply"]
            streak = streak + 1 if reply.get("ok") and reply.get("done") else 0
            assert check_events[step]["confirmed"] == (streak >= 2)
            assert check_events[step]["done"] == bool(reply.get("done"))
            grip = trace["gripper"][step - 1]
            assert trace["applied"][step - 1, 6] < 0 and grip[0] - grip[1] > 0.07
            assert (step // 5) % 4 == 0
            for name, value in zip(call["original_images"], call["original_pixel_files_sha256"]):
                assert sha(name) == value
            for name, value in zip(call["actual_images"], call["actual_pixel_files_sha256"]):
                assert sha(name) == value
            if r["study_mode"] in ["live_base", "live_lora"]:
                assert call["actual_images"] == call["original_images"]
            if call["actual_model_call"]:
                information = reply["inference"]
                assert information["revision"] == "ebb281ec70b05090aa6165b016eac8ec08e71b17"
                assert hashlib.sha256(information["prompt"].encode()).hexdigest() == information["prompt_sha256"]
                for name, description in zip(call["actual_images"], information["images"], strict=True):
                    with Image.open(name) as image:
                        rgb = image.convert("RGB")
                        assert rgb.size == (512, 512)
                        assert hashlib.sha256(rgb.tobytes()).hexdigest() == description["pixel_sha256"]
            if r["study_mode"] == "frozen_base":
                assert call["actual_images"] == [call["original_images"][0]] * 2
                hashes = call["reply"].get("inference", {}).get("images", [])
                assert (
                    len(hashes) == 2
                    and hashes[0]["pixel_sha256"] == hashes[1]["pixel_sha256"]
                )
                replaced += (
                    call["original_pixel_files_sha256"][0]
                    != call["original_pixel_files_sha256"][1]
                )
            if r["study_mode"] in ["timer150", "hold_first"]:
                assert not call["actual_model_call"] and not call["actual_images"]
                assert call["reply"]["done"] == (
                    r["study_mode"] == "timer150" and step >= 150
                )
        switches = [e for e in r["events"] if e["kind"] == "subgoal_end"]
        for event in switches:
            assert np.array_equal(event["long_task_atoms"], trace["goal_atoms"][event["step"] - 1])
            if event["reason"] == "done":
                assert check_events[event["step"]]["confirmed"]
        if r["study_mode"] == "hold_first":
            assert switches == [] and not r["stage_budget_enabled"]
        else:
            assert r["stage_budget_enabled"]
        decoder = imageio_ffmpeg.read_frames(str(path / "rollout.mp4"))
        metadata = next(decoder)
        n = sum(1 for _ in decoder)
        assert n == r["steps"] + 1, (path, n, r["steps"])
        frames += n
        assert bool(trace["success_predicate"][-1]) == r["success"]
        rows.append(episode(path))
    prefixes = prefix_checks(root, rows)
    assert all(
        x["max_action_difference"] == 0
        and x["policy_inputs_and_noise_exact"]
        and x["initial_state_exact"]
        and x["settled_state_exact"]
        for x in prefixes
    )
    result = dict(
        episodes_verified=len(rows),
        expected=150,
        all_complete=len(rows) == 150,
        frames_decoded=frames,
        queries_verified=queries,
        decisions_verified=decisions,
        frozen_distinct_current_frames_replaced=replaced,
        prefix_pairs=len(prefixes),
        max_prefix_action_difference=max(
            [r["max_action_difference"] for r in prefixes], default=0.0
        ),
        policy_inputs_unchanged_before_switch=True,
        noise_and_action_execution_exact=True,
        image_prompt_metadata_verified=True,
        confirmation_streaks_reconstructed=True,
        original_sources_and_adapter_unchanged=True,
    )
    atomic_json(S / "VERIFICATION.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
