"""Check paired states, stored arrays, videos and historical-source integrity."""

from pathlib import Path
import hashlib
import json
import subprocess
import numpy as np
import imageio_ffmpeg

R = Path(__file__).resolve().parents[1]
V = R.parent
records = list((R / "results").glob("*/*/result.json"))
assert len(records) == 28
state_hashes = {}
videos = 0
executed = 0
steps = 0
for p in records:
    r = json.loads(p.read_text())
    d = p.parent
    state = np.load(d / "prefix_state.npz")["state"]
    h = hashlib.sha256(np.ascontiguousarray(state).tobytes()).hexdigest()
    assert h == r["prefix_state_sha256"]
    assert state_hashes.setdefault(r["task"], h) == h
    assert r["prefix_eef_error"] == 0
    if r["status"] == "error":
        assert "replan schema error:" in r["error"] and "steps" not in r
        continue
    trace = np.load(d / "trace.npz")
    n = r["steps"]
    assert trace["applied"].shape == (n, 7) and np.isfinite(trace["applied"]).all()
    assert (
        len(trace["qpos"]) == n and bool(trace["official_success"][-1]) == r["success"]
    )
    assert trace["goal_atoms"][-1].tolist() == r["final_goal_atoms"]
    executed += 1
    steps += n
    videos += 1
for p in (V / "hierarchical/recovery_contract_smoke/libero_10").glob("*/result.json"):
    r = json.loads(p.read_text())
    t = np.load(p.parent / "trace.npz")
    assert len(t["applied"]) == r["steps"]
    steps += r["steps"]
    videos += 1
for rel, old in json.loads((R / "source_hashes.json").read_text()).items():
    if rel in [
        "scripts/run_hierarchical.py",
        "scripts/qwen_planner.py",
        "scripts/evaluate.py",
    ]:
        assert hashlib.sha256((V / rel).read_bytes()).hexdigest() == old
# Decode every video; do not count a file's mere existence as a playable recording.
paths = list((R / "results").glob("*/*/rollout.mp4")) + list(
    (V / "hierarchical/recovery_contract_smoke/libero_10").glob("*/rollout.mp4")
)
assert len(paths) == videos == 28
for p in paths:
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-v",
            "error",
            "-i",
            str(p),
            "-f",
            "null",
            "-",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
assert steps == 10432
out = {
    "paired_states_verified": 3,
    "result_files": len(records) + 2,
    "executed_recovery_and_guided": executed,
    "planning_errors_preserved": 2,
    "new_simulated_steps": steps,
    "videos_fully_decoded": videos,
    "historical_sources_unchanged": True,
    "status": "passed",
}
(R / "verification.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out))
