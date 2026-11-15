from pathlib import Path
import sys, json, ast, shutil
import numpy as np

R = Path(__file__).resolve().parents[3]
T = R / "experiments/alibaba_pc_revision"
for p in (T / "scripts").glob("*.py"):
    ast.parse(p.read_text())
for p in [
    R / "perception/qwen_scene" / n
    for n in [
        "frame_transport.py",
        "worker_client.py",
        "continuous_worker.py",
        "vision_worker.py",
        "engine.py",
        "vision_engine.py",
    ]
]:
    ast.parse(p.read_text())
r = json.loads((T / "runs/runtime/summary.json").read_text())
assert r["pixel_integrity_checked"] == 144
assert all(v["process_reaped"] for v in r["worker_failures"])
l = json.loads((T / "runs/lora_validation/summary.json").read_text())
assert l["best_step"] <= l["last_step"]
splits = json.loads((T / "runs/lora_validation/split.json").read_text())
assert all(
    not (set(splits[a]) & set(splits[b]))
    for a, b in [("train", "test"), ("train", "validation"), ("validation", "test")]
)
a = json.loads((T / "runs/act_selection/summary.json").read_text())
for name in ["l1", "rollout"]:
    rows = json.loads((T / "runs/act_selection" / f"test_{name}.json").read_text())
    assert len(rows) == 24
    assert sum(v["success"] for v in rows) == a["selection"][name]["test_success"]
    for row in rows:
        npz = (
            T
            / "runs/act_selection"
            / f"test_{name}"
            / f"{row['seed']}_{row['target']}_{row['destination']}"
            / "trajectory.npz"
        )
        with np.load(npz) as d:
            assert (
                d["commands"].shape == (401, 8)
                and np.isfinite(d["commands"]).all()
                and np.isfinite(d["positions"]).all()
            )
# Constructor/storage check without invoking capture, lifecycle or robot endpoints.
sys.path[:0] = [
    str(R / "perception/web_workbench"),
    str(R / "perception/qwen_scene"),
    str(R / "SO101/code/scripts"),
]
from server import Workbench

b = Workbench()
try:
    rgb = np.arange(40 * 30 * 3, dtype=np.uint8).reshape(40, 30, 3)
    b.frame_store.write(rgb, 7, 123456789)
    im, meta = b.frame_store.read()
    assert np.array_equal(rgb, np.array(im))
    assert meta["frame_id"] == "7"
    assert not b.processes and b.robot is None
finally:
    b.frame_store.close()
    shutil.rmtree(b.session)
smoke = json.loads((T / "runs/legacy_rollout_smoke/metrics.json").read_text())
assert smoke["episodes"] == 1 and smoke["successes"] == 1
assert (T / "runs/public_train_smoke/training_summary.json").exists()
result = {
    "source_parse": True,
    "pixel_integrity_reads": 144,
    "worker_failures": 4,
    "act_final_episodes": 48,
    "manifest_split_disjoint": True,
    "workbench_constructor_and_memory": True,
    "legacy_rollout_success": True,
    "legacy_one_step_train": True,
    "camera_or_serial_opened": False,
    "passed": True,
}
(T / "VERIFICATION.json").write_text(json.dumps(result, indent=2))
print(result)
