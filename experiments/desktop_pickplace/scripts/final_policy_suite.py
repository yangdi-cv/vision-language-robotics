"""Freeze trained models and compare on new layouts800-809, without further tuning."""

from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
TASK = ROOT / "experiments/desktop_pickplace"
for name, model, extra in [
    ("policy_test60_rgb", TASK / "runs/goal_act/best_checkpoint", []),
    (
        "policy_test60_rgbd",
        TASK / "runs/goal_act_rgbd/best_checkpoint",
        ["--target-position"],
    ),
]:
    command = [
        str(ROOT / "experiments/desktop_pickplace/scripts/act-python"),
        str(TASK / "scripts/evaluate_policy.py"),
        "--model",
        str(model),
        "--out",
        str(TASK / "runs" / name),
        "--seed-start",
        "800",
        "--seed-count",
        "10",
        *extra,
    ]
    with (TASK / "runs" / (name + ".log")).open("w") as log:
        p = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if p.returncode:
        raise RuntimeError(name + " failed")
    print(name, "complete", flush=True)
