"""Frozen policy with explicit initial coordinate-error stress; no training updates."""

from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
TASK = ROOT / "experiments/desktop_pickplace"
for sigma in (5, 10):
    name = f"coordinate_noise_{sigma}mm"
    command = [
        str(TASK / "scripts/act-python"),
        str(TASK / "scripts/evaluate_policy.py"),
        "--model",
        str(TASK / "runs/goal_act_rgbd/best_checkpoint"),
        "--out",
        str(TASK / "runs" / name),
        "--target-position",
        "--coordinate-noise-mm",
        str(sigma),
        "--seed-start",
        "820",
        "--seed-count",
        "2",
    ]
    with (TASK / "runs" / (name + ".log")).open("w") as log:
        p = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if p.returncode:
        raise RuntimeError(name + " failed")
    print(name, "complete", flush=True)
