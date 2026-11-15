import subprocess, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TASK = ROOT / "experiments/desktop_pickplace"
while not (TASK / "runs/goal_act/training_summary.json").exists():
    time.sleep(2)
cmd = [
    str(ROOT / "experiments/desktop_pickplace/scripts/act-python"),
    str(TASK / "scripts/evaluate_policy.py"),
    "--model",
    str(TASK / "runs/goal_act/best_checkpoint"),
    "--out",
    str(TASK / "runs/goal_act_eval"),
]
with (TASK / "runs/goal_act_eval.log").open("w") as f:
    p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
if p.returncode:
    raise RuntimeError("ACT evaluation failed")
