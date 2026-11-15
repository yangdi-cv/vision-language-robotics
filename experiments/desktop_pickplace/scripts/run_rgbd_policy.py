from pathlib import Path
import subprocess, json

ROOT = Path(__file__).resolve().parents[3]
TASK = ROOT / "experiments/desktop_pickplace"
python = ROOT / "experiments/desktop_pickplace/scripts/act-python"
for script, options, name in [
    (
        "train_policy_rgbd.py",
        ["--steps", "10000", "--out", str(TASK / "runs/goal_act_rgbd")],
        "goal_act_rgbd",
    ),
    (
        "evaluate_policy_rgbd.py",
        [
            "--model",
            str(TASK / "runs/goal_act_rgbd/best_checkpoint"),
            "--out",
            str(TASK / "runs/goal_act_rgbd_eval"),
        ],
        "goal_act_rgbd_eval",
    ),
]:
    with (TASK / "runs" / (name + ".log")).open("w") as log:
        p = subprocess.run(
            [str(python), str(TASK / "scripts" / script), *options],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if p.returncode:
        raise RuntimeError(name + " failed")
(TASK / "runs/rgbd_policy_done.json").write_text(json.dumps({"done": True}))
