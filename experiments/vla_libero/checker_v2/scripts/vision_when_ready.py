from pathlib import Path
import json
import os
import subprocess
import time

R = Path(__file__).resolve().parents[1]
V = R.parent
ROOT = R.parents[2]
while not (R / "results/model_vision/development_summary.json").exists():
    time.sleep(4)
d = json.loads((R / "results/model_vision/development_summary.json").read_text())
if not d["dev_pass"]:
    print(
        "Vision development did not pass validation; confirmation remains unscored",
        flush=True,
    )
    raise SystemExit(0)
while not (R / "confirmation/data/audit.json").exists():
    time.sleep(3)
while (
    int(
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
    )
    < 22000
):
    time.sleep(3)
with (R / "logs/confirmation_natural.log").open("w") as f:
    subprocess.run(
        [
            str(V / "scripts/python_shared"),
            str(R / "scripts/collect_confirmation_natural.py"),
        ],
        stdout=f,
        stderr=subprocess.STDOUT,
        check=True,
    )
env = os.environ.copy()
env["HF_HUB_OFFLINE"] = "1"
with (R / "logs/confirmation_evaluation.log").open("w") as f:
    subprocess.run(
        [
            str(ROOT / ".venvs/qwen-scene/bin/python"),
            str(R / "scripts/confirmation_evaluate.py"),
        ],
        env=env,
        stdout=f,
        stderr=subprocess.STDOUT,
        check=True,
    )
with (R / "logs/closed_loop.log").open("w") as f:
    subprocess.run(
        [str(V / "scripts/python_shared"), str(R / "scripts/evaluate_loop.py")],
        stdout=f,
        stderr=subprocess.STDOUT,
        check=True,
    )
