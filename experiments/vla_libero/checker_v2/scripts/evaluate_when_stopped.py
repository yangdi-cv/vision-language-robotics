from pathlib import Path
import os
import subprocess
import time

R = Path(__file__).resolve().parents[1]
ROOT = R.parents[2]
while (
    not (R / "logs/dev_stop_classification.log").exists()
    or not (R / "logs/dev_stop_classification.log").read_text().strip()
):
    time.sleep(2)
while (
    int(
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
    )
    < 22000
):
    time.sleep(2)
env = os.environ.copy()
env["HF_HUB_OFFLINE"] = "1"
with (R / "logs/final_evaluation.log").open("w") as f:
    subprocess.run(
        [
            str(ROOT / ".venvs/qwen-scene/bin/python"),
            str(R / "scripts/final_evaluate.py"),
        ],
        env=env,
        stdout=f,
        stderr=subprocess.STDOUT,
        check=True,
    )
with (R / "logs/closed_loop.log").open("w") as f:
    subprocess.run(
        [str(R.parent / "scripts/python_shared"), str(R / "scripts/evaluate_loop.py")],
        stdout=f,
        stderr=subprocess.STDOUT,
        check=True,
    )
