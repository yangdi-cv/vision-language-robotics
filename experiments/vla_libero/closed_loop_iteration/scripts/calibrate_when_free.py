from pathlib import Path
import os
import subprocess
import time

R = Path(__file__).resolve().parents[1]
ROOT = R.parents[2]
start = time.monotonic()
while True:
    complete = (R / "results/checker_lora/summary.json").exists()
    free = int(
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
    )
    if complete and free > 22000:
        break
    if time.monotonic() - start > 3600:
        raise TimeoutError("training/calibration timebox")
    time.sleep(3)
env = os.environ.copy()
env["HF_HUB_OFFLINE"] = "1"
with (R / "logs/calibration.log").open("w") as f:
    subprocess.run(
        [
            str(ROOT / ".venvs/qwen-scene/bin/python"),
            str(R / "scripts/calibrate_checker.py"),
        ],
        env=env,
        stdout=f,
        stderr=subprocess.STDOUT,
        check=True,
    )
