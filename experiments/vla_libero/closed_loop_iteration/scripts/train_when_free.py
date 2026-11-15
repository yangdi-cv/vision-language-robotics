"""Wait without allocating GPU memory; start LoRA only after injection experiments exit."""

from pathlib import Path
import json
import os
import subprocess
import time

R = Path(__file__).resolve().parents[1]
ROOT = R.parents[2]
start = time.monotonic()
while True:
    p = R / "injection_results.json"
    complete = p.exists() and len(json.loads(p.read_text())) == 27
    free = int(
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
    )
    if complete and free > 22000:
        break
    if time.monotonic() - start > 7200:
        raise TimeoutError("GPU queue time box exceeded")
    time.sleep(5)
env = os.environ.copy()
env["HF_HUB_OFFLINE"] = "1"
with (R / "logs/checker_train.log").open("w") as f:
    subprocess.run(
        [
            str(ROOT / ".venvs/qwen-scene/bin/python"),
            str(R / "scripts/train_checker.py"),
        ],
        env=env,
        stdout=f,
        stderr=subprocess.STDOUT,
        check=True,
    )
print("Checker training and independent evaluation completed", flush=True)
