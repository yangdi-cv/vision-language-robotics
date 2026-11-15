from pathlib import Path
import os
import subprocess
import time

R = Path(__file__).resolve().parents[1]
ROOT = R.parents[2]
while (
    not (R / "data/audit.json").exists()
    or not (R / "archive/pilot_cases.json").exists()
):
    time.sleep(3)
# Completion of quality generator is signalled by its queue process only after audit.json is written.
with (R / "logs/audit.log").open("w") as f:
    subprocess.run(
        [
            str(ROOT / ".venvs/openpi-libero/bin/python"),
            str(R / "scripts/audit_data.py"),
        ],
        stdout=f,
        stderr=subprocess.STDOUT,
        check=True,
    )
env = os.environ.copy()
env["HF_HUB_OFFLINE"] = "1"
with (R / "logs/train.log").open("w") as f:
    subprocess.run(
        [str(ROOT / ".venvs/qwen-scene/bin/python"), str(R / "scripts/train.py")],
        env=env,
        stdout=f,
        stderr=subprocess.STDOUT,
        check=True,
    )
