from pathlib import Path
import os
import subprocess

R = Path(__file__).resolve().parents[1]
V = R.parent
ROOT = R.parents[2]
with (R / "logs/generate_final.log").open("w") as f:
    subprocess.run(
        [str(V / "scripts/python_shared"), str(R / "scripts/generate.py")],
        stdout=f,
        stderr=subprocess.STDOUT,
        check=True,
    )
with (R / "logs/audit_final.log").open("w") as f:
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
