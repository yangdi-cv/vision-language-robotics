"""Install simulation dependencies into the new VLA environment only."""

from pathlib import Path
import subprocess

R = Path(__file__).resolve().parents[1]
W = R.parents[1]
subprocess.run(
    [
        str(R / "tools/bin/uv"),
        "pip",
        "install",
        "--python",
        str(W / ".venvs/openpi-libero/bin/python"),
        "numpy==1.26.4",
        "robosuite==1.4.1",
        "mujoco==3.2.3",
        "dm-control==1.0.24",
        "gym-aloha==0.1.5",
        "bddl==1.0.1",
        "easydict==1.9",
        "gym==0.25.2",
        "future==0.18.2",
        "-e",
        str(R / "third_party/libero"),
    ],
    check=True,
)
