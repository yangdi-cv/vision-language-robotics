from pathlib import Path
import subprocess
import time

R = Path(__file__).resolve().parents[1]
V = R.parent
while not (R / "results/model/summary.json").exists():
    time.sleep(5)
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
with (R / "logs/closed_loop.log").open("w") as f:
    subprocess.run(
        [str(V / "scripts/python_shared"), str(R / "scripts/evaluate_loop.py")],
        stdout=f,
        stderr=subprocess.STDOUT,
        check=True,
    )
