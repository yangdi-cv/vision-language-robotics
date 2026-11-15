from pathlib import Path
import subprocess
import time

R = Path(__file__).resolve().parents[1]
V = R.parent
while not (R / "data/cases.json").exists():
    time.sleep(3)
for name in ["cases.json", "audit.json", "rejections.json"]:
    (R / "data" / name).rename(R / "archive" / ("pilot_" + name))
with (R / "logs/generate_quality.log").open("w") as f:
    subprocess.run(
        [str(V / "scripts/python_shared"), str(R / "scripts/generate.py")],
        stdout=f,
        stderr=subprocess.STDOUT,
        check=True,
    )
print("Quality-checked generation complete", flush=True)
