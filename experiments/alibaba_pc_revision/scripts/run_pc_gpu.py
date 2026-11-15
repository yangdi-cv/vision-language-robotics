from pathlib import Path
import subprocess, os, json, time

ROOT = Path(__file__).resolve().parents[3]
T = ROOT / "experiments/alibaba_pc_revision"
while not (T / "runs/lora_validation/summary.json").exists():
    time.sleep(2)
# No overlapping GPU training or contact rendering during measured inference.
while not (
    ROOT / "experiments/bytedance_week3/runs/contact_suite/no_contact_check.json"
).exists():
    time.sleep(2)
for script in ["bench_qwen.py", "bench_vision.py"]:
    with (T / "runs" / f"{script}.log").open("w") as f:
        subprocess.run(
            [str(ROOT / ".venvs/qwen-scene/bin/python"), str(T / "scripts" / script)],
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
        )
with (T / "runs/act_training.log").open("w") as f:
    subprocess.run(
        [
            str(ROOT / "experiments/desktop_pickplace/scripts/act-python"),
            str(T / "scripts/train_act.py"),
        ],
        stdout=f,
        stderr=subprocess.STDOUT,
        check=True,
    )
