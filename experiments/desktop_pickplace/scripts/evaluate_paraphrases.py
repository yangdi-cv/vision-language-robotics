from pathlib import Path
import subprocess, os, time

ROOT = Path(__file__).resolve().parents[3]
TASK = ROOT / "experiments/desktop_pickplace"
while not (TASK / "runs/lora_heldout/metrics.json").exists():
    time.sleep(2)
env = dict(os.environ)
for name in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH"):
    env.pop(name, None)
for name, extra in [
    ("base_paraphrases", []),
    ("lora_paraphrases", ["--adapter", str(TASK / "runs/goal_lora/adapter")]),
]:
    cmd = [
        str(ROOT / ".venvs/qwen-scene/bin/python"),
        str(TASK / "scripts/qwen_goals.py"),
        "--cases",
        str(TASK / "data/language/paraphrases.json"),
        "--out",
        str(TASK / "runs" / name),
    ] + extra
    with (TASK / "runs" / (name + ".log")).open("w") as log:
        p = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    if p.returncode:
        raise RuntimeError(name + " failed")
print("Paraphrase eval complete")
