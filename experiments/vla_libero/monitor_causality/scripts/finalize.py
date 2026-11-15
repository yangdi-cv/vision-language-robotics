"""Finish analysis and delivery after the frozen study; never rerun experiments."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

S = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-for-pid", type=int)
    parser.add_argument("--cpu-python", required=True)
    args = parser.parse_args()
    while True:
        status = json.loads((S / "STATUS.json").read_text())
        if status["state"] == "complete":
            assert status["completed"] == 150 and not status["pilot"]
            break
        if not args.wait_for_pid:
            raise RuntimeError("150 episodes have not completed")
        try:
            os.kill(args.wait_for_pid, 0)
        except ProcessLookupError as exc:
            raise RuntimeError("Study exited before completing; inspect full.log") from exc
        time.sleep(10)
    for name in ["analyze", "collect_queries", "verify", "figures", "video", "report"]:
        print(f"Running {name}", flush=True)
        subprocess.run([sys.executable, str(S / f"scripts/{name}.py")], check=True)
    validate = [sys.executable, str(S / "scripts/validate_package.py"), "--python", args.cpu_python]
    subprocess.run(validate + ["--record"], check=True)
    # Include the clean CPU-test result, then recheck the exact final source archive.
    subprocess.run([sys.executable, str(S / "scripts/report.py")], check=True)
    subprocess.run(validate, check=True)
    (S / "FINALIZATION.json").write_text(json.dumps(dict(
        complete=True, episodes=150, cpu_acceptance=True,
        delivery="deliverables/20261113_阿里_视觉检查闭环作用",
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
