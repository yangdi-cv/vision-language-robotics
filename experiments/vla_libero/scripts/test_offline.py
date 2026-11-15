"""Run packaged CPU tests in isolated processes without any model or simulator."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

R = Path(__file__).resolve().parents[1]
ROOT = R.parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("offline_test_results.json"))
    args = ap.parse_args()
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    # The two small upstream image utility files are shipped with their license.
    env["PYTHONPATH"] = os.pathsep.join(
        [str(R / "scripts"), str(R / "third_party/openpi/packages/openpi-client/src")]
    )
    env.pop("PYTHONHOME", None)
    groups = [
        ("scripts", "tests.py"),
        ("recovery_study/scripts", "test_*.py"),
        ("closed_loop_iteration/scripts", "test_*.py"),
        ("checker_v2/scripts", "test_*.py"),
        ("tests", "test_*.py"),
    ]
    results = []
    for directory, pattern in groups:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-s",
                str(R / directory),
                "-p",
                pattern,
            ],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
        results.append(
            dict(
                group=directory,
                returncode=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
            )
        )
        print(directory, "PASS" if proc.returncode == 0 else "FAIL", flush=True)
    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    return int(any(r["returncode"] for r in results))


if __name__ == "__main__":
    sys.exit(main())
