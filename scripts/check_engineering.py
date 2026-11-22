"""Run the explicitly maintained code quality baseline without model imports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=root / "repository/engineering_checks.json")
    args = parser.parse_args()
    paths = json.loads((root / "repository/MAINTAINED_ENTRYPOINTS.json").read_text())
    commands = [
        [sys.executable, "-m", "ruff", "check", *paths["format_and_lint"]],
        [sys.executable, "-m", "ruff", "format", "--check", *paths["format_and_lint"]],
        [sys.executable, "-m", "mypy", *paths["strict_typecheck"]],
    ]
    rows = []
    for command in commands:
        process = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=120)
        rows.append(
            {
                "command": command[2:],
                "returncode": process.returncode,
                "stdout": process.stdout,
                "stderr": process.stderr,
            }
        )
        print(command[2:4], process.returncode)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2) + "\n")
    return int(any(row["returncode"] for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
