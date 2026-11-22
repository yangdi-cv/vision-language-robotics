"""Export the current source and required offline inputs, without weights or videos."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile

from vla_runtime.artifacts import manifest
from vla_runtime.records import atomic_json


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    names = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=root, text=True
    ).splitlines()
    allowed = {
        ".py",
        ".sh",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".xml",
        ".urdf",
        ".xacro",
        ".srdf",
        ".cfg",
        ".ini",
        ".txt",
        ".md",
        ".csv",
        ".html",
        ".msg",
        ".srv",
        ".action",
        ".cmake",
        ".npz",
    }
    excluded_roots = {"reports", "SO101", ".github"}
    excluded_names = {"MANIFEST.sha256.json", "SOURCE_VERSION.json"}
    # Password-based one-off deployment drivers remain historical source, not supported entrypoints.
    historical_remote = set()
    for name in names:
        path = root / name
        if name.startswith("experiments/mobile_qwen/jetson/") and path.suffix in {".py", ".sh"}:
            if any(token in path.read_text() for token in ("sshpass", "SSHPASS")):
                historical_remote.add(name)
    with tempfile.TemporaryDirectory(prefix="vla export ") as directory:
        stage = Path(directory) / "vision-language-robotics"
        stage.mkdir()
        for name in sorted(set(names)):
            path = Path(name)
            if (
                path.parts[0] in excluded_roots
                or name in historical_remote
                or name in excluded_names
            ):
                continue
            if path.suffix not in allowed and not path.name.startswith(("LICENSE", "NOTICE")):
                if name not in {
                    "experiments/mobile_vla/scripts/ros-python",
                    "experiments/vla_libero/scripts/python_shared",
                }:
                    continue
            source = root / path
            if not source.is_file() or source.is_symlink():
                continue
            target = stage / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        atomic_json(
            stage / "SOURCE_VERSION.json",
            {
                "commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=root, text=True
                ).strip(),
                "working_tree": subprocess.check_output(
                    ["git", "status", "--porcelain"], cwd=root, text=True
                ),
                "historical_remote_drivers_excluded": sorted(historical_remote),
                "scope": "Source and required offline fixtures; no weights, PDFs or videos. See docs/ENGINEERING_HARDENING.md.",
            },
        )
        atomic_json(stage / "MANIFEST.sha256.json", manifest(stage, {"MANIFEST.sha256.json"}))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.out, "x", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(stage.parent))
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
