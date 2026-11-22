"""Atomic snapshots and durable append-only progress, with stderr logging."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(name).unlink(missing_ok=True)


def error_record(exc: BaseException) -> dict[str, str]:
    return {
        "error": str(exc),
        "error_type": type(exc).__name__,
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    }


def logger(name: str) -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        handler = logging.StreamHandler()  # stderr: stdout may be a JSON protocol.
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        log.addHandler(handler)
        log.setLevel(logging.INFO)
        log.propagate = False
    return log


class Journal:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents silently appending to an earlier experiment.
        self.stream = path.open("x", encoding="utf-8")
        self.sequence = 0

    def append(self, phase: str, **fields: Any) -> None:
        self.sequence += 1
        value = {
            "schema_version": 1,
            "sequence": self.sequence,
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "phase": phase,
            **fields,
        }
        self.stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def close(self) -> None:
        self.stream.close()


def run_metadata(root: Path, *, parameters: dict[str, Any], **models: Any) -> dict[str, Any]:
    def git(*args: str) -> str:
        process = subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, timeout=5, check=False
        )
        return process.stdout.strip() if process.returncode == 0 else "unavailable"

    canonical = json.dumps(parameters, sort_keys=True, allow_nan=False).encode()
    return {
        "schema_version": 1,
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "commit": git("rev-parse", "HEAD"),
        "working_tree": git("status", "--porcelain"),
        "parameters": parameters,
        "parameters_sha256": hashlib.sha256(canonical).hexdigest(),
        "models": models,
    }
