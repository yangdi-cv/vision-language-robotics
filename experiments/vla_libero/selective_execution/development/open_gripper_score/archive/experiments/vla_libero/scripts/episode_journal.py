"""Durable incremental records; preserves partial episodes, not resumable simulation."""

import json
import os
from pathlib import Path


def json_value(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, default=json_value)
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def append_jsonl(path, value):
    with Path(path).open("a") as f:
        f.write(json.dumps(value, ensure_ascii=False, default=json_value) + "\n")
        f.flush()
        os.fsync(f.fileno())


class EpisodeJournal:
    def __init__(self, directory, record):
        self.directory = Path(directory)
        self.record = record
        self.save()

    def save(self):
        atomic_json(self.directory / "result.json", self.record)

    def event(self, event):
        append_jsonl(self.directory / "events.jsonl", event)
        self.save()

    def step(self, step, trace):
        append_jsonl(self.directory / "trace.jsonl", {"step": step, **trace})
        self.record["steps"] = step
        self.save()
