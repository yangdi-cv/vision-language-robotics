"""Explicit per-machine configuration. No credentials belong in this file."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RuntimeConfig:
    root: Path
    values: dict[str, Any]

    def path(self, name: str) -> Path:
        value = os.environ.get(f"VLA_{name.upper()}", self.values.get("paths", {}).get(name))
        if not isinstance(value, str) or not value:
            raise ValueError(f"Missing path configuration: {name}")
        path = Path(value).expanduser()
        return path if path.is_absolute() else self.root / path

    def limit(self, name: str) -> float:
        from .schemas import number

        value = number(self.values["limits"][name], name)
        if value <= 0:
            raise ValueError(f"{name} must be positive")
        return value

    def remote(self, name: str) -> Any:
        value = os.environ.get(
            f"VLA_REMOTE_{name.upper()}", self.values.get("remote", {}).get(name)
        )
        if value is None or value == "":
            raise ValueError(f"Missing remote setting: {name}")
        return value


def load_config(root: Path | None = None) -> RuntimeConfig:
    root = Path(
        os.environ.get("VLA_ROOT", str(root or Path(__file__).resolve().parents[1]))
    ).resolve()
    example = root / "configs/runtime.example.json"
    defaults: dict[str, Any] = json.loads(example.read_text())
    custom = os.environ.get("VLA_CONFIG")
    if custom:
        override = json.loads(Path(custom).expanduser().read_text())
        if not isinstance(override, dict) or override.get("schema_version") != 1:
            raise ValueError("Configuration must be an object with schema_version=1")
        if set(override) - set(defaults):
            raise ValueError("Unknown configuration sections")
        for section, values in override.items():
            if section == "schema_version":
                continue
            if not isinstance(values, dict) or set(values) - set(defaults[section]):
                raise ValueError(f"Unknown or invalid settings in {section}")
            defaults[section].update(values)
    return RuntimeConfig(root, defaults)


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only local configuration check")
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    config = load_config(args.root)
    rows = {
        name: {"path": str(config.path(name)), "exists": config.path(name).exists()}
        for name in config.values["paths"]
    }
    print(
        json.dumps(
            {
                "root": str(config.root),
                "paths": rows,
                "note": "No models, SSH connections or devices were started.",
            },
            indent=2,
        )
    )
    return 0 if all(row["exists"] for row in rows.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
