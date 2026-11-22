"""Content manifests shared by source delivery and verification."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest(root: Path, exclude: set[str] | None = None) -> dict[str, str]:
    excluded = exclude or set()
    result = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        if path.is_symlink():
            raise ValueError(f"Symlink not accepted in delivery: {relative}")
        if path.is_file():
            result[relative] = sha256(path)
    return result


def verify(root: Path, expected: dict[str, str], exclude: set[str] | None = None) -> None:
    actual = manifest(root, exclude)
    if actual != expected:
        missing = sorted(expected.keys() - actual.keys())
        extra = sorted(actual.keys() - expected.keys())
        changed = sorted(
            key for key in actual.keys() & expected.keys() if actual[key] != expected[key]
        )
        raise ValueError(f"Manifest mismatch: missing={missing}, extra={extra}, changed={changed}")
