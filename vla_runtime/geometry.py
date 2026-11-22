"""Pure geometry guards shared by ROS adapters and offline tests."""

from __future__ import annotations

import math
from typing import Any
from .schemas import number, vector


def rotation(value: Any) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("Rotation must be 3x3")
    rows = [vector(row, 3, "rotation") for row in value]
    for i in range(3):
        for j in range(3):
            if abs(sum(rows[k][i] * rows[k][j] for k in range(3)) - (i == j)) > 1e-5:
                raise ValueError("Rotation is not orthonormal")
    a, b, c = rows
    det = (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )
    if abs(det - 1) > 1e-5:
        raise ValueError("Rotation must have determinant +1")
    return rows


def local_point(base_rotation: Any, base_position: Any, world_position: Any) -> list[float]:
    rows = rotation(base_rotation)
    base = vector(base_position, 3, "base_position")
    world = vector(world_position, 3, "world_position")
    return [sum(rows[k][i] * (world[k] - base[k]) for k in range(3)) for i in range(3)]


def fk_error(expected: Any, actual: Any, tolerance: float = 1e-5) -> float:
    expected = vector(expected, 3, "physics FK")
    actual = vector(actual, 3, "MoveIt FK")
    tolerance = number(tolerance, "FK tolerance")
    if tolerance <= 0:
        raise ValueError("FK tolerance must be positive")
    error = math.sqrt(sum((a - b) ** 2 for a, b in zip(expected, actual)))
    if error > tolerance:
        raise ValueError(f"MoveIt/physics FK mismatch: {error:.8g} m > {tolerance:g} m")
    return error


def collision_matrix(names: list[str], allowed_pairs: set[frozenset[str]]) -> list[list[bool]]:
    if not names or len(names) != len(set(names)) or not all(names):
        raise ValueError("Invalid collision link names")
    known = set(names)
    if any(len(pair) != 2 or not pair <= known for pair in allowed_pairs):
        raise ValueError("Allowed collision pair references an unknown or repeated link")
    return [[a == b or frozenset((a, b)) in allowed_pairs for b in names] for a in names]
