"""Runtime checks at model, simulator and planner boundaries."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, TypedDict


class GroundingResponse(TypedDict):
    ok: bool
    point_2d: list[float]


def number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def vector(value: Any, length: int, name: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{name} must contain {length} numbers")
    return [number(x, name) for x in value]


def grounding(value: Any) -> GroundingResponse:
    if not isinstance(value, dict) or value.get("ok") is not True:
        raise ValueError(f"Grounding failed: {value!r}")
    point = vector(value.get("point_2d"), 2, "point_2d")
    if not all(0 <= x < 1000 for x in point):
        raise ValueError("point_2d must use normalized coordinates [0, 1000)")
    return {"ok": True, "point_2d": point}


@dataclass(frozen=True)
class ArmState:
    joint_positions: tuple[float, ...]
    finger_positions: tuple[float, ...]
    tcp_p: tuple[float, ...]

    @classmethod
    def parse(cls, value: Any) -> ArmState:
        if not isinstance(value, dict):
            raise ValueError("Arm state must be an object")
        return cls(
            tuple(vector(value.get("joint_positions"), 7, "joint_positions")),
            tuple(vector(value.get("finger_positions"), 2, "finger_positions")),
            tuple(vector(value.get("tcp_p"), 3, "tcp_p")),
        )


def action_success(error_code: int, status: int, executed_points: int) -> None:
    if error_code != 1 or status != 4 or executed_points <= 0:
        raise ValueError(
            f"Execution failed: MoveIt={error_code}, status={status}, points={executed_points}"
        )


def trajectory(names: Any, points: Any) -> None:
    """Validate recorded ROS-like trajectory messages without importing ROS."""
    if not isinstance(names, list) or not names or not all(isinstance(x, str) for x in names):
        raise ValueError("Joint names must be nonempty strings")
    if len(set(names)) != len(names):
        raise ValueError("Duplicate joint names")
    if not isinstance(points, list) or not points:
        raise ValueError("Empty trajectory")
    previous = -1.0
    for point in points:
        if not isinstance(point, dict):
            raise ValueError("Invalid trajectory point")
        vector(point.get("positions"), len(names), "positions")
        stamp = number(point.get("time_from_start"), "time_from_start")
        if stamp < 0 or stamp <= previous:
            raise ValueError("Trajectory time must increase strictly")
        previous = stamp


def collision_geometry(value: Any) -> None:
    from .geometry import rotation

    if not isinstance(value, dict) or not isinstance(value.get("id"), str) or not value["id"]:
        raise ValueError("Collision geometry requires a nonempty id")
    vector(value.get("p"), 3, "geometry position")
    rotation(value.get("R"))
    kind = value.get("type")
    if kind == "mesh":
        vertices = value.get("vertices")
        triangles = value.get("triangles")
        if (
            not isinstance(vertices, list)
            or not vertices
            or not isinstance(triangles, list)
            or not triangles
        ):
            raise ValueError("Mesh needs vertices and triangles")
        for vertex in vertices:
            vector(vertex, 3, "mesh vertex")
        for face in triangles:
            if (
                not isinstance(face, list)
                or len(face) != 3
                or any(type(index) is not int or not 0 <= index < len(vertices) for index in face)
            ):
                raise ValueError("Mesh triangle index out of range")
    else:
        counts = {"box": 3, "sphere": 1, "cylinder": 2}
        if kind not in counts:
            raise ValueError("Unsupported collision geometry")
        if any(x <= 0 for x in vector(value.get("dimensions"), counts[kind], "dimensions")):
            raise ValueError("Collision dimensions must be positive")


def target_pose(value: Any) -> None:
    if not isinstance(value, dict):
        raise ValueError("Target pose must be an object")
    vector(value.get("target_position"), 3, "target position")
    quaternion = vector(value.get("target_orientation"), 4, "target quaternion")
    if abs(sum(x * x for x in quaternion) - 1) > 1e-4:
        raise ValueError("Target quaternion must be unit length")
