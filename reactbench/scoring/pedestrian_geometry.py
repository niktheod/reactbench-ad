"""Shared contract for measured pedestrian asset geometry.

The contract deliberately describes the asset in its canonical local root
frame.  Editors transform the 3D bounds with the same root transform used by
the renderer; evaluators project the same bounds onto the declared local
forward/left axes for collision footprints.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Optional, Sequence


PEDESTRIAN_GEOMETRY_PROFILE_VERSION = "pedestrian_gait_union_aabb_v1"
PEDESTRIAN_GEOMETRY_MEASUREMENT_METHOD = (
    "evaluated_render_visible_mesh_vertices_full_gait_union_v1"
)
PEDESTRIAN_GEOMETRY_UNITS = "meters"
PEDESTRIAN_GEOMETRY_COORDINATE_FRAME = "asset_local_root"

_AXES = ("x", "y", "z")
_FORWARD_VECTORS = {
    "+X": (1.0, 0.0),
    "-X": (-1.0, 0.0),
    "+Y": (0.0, 1.0),
    "-Y": (0.0, -1.0),
}


def _json_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be a JSON integer")
    return value


def _finite_interval(value: Any, label: str) -> list[float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
    ):
        raise ValueError(f"{label} must be a two-value interval")
    try:
        lower, upper = (float(value[0]), float(value[1]))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain numeric values") from exc
    if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
        raise ValueError(
            f"{label} must contain finite increasing bounds, got {value!r}"
        )
    return [lower, upper]


def normalize_local_bounds(value: Any) -> dict[str, list[float]]:
    """Validate and JSON-normalize one local 3D axis-aligned box."""

    if not isinstance(value, Mapping):
        raise ValueError("local_bounds_m must be a mapping")
    bounds = {
        axis: _finite_interval(value.get(axis), f"local_bounds_m.{axis}")
        for axis in _AXES
    }
    for axis in ("x", "y"):
        lower, upper = bounds[axis]
        if not lower < 0.0 < upper:
            raise ValueError(
                f"local_bounds_m.{axis} must straddle the asset root, "
                f"got {bounds[axis]!r}"
            )
    z_min, z_max = bounds["z"]
    if abs(z_min) > 0.05 or z_max <= 0.0:
        raise ValueError(
            "local_bounds_m.z must be grounded within 5 cm and extend above "
            f"the root, got {bounds['z']!r}"
        )
    return bounds


def union_local_bounds(
    per_frame_bounds: Iterable[Mapping[str, Sequence[float]]],
) -> dict[str, list[float]]:
    """Return the union of local bounds measured at individual gait frames."""

    normalized = [normalize_local_bounds(value) for value in per_frame_bounds]
    if not normalized:
        raise ValueError("At least one gait-frame bound is required")
    return {
        axis: [
            min(bounds[axis][0] for bounds in normalized),
            max(bounds[axis][1] for bounds in normalized),
        ]
        for axis in _AXES
    }


def make_pedestrian_geometry(
    *,
    asset_id: str,
    local_bounds_m: Mapping[str, Sequence[float]],
    local_forward_axis: str,
    frame_start: int,
    frame_end_exclusive: int,
    measured_mesh_objects: Sequence[str],
    evaluated_vertex_samples: int,
    blender_version: str,
) -> dict[str, Any]:
    """Build one measured sidecar geometry record."""

    value = {
        "profile_version": PEDESTRIAN_GEOMETRY_PROFILE_VERSION,
        "actor_type": "pedestrian",
        "asset_id": str(asset_id),
        "source": "measured_asset_sidecar",
        "units": PEDESTRIAN_GEOMETRY_UNITS,
        "coordinate_frame": PEDESTRIAN_GEOMETRY_COORDINATE_FRAME,
        "local_forward_axis": str(local_forward_axis),
        "local_bounds_m": {
            axis: list(local_bounds_m[axis]) for axis in _AXES
        },
        "measurement_method": PEDESTRIAN_GEOMETRY_MEASUREMENT_METHOD,
        "measurement_frame_range": {
            "start_inclusive": int(frame_start),
            "end_exclusive": int(frame_end_exclusive),
        },
        "measurement_frame_count": int(frame_end_exclusive) - int(frame_start),
        "measured_mesh_objects": [str(name) for name in measured_mesh_objects],
        "measured_mesh_count": len(measured_mesh_objects),
        "evaluated_vertex_samples": int(evaluated_vertex_samples),
        "measurement_blender_version": str(blender_version),
    }
    return normalize_pedestrian_geometry(value, expected_asset_id=str(asset_id))


def normalize_pedestrian_geometry(
    value: Any,
    *,
    expected_asset_id: Optional[str] = None,
) -> dict[str, Any]:
    """Validate the supported measured pedestrian geometry contract.

    Unknown or incomplete versions are rejected intentionally.  This prevents
    edited samples from silently reverting to a guessed legacy footprint.
    """

    if not isinstance(value, Mapping):
        raise ValueError("pedestrian geometry must be a mapping")
    version = str(value.get("profile_version", ""))
    if version != PEDESTRIAN_GEOMETRY_PROFILE_VERSION:
        raise ValueError(
            "Unsupported pedestrian geometry profile_version "
            f"{version!r}; expected {PEDESTRIAN_GEOMETRY_PROFILE_VERSION!r}"
        )
    if str(value.get("actor_type", "")).lower() != "pedestrian":
        raise ValueError("pedestrian geometry actor_type must be 'pedestrian'")
    asset_id_value = value.get("asset_id")
    if not isinstance(asset_id_value, str) or not asset_id_value.strip():
        raise ValueError("pedestrian geometry asset_id must be non-empty")
    asset_id = asset_id_value.strip()
    if expected_asset_id is not None and asset_id != str(expected_asset_id):
        raise ValueError(
            "pedestrian geometry asset_id disagrees with its sidecar: "
            f"{asset_id!r} != {str(expected_asset_id)!r}"
        )
    if value.get("source") != "measured_asset_sidecar":
        raise ValueError(
            "pedestrian geometry source must be 'measured_asset_sidecar'"
        )
    if value.get("units") != PEDESTRIAN_GEOMETRY_UNITS:
        raise ValueError(
            f"pedestrian geometry units must be {PEDESTRIAN_GEOMETRY_UNITS!r}"
        )
    if value.get("coordinate_frame") != PEDESTRIAN_GEOMETRY_COORDINATE_FRAME:
        raise ValueError(
            "pedestrian geometry coordinate_frame must be "
            f"{PEDESTRIAN_GEOMETRY_COORDINATE_FRAME!r}"
        )
    local_forward_axis = value.get("local_forward_axis")
    if local_forward_axis not in _FORWARD_VECTORS:
        raise ValueError(
            "pedestrian geometry local_forward_axis must be one of "
            f"{sorted(_FORWARD_VECTORS)}, got {local_forward_axis!r}"
        )
    bounds = normalize_local_bounds(value.get("local_bounds_m"))
    if value.get("measurement_method") != PEDESTRIAN_GEOMETRY_MEASUREMENT_METHOD:
        raise ValueError(
            "Unsupported pedestrian geometry measurement_method "
            f"{value.get('measurement_method')!r}"
        )

    frame_range = value.get("measurement_frame_range")
    if not isinstance(frame_range, Mapping):
        raise ValueError("measurement_frame_range must be a mapping")
    try:
        frame_start = _json_integer(
            frame_range["start_inclusive"], "measurement frame start"
        )
        frame_end = _json_integer(
            frame_range["end_exclusive"], "measurement frame end"
        )
        frame_count = _json_integer(
            value["measurement_frame_count"], "measurement_frame_count"
        )
    except (KeyError, ValueError) as exc:
        raise ValueError(
            "measurement frame range/count must contain integers"
        ) from exc
    if frame_start < 0 or frame_end <= frame_start:
        raise ValueError(
            f"Invalid measurement frame range [{frame_start}, {frame_end})"
        )
    if frame_count != frame_end - frame_start:
        raise ValueError(
            "measurement_frame_count must cover every frame in the measured "
            f"range, got {frame_count} for [{frame_start}, {frame_end})"
        )

    mesh_objects_value = value.get("measured_mesh_objects")
    if (
        not isinstance(mesh_objects_value, Sequence)
        or isinstance(mesh_objects_value, (str, bytes))
    ):
        raise ValueError("measured_mesh_objects must be a string list")
    if any(not isinstance(name, str) for name in mesh_objects_value):
        raise ValueError("measured_mesh_objects must contain only strings")
    mesh_objects = [name.strip() for name in mesh_objects_value]
    if not mesh_objects or any(not name for name in mesh_objects):
        raise ValueError("measured_mesh_objects must contain non-empty names")
    if len(set(mesh_objects)) != len(mesh_objects):
        raise ValueError("measured_mesh_objects must not contain duplicates")
    try:
        mesh_count = _json_integer(
            value["measured_mesh_count"], "measured_mesh_count"
        )
        vertex_samples = _json_integer(
            value["evaluated_vertex_samples"], "evaluated_vertex_samples"
        )
    except (KeyError, ValueError) as exc:
        raise ValueError(
            "measured_mesh_count and evaluated_vertex_samples must be integers"
        ) from exc
    if mesh_count != len(mesh_objects):
        raise ValueError(
            "measured_mesh_count disagrees with measured_mesh_objects"
        )
    if vertex_samples <= 0:
        raise ValueError("evaluated_vertex_samples must be positive")
    blender_version_value = value.get("measurement_blender_version")
    if (
        not isinstance(blender_version_value, str)
        or not blender_version_value.strip()
    ):
        raise ValueError("measurement_blender_version must be non-empty")
    blender_version = blender_version_value.strip()

    return {
        "profile_version": version,
        "actor_type": "pedestrian",
        "asset_id": asset_id,
        "source": "measured_asset_sidecar",
        "units": PEDESTRIAN_GEOMETRY_UNITS,
        "coordinate_frame": PEDESTRIAN_GEOMETRY_COORDINATE_FRAME,
        "local_forward_axis": local_forward_axis,
        "local_bounds_m": bounds,
        "measurement_method": PEDESTRIAN_GEOMETRY_MEASUREMENT_METHOD,
        "measurement_frame_range": {
            "start_inclusive": frame_start,
            "end_exclusive": frame_end,
        },
        "measurement_frame_count": frame_count,
        "measured_mesh_objects": mesh_objects,
        "measured_mesh_count": mesh_count,
        "evaluated_vertex_samples": vertex_samples,
        "measurement_blender_version": blender_version,
    }


def local_bounds_corners(
    local_bounds_m: Mapping[str, Sequence[float]],
) -> list[list[float]]:
    """Return all eight corners of a validated local 3D box."""

    bounds = normalize_local_bounds(local_bounds_m)
    return [
        [x, y, z]
        for x in bounds["x"]
        for y in bounds["y"]
        for z in bounds["z"]
    ]


def local_forward_vector(local_forward_axis: str) -> tuple[float, float]:
    try:
        return _FORWARD_VECTORS[str(local_forward_axis)]
    except KeyError as exc:
        raise ValueError(
            "local_forward_axis must be one of "
            f"{sorted(_FORWARD_VECTORS)}, got {local_forward_axis!r}"
        ) from exc


def local_forward_angle_rad(local_forward_axis: str) -> float:
    forward_x, forward_y = local_forward_vector(local_forward_axis)
    return math.atan2(forward_y, forward_x)


def pedestrian_footprint_extents(
    geometry: Mapping[str, Any],
) -> dict[str, float]:
    """Project local x/y bounds onto forward/left footprint axes."""

    normalized = normalize_pedestrian_geometry(geometry)
    bounds = normalized["local_bounds_m"]
    forward_x, forward_y = local_forward_vector(
        normalized["local_forward_axis"]
    )
    left_x, left_y = -forward_y, forward_x
    projections = [
        (
            x * forward_x + y * forward_y,
            x * left_x + y * left_y,
        )
        for x in bounds["x"]
        for y in bounds["y"]
    ]
    forward_values = [value[0] for value in projections]
    left_values = [value[1] for value in projections]
    front = max(forward_values)
    rear = -min(forward_values)
    left = max(left_values)
    right = -min(left_values)
    if min(front, rear, left, right) <= 0.0:
        raise ValueError("pedestrian asset root must lie inside its x/y footprint")
    return {
        "front_length_m": float(front),
        "rear_length_m": float(rear),
        "left_width_m": float(left),
        "right_width_m": float(right),
        "length_m": float(front + rear),
        "width_m": float(left + right),
    }
