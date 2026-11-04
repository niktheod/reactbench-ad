"""Ego-motion projection and the fixed-width speed and acceleration bins of the result tables.

Every scored sample is labelled with its ego speed at the current frame in 5 km/h bins (``[20,25)``) and its
longitudinal acceleration in 0.5 m/s^2 bins; bins are never merged.
"""

from __future__ import annotations

import math
import re
from typing import Any, Optional, Sequence


SPEED_BIN_WIDTH_KMH = 5.0
ACCELERATION_BIN_WIDTH_MPS2 = 0.5
MINIMUM_PROJECTION_SPEED_MPS = 0.01


_INTERVAL_LABEL = re.compile(
    r"^\[\s*(?P<lower>-?\d+(?:\.\d+)?)\s*,\s*"
    r"(?P<upper>-?\d+(?:\.\d+)?)\s*\)$"
)


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite, got {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite, got {value!r}") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return number


def _format_boundary(value: float) -> str:
    value = 0.0 if value == 0.0 else value
    return str(int(value)) if value.is_integer() else f"{value:g}"


def interval_label(lower: float, upper: float) -> str:
    """Return the common lower-inclusive, upper-exclusive interval label."""

    return f"[{_format_boundary(float(lower))},{_format_boundary(float(upper))})"


def _bin_index(value: float, width: float) -> int:
    return math.floor(value / width)


def _base_label(lower_index: int, upper_index: int, width: float) -> str:
    return interval_label(lower_index * width, upper_index * width)


def project_longitudinal_acceleration(
    velocity: Sequence[Any],
    acceleration: Sequence[Any],
    *,
    minimum_speed_mps: float = MINIMUM_PROJECTION_SPEED_MPS,
) -> Optional[float]:
    """Return ``dot(a_xy, v_xy) / ||v_xy||`` above the shared speed floor."""

    try:
        velocity_x = _finite_float(velocity[0], "velocity x")
        velocity_y = _finite_float(velocity[1], "velocity y")
    except (IndexError, KeyError, TypeError) as error:
        raise ValueError(
            f"velocity must contain at least two finite values, got {velocity!r}"
        ) from error
    try:
        acceleration_x = _finite_float(acceleration[0], "acceleration x")
        acceleration_y = _finite_float(acceleration[1], "acceleration y")
    except (IndexError, KeyError, TypeError) as error:
        raise ValueError(
            "acceleration must contain at least two finite values, "
            f"got {acceleration!r}"
        ) from error
    minimum_speed = _finite_float(minimum_speed_mps, "minimum_speed_mps")
    if minimum_speed < 0:
        raise ValueError("minimum_speed_mps must be non-negative")
    speed_mps = math.hypot(velocity_x, velocity_y)
    if speed_mps <= minimum_speed:
        return None
    return (acceleration_x * velocity_x + acceleration_y * velocity_y) / speed_mps


def speed_bin_label(speed_kmh: float) -> str:
    """The 5 km/h bin of a speed, e.g. ``[20,25)``."""

    value = _finite_float(speed_kmh, "speed_kmh")
    if value < 0:
        raise ValueError(f"speed_kmh must be non-negative, got {speed_kmh!r}")
    index = _bin_index(value, SPEED_BIN_WIDTH_KMH)
    return _base_label(index, index + 1, SPEED_BIN_WIDTH_KMH)


def acceleration_bin_label(longitudinal_acceleration_mps2: Optional[float]) -> Optional[str]:
    """The 0.5 m/s^2 bin of a longitudinal acceleration (None stays None)."""

    if longitudinal_acceleration_mps2 is None:
        return None
    value = _finite_float(
        longitudinal_acceleration_mps2, "longitudinal_acceleration_mps2"
    )
    index = _bin_index(value, ACCELERATION_BIN_WIDTH_MPS2)
    return _base_label(index, index + 1, ACCELERATION_BIN_WIDTH_MPS2)


def motion_binning_policy() -> dict[str, Any]:
    """Reporting bins cover all finite measurements without cohort-dependent merging."""
    return {"id": "fixed-width-motion-v1", "speed_width_kmh": SPEED_BIN_WIDTH_KMH,
            "acceleration_width_mps2": ACCELERATION_BIN_WIDTH_MPS2,
            "origin": 0, "interval_convention": "[lower, upper)", "sparse_merge": False}


def observed_motion_labels(rows, field):
    """Numerically order saved bins, including historical frozen/merged labels."""
    return sorted({r.get("groups", {}).get(field) for r in rows
                   if r.get("groups", {}).get(field) is not None}, key=speed_bin_sort_key)


def speed_bin_sort_key(label: str) -> float:
    """Return the lower boundary of a canonical or legacy speed label."""

    match = _INTERVAL_LABEL.fullmatch(label)
    if match is not None:
        return _finite_float(match.group("lower"), "speed-bin lower boundary")
    lower, separator, _ = label.partition("-")
    if not separator:
        raise ValueError(f"Invalid speed-bin label: {label!r}")
    return _finite_float(lower, "speed-bin lower boundary")
