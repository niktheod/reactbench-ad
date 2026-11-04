"""Canonical ego-motion projection and dataset-level result bins.

PAI-AV collision reporting uses unmerged fixed-width labels. Frozen specs
remain available for historical evaluations. The dynamic
helpers remain available for distribution generation and small standalone
analyses, and deliberately implement the deterministic sparse-bin merge used
when the frozen artifacts were created.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


SPEED_BIN_WIDTH_KMH = 5.0
ACCELERATION_BIN_WIDTH_MPS2 = 0.5
MIN_BIN_SAMPLE_COUNT = 10
MINIMUM_PROJECTION_SPEED_MPS = 0.01

REPO_ROOT = Path(__file__).resolve().parents[2]
BIN_SPEC_DIR = REPO_ROOT / "data" / "motion_bins"
NAVSIM_BIN_SPEC_ID = "navsim-navtest-filtered-plus-5-v1"
PAIAV_BIN_SPEC_ID = "paiav-5500-plus5-exact-5.1s-v1"

_SPEC_FILENAMES = {
    NAVSIM_BIN_SPEC_ID: "navsim_navtest_filtered_plus_5_v1.json",
    PAIAV_BIN_SPEC_ID: "paiav_5500_plus5_exact_5p1s_v1.json",
}
_SPEC_ALIASES = {
    "navsim": NAVSIM_BIN_SPEC_ID,
    "navtest_filtered_plus_5": NAVSIM_BIN_SPEC_ID,
    "paiav": PAIAV_BIN_SPEC_ID,
    "pai-av": PAIAV_BIN_SPEC_ID,
    "pai_av": PAIAV_BIN_SPEC_ID,
    "paiav_edited_images": PAIAV_BIN_SPEC_ID,
}
_INTERVAL_LABEL = re.compile(
    r"^\[\s*(?P<lower>-?\d+(?:\.\d+)?)\s*,\s*"
    r"(?P<upper>-?\d+(?:\.\d+)?)\s*\)$"
)


@dataclass(frozen=True)
class NumericBin:
    lower: float
    upper: float
    source_count: int

    @property
    def label(self) -> str:
        return interval_label(self.lower, self.upper)


@dataclass(frozen=True)
class MotionBinSpec:
    """Validated frozen bins and their immutable artifact identity."""

    schema_version: int
    spec_id: str
    dataset: str
    artifact_path: Path
    artifact_sha256: str
    speed_bins: tuple[NumericBin, ...]
    acceleration_bins: tuple[NumericBin, ...]
    document: Mapping[str, Any]

    def metadata(self) -> dict[str, Any]:
        try:
            relative_path = self.artifact_path.relative_to(REPO_ROOT)
        except ValueError:
            relative_path = self.artifact_path
        return {
            "id": self.spec_id,
            "dataset": self.dataset,
            "schema_version": self.schema_version,
            "artifact_path": str(relative_path),
            "artifact_sha256": self.artifact_sha256,
        }


@dataclass
class _BinGroup:
    lower_index: int
    upper_index: int
    records: list[Mapping[str, Any]]


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


def _dimension_bins(document: Mapping[str, Any], name: str) -> tuple[NumericBin, ...]:
    dimensions = document.get("dimensions")
    dimension = dimensions.get(name) if isinstance(dimensions, Mapping) else None
    raw_bins = dimension.get("bins") if isinstance(dimension, Mapping) else None
    if not isinstance(raw_bins, list) or not raw_bins:
        raise ValueError(f"Motion-bin spec dimension {name!r} has no bins")

    bins: list[NumericBin] = []
    for position, raw_bin in enumerate(raw_bins):
        if not isinstance(raw_bin, Mapping):
            raise ValueError(f"Motion-bin spec {name} bin {position} is not an object")
        lower = _finite_float(raw_bin.get("lower"), f"{name} bin lower")
        upper = _finite_float(raw_bin.get("upper"), f"{name} bin upper")
        source_count = raw_bin.get("source_count")
        if (
            not isinstance(source_count, int)
            or isinstance(source_count, bool)
            or source_count < 0
        ):
            raise ValueError(
                f"Motion-bin spec {name} bin {position} has invalid source_count"
            )
        if lower >= upper:
            raise ValueError(f"Motion-bin spec {name} bin {position} is not increasing")
        label = raw_bin.get("label")
        expected_label = interval_label(lower, upper)
        if label != expected_label:
            raise ValueError(
                f"Motion-bin spec {name} bin {position} label must be "
                f"{expected_label!r}, got {label!r}"
            )
        if bins and not math.isclose(bins[-1].upper, lower, abs_tol=1e-12):
            raise ValueError(f"Motion-bin spec {name} bins must be contiguous")
        bins.append(NumericBin(lower, upper, source_count))

    source_value_count = dimension.get("source_value_count")
    if source_value_count != sum(item.source_count for item in bins):
        raise ValueError(
            f"Motion-bin spec {name} source_value_count does not match bin counts"
        )
    return tuple(bins)


@lru_cache(maxsize=None)
def _load_motion_bin_spec_path(path_string: str) -> MotionBinSpec:
    path = Path(path_string)
    raw_bytes = path.read_bytes()
    try:
        document = json.loads(raw_bytes)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid motion-bin spec JSON at {path}: {error}") from error
    if not isinstance(document, Mapping):
        raise ValueError(f"Motion-bin spec root must be an object: {path}")
    schema_version = document.get("schema_version")
    spec_id = document.get("id")
    dataset = document.get("dataset")
    if schema_version != 1:
        raise ValueError(
            f"Unsupported motion-bin spec schema at {path}: "
            f"{schema_version!r}"
        )
    if not isinstance(spec_id, str) or not spec_id:
        raise ValueError(f"Motion-bin spec has no valid id: {path}")
    if not isinstance(dataset, str) or not dataset:
        raise ValueError(f"Motion-bin spec has no valid dataset: {path}")

    expected_filename = _SPEC_FILENAMES.get(spec_id)
    if expected_filename is not None and path.name != expected_filename:
        raise ValueError(
            f"Motion-bin spec {spec_id!r} must use filename {expected_filename!r}"
        )
    return MotionBinSpec(
        schema_version=schema_version,
        spec_id=spec_id,
        dataset=dataset,
        artifact_path=path,
        artifact_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        speed_bins=_dimension_bins(document, "speed_kmh"),
        acceleration_bins=_dimension_bins(
            document, "longitudinal_acceleration_mps2"
        ),
        document=document,
    )


def load_motion_bin_spec(value: str | Path | MotionBinSpec) -> MotionBinSpec:
    """Load a named/aliased spec or validate and return an existing spec."""

    if isinstance(value, MotionBinSpec):
        return value
    raw_value = str(value)
    spec_id = _SPEC_ALIASES.get(raw_value, raw_value)
    if spec_id in _SPEC_FILENAMES:
        path = BIN_SPEC_DIR / _SPEC_FILENAMES[spec_id]
    else:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Motion-bin spec does not exist: {path}")
    spec = _load_motion_bin_spec_path(str(path.resolve()))
    if spec_id in _SPEC_FILENAMES and spec.spec_id != spec_id:
        raise ValueError(
            f"Motion-bin artifact declares {spec.spec_id!r}, expected {spec_id!r}"
        )
    return spec


def infer_motion_bin_spec(result_dir: Path) -> MotionBinSpec:
    """Resolve the frozen spec recorded by, or unambiguously implied by, a result."""

    summary_path = result_dir / "metrics" / "summary.json"
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Cannot read result summary {summary_path}: {error}"
            ) from error
        if isinstance(summary, Mapping):
            metadata = summary.get("motion_bin_spec")
            if isinstance(metadata, Mapping) and isinstance(metadata.get("id"), str):
                spec = load_motion_bin_spec(metadata["id"])
                declared_hash = metadata.get("artifact_sha256")
                if (
                    declared_hash is not None
                    and declared_hash != spec.artifact_sha256
                ):
                    raise ValueError(
                        f"Result summary {summary_path} has a stale "
                        "motion-bin artifact hash"
                    )
                return spec
            dataset = summary.get("dataset")
            if dataset in {"pai-av", "paiav", "pai_av"}:
                return load_motion_bin_spec(PAIAV_BIN_SPEC_ID)

    parts = set(result_dir.parts)
    if "paiav_edited_images" in parts:
        return load_motion_bin_spec(PAIAV_BIN_SPEC_ID)
    if "navtest_filtered_plus_5" in parts:
        return load_motion_bin_spec(NAVSIM_BIN_SPEC_ID)
    raise ValueError(
        f"Cannot infer a motion-bin spec for {result_dir}; pass one explicitly"
    )


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


def _label_for_frozen_value(
    value: float, bins: Sequence[NumericBin], field: str
) -> str:
    for item in bins:
        if item.lower <= value < item.upper:
            return item.label
    raise ValueError(
        f"{field}={value:g} falls outside frozen coverage "
        f"[{bins[0].lower:g},{bins[-1].upper:g})"
    )


def speed_bin_label(
    speed_kmh: float,
    *,
    bin_spec: str | Path | MotionBinSpec | None = None,
) -> str:
    """Return the frozen group, or the unmerged 5 km/h base bin if omitted."""

    value = _finite_float(speed_kmh, "speed_kmh")
    if value < 0:
        raise ValueError(f"speed_kmh must be non-negative, got {speed_kmh!r}")
    if bin_spec is not None:
        spec = load_motion_bin_spec(bin_spec)
        return _label_for_frozen_value(value, spec.speed_bins, "speed_kmh")
    index = _bin_index(value, SPEED_BIN_WIDTH_KMH)
    return _base_label(index, index + 1, SPEED_BIN_WIDTH_KMH)


def acceleration_bin_label(
    longitudinal_acceleration_mps2: Optional[float],
    *,
    bin_spec: str | Path | MotionBinSpec | None = None,
) -> Optional[str]:
    """Return the frozen group, or the unmerged 0.5 m/s² base bin."""

    if longitudinal_acceleration_mps2 is None:
        return None
    value = _finite_float(
        longitudinal_acceleration_mps2, "longitudinal_acceleration_mps2"
    )
    if bin_spec is not None:
        spec = load_motion_bin_spec(bin_spec)
        return _label_for_frozen_value(
            value,
            spec.acceleration_bins,
            "longitudinal_acceleration_mps2",
        )
    index = _bin_index(value, ACCELERATION_BIN_WIDTH_MPS2)
    return _base_label(index, index + 1, ACCELERATION_BIN_WIDTH_MPS2)


def _merge_underpopulated_bins(
    grouped: Mapping[int, list[Mapping[str, Any]]],
) -> list[_BinGroup]:
    """Merge the lowest-position sparse bin with its less-populated neighbor."""

    groups = [
        _BinGroup(index, index + 1, list(grouped[index]))
        for index in sorted(grouped)
    ]
    while len(groups) > 1:
        position = next(
            (
                index
                for index, group in enumerate(groups)
                if len(group.records) < MIN_BIN_SAMPLE_COUNT
            ),
            None,
        )
        if position is None:
            break
        if position == 0:
            neighbor_position = 1
        elif position == len(groups) - 1:
            neighbor_position = position - 1
        else:
            left_count = len(groups[position - 1].records)
            right_count = len(groups[position + 1].records)
            neighbor_position = (
                position - 1 if left_count <= right_count else position + 1
            )
        left_position = min(position, neighbor_position)
        right_position = max(position, neighbor_position)
        left = groups[left_position]
        right = groups[right_position]
        groups[left_position : right_position + 1] = [
            _BinGroup(
                left.lower_index,
                right.upper_index,
                left.records + right.records,
            )
        ]
    return groups


def _dynamic_groups(
    records: Sequence[Mapping[str, Any]],
    *,
    field: str,
    width: float,
    nonnegative: bool,
) -> tuple[dict[str, list[Mapping[str, Any]]], int]:
    grouped: dict[int, list[Mapping[str, Any]]] = {}
    unbinned_count = 0
    for record in records:
        raw_value = record.get(field)
        if raw_value is None:
            unbinned_count += 1
            continue
        value = _finite_float(raw_value, field)
        if nonnegative and value < 0:
            raise ValueError(f"{field} must be non-negative, got {raw_value!r}")
        grouped.setdefault(_bin_index(value, width), []).append(record)
    return (
        {
            _base_label(group.lower_index, group.upper_index, width): group.records
            for group in _merge_underpopulated_bins(grouped)
        },
        unbinned_count,
    )


def _frozen_groups(
    records: Sequence[Mapping[str, Any]],
    *,
    field: str,
    bins: Sequence[NumericBin],
    nonnegative: bool,
) -> tuple[dict[str, list[Mapping[str, Any]]], int]:
    grouped = {item.label: [] for item in bins}
    unbinned_count = 0
    for record in records:
        raw_value = record.get(field)
        if raw_value is None:
            unbinned_count += 1
            continue
        value = _finite_float(raw_value, field)
        if nonnegative and value < 0:
            raise ValueError(f"{field} must be non-negative, got {raw_value!r}")
        label = _label_for_frozen_value(value, bins, field)
        grouped[label].append(record)
    return {
        label: group for label, group in grouped.items() if group
    }, unbinned_count


def occupied_speed_bin_groups(
    records: Iterable[Mapping[str, Any]],
    *,
    bin_spec: str | Path | MotionBinSpec | None = None,
) -> tuple[dict[str, list[Mapping[str, Any]]], int]:
    """Group records by frozen speed bins, or dynamically if no spec is passed."""

    record_list = list(records)
    if bin_spec is None:
        return _dynamic_groups(
            record_list,
            field="speed_kmh",
            width=SPEED_BIN_WIDTH_KMH,
            nonnegative=True,
        )
    spec = load_motion_bin_spec(bin_spec)
    return _frozen_groups(
        record_list,
        field="speed_kmh",
        bins=spec.speed_bins,
        nonnegative=True,
    )


def occupied_acceleration_bin_groups(
    records: Iterable[Mapping[str, Any]],
    *,
    bin_spec: str | Path | MotionBinSpec | None = None,
) -> tuple[dict[str, list[Mapping[str, Any]]], int]:
    """Group records by frozen acceleration bins, or derive bins if omitted."""

    record_list = list(records)
    if bin_spec is None:
        return _dynamic_groups(
            record_list,
            field="longitudinal_acceleration_mps2",
            width=ACCELERATION_BIN_WIDTH_MPS2,
            nonnegative=False,
        )
    spec = load_motion_bin_spec(bin_spec)
    return _frozen_groups(
        record_list,
        field="longitudinal_acceleration_mps2",
        bins=spec.acceleration_bins,
        nonnegative=False,
    )


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
