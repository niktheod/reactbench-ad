"""Per-sample bookkeeping written by the runners next to their predictions.

``metrics/per_sample.json`` keeps only what the saved-result inventory needs (identity, motion
bins, inserted object); every metric comes from scoring (``metrics/<dataset>_at_fault_*``).
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

METRIC_PROTOCOL = "protocol V2: metrics/navsim_at_fault_* or metrics/paiav_at_fault_* (no legacy metrics)"


def sample_record(*, token: str, output_stem: str, speed_mps: float,
                  longitudinal_acceleration_mps2: Optional[float] = None, bin_spec: Any = None,
                  actor_motion: Optional[Mapping[str, Any]] = None,
                  pedestrian_motion: Optional[Mapping[str, Any]] = None,
                  vehicle_geometry: Optional[Mapping[str, Any]] = None,
                  original_trajectory: Any = None, edited_trajectory: Any = None,
                  interval_length_s: Optional[float] = None) -> dict[str, Any]:
    """Bookkeeping fields of one sample; the trajectory and interval arguments are not used."""
    from .actor_motion import validated_motion
    from .geometry import vehicle_footprint_parameters
    from .motion_bins import acceleration_bin_label, load_motion_bin_spec, speed_bin_label
    import numpy as np
    speed_mps = float(speed_mps)
    if not np.isfinite(speed_mps) or speed_mps < 0:
        raise ValueError(f"speed_mps must be finite and non-negative, got {speed_mps}")
    if longitudinal_acceleration_mps2 is not None:
        longitudinal_acceleration_mps2 = float(longitudinal_acceleration_mps2)
        if not np.isfinite(longitudinal_acceleration_mps2):
            raise ValueError(f"longitudinal_acceleration_mps2 must be finite or None, got {longitudinal_acceleration_mps2!r}")
    if actor_motion is not None and pedestrian_motion is not None:
        raise ValueError("Pass actor_motion or pedestrian_motion, not both")
    value = actor_motion if actor_motion is not None else pedestrian_motion
    if value is None:
        raise ValueError("actor_motion is required")
    motion = validated_motion(value)
    actor_type = str(motion.get("actor_type", "pedestrian"))
    geometry = motion.get("geometry")
    if actor_type.lower() == "pedestrian" and not isinstance(geometry, Mapping):
        raise ValueError("Pedestrian records require measured geometry provenance")
    spec = load_motion_bin_spec(bin_spec) if bin_spec is not None else None
    speed_kmh = speed_mps * 3.6
    record = {
        "token": token, "output_stem": output_stem, "metric_protocol": METRIC_PROTOCOL,
        "speed_mps": speed_mps, "speed_kmh": speed_kmh,
        "speed_bin_kmh": speed_bin_label(speed_kmh, bin_spec=spec),
        "longitudinal_acceleration_mps2": longitudinal_acceleration_mps2,
        "acceleration_bin_mps2": acceleration_bin_label(longitudinal_acceleration_mps2, bin_spec=spec),
        "ego_origin": "rear_axle", "vehicle_footprint": vehicle_footprint_parameters(vehicle_geometry),
        "actor_type": actor_type, "actor_motion": motion,
    }
    if isinstance(geometry, Mapping):
        record.update(actor_geometry=geometry, actor_geometry_source=geometry["source"],
                      actor_geometry_version=geometry["profile_version"], actor_geometry_asset_id=geometry["asset_id"])
    if spec is not None:
        record.update(motion_bin_spec_id=spec.spec_id, motion_bin_spec_sha256=spec.artifact_sha256)
    if actor_type.lower() != "car":
        record["pedestrian_motion"] = motion
    return record


def run_summary(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Base of ``metrics/summary.json`` for new runs: counts only; callers add run bookkeeping."""
    return {"metric_protocol": METRIC_PROTOCOL, "sample_count": len(records)}


def visualization_title(metrics: Mapping[str, Any], format_time) -> str:
    """Per-sample figure title; legacy fields (ADE, inserted contact times) appear only when present."""
    parts = [f"Token {metrics['token']}", f"{metrics['speed_kmh']:.1f} km/h"]
    if metrics.get("ade_m") is not None:
        parts.append(f"ADE {metrics['ade_m']:.2f} m")
    if "original_time_to_collision_s" in metrics:
        parts.append(f"raw inserted contact original {format_time(metrics['original_time_to_collision_s'])}")
        parts.append(f"edited {format_time(metrics['edited_time_to_collision_s'])}")
    return " | ".join(parts)
