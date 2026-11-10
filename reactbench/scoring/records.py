"""Per-sample bookkeeping written by the runners next to their predictions.

``metrics/per_sample.json`` keeps only what the saved-result inventory needs (identity, motion
bins, inserted object); every metric comes from scoring (``metrics/<dataset>_at_fault_*``).
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

METRIC_PROTOCOL = "protocol V2: metrics/navsim_at_fault_* or metrics/paiav_at_fault_*"


def sample_record(*, token: str, output_stem: str, speed_mps: float, actor_motion: Mapping[str, Any],
                  longitudinal_acceleration_mps2: Optional[float] = None,
                  vehicle_geometry: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Bookkeeping fields of one sample."""
    from .actor_motion import validated_motion
    from .geometry import vehicle_footprint_parameters
    from .motion_bins import speed_bin_label
    import numpy as np
    speed_mps = float(speed_mps)
    if not np.isfinite(speed_mps) or speed_mps < 0:
        raise ValueError(f"speed_mps must be finite and non-negative, got {speed_mps}")
    if longitudinal_acceleration_mps2 is not None:
        longitudinal_acceleration_mps2 = float(longitudinal_acceleration_mps2)
        if not np.isfinite(longitudinal_acceleration_mps2):
            raise ValueError(f"longitudinal_acceleration_mps2 must be finite or None, got {longitudinal_acceleration_mps2!r}")
    if actor_motion is None:
        raise ValueError("actor_motion is required")
    motion = validated_motion(actor_motion)
    actor_type = str(motion.get("actor_type", "pedestrian"))
    geometry = motion.get("geometry")
    if actor_type.lower() == "pedestrian" and not isinstance(geometry, Mapping):
        raise ValueError("Pedestrian records require measured geometry provenance")
    speed_kmh = speed_mps * 3.6
    record = {
        "token": token, "output_stem": output_stem, "metric_protocol": METRIC_PROTOCOL,
        "speed_mps": speed_mps, "speed_kmh": speed_kmh,
        "speed_bin_kmh": speed_bin_label(speed_kmh),
        "longitudinal_acceleration_mps2": longitudinal_acceleration_mps2,
        "ego_origin": "rear_axle", "vehicle_footprint": vehicle_footprint_parameters(vehicle_geometry),
        "actor_type": actor_type, "actor_motion": motion,
    }
    if isinstance(geometry, Mapping):
        record.update(actor_geometry=geometry, actor_geometry_source=geometry["source"],
                      actor_geometry_version=geometry["profile_version"], actor_geometry_asset_id=geometry["asset_id"])
    return record


def run_summary(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Base of ``metrics/summary.json`` for new runs: counts only; callers add run bookkeeping."""
    return {"metric_protocol": METRIC_PROTOCOL, "sample_count": len(records)}


