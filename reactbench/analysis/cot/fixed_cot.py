"""Fixed reasoning contracts and trajectory comparisons (CPU only).

Collision metrics come from protocol V2 scoring (``metrics/<dataset>_at_fault_*``), not from here.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import yaml

from ...models.original_cache import file_sha256, load_cached_trajectory
from ...scoring.geometry import average_displacement_error, trajectory_distance_driven

CONDITIONS = ("accelerate", "resume_speed", "slow_down", "yield", "immediate_stop")
REASONS = {
    "car": "vehicle blocking the lane ahead",
    "standing_pedestrian": "pedestrian blocking the lane ahead",
    "crossing_pedestrian": "pedestrian crossing ahead",
}
BASELINE = "natural_edited"
ORIGINAL = "natural_original"
SCHEMA_VERSION = 1


class ResumeConflict(ValueError):
    """An existing result must not be overwritten with different inputs."""


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def digest(value):
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True,
                                     allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_config(path):
    value = yaml.safe_load(Path(path).read_text())
    if not isinstance(value, dict) or value.get("version") != 1:
        raise ValueError("Fixed-CoT config requires version: 1")
    if set(value.get("conditions", {})) != set(CONDITIONS):
        raise ValueError(f"Fixed-CoT config requires exactly {CONDITIONS}")
    for text in value["conditions"].values():
        if not isinstance(text, str) or not text.strip() or "<" in text or ">" in text:
            raise ValueError("CoTs must be nonempty plain text without control tags")
        for reason in REASONS.values():
            try:
                text.format(reason=reason)
            except (KeyError, ValueError, IndexError) as exc:
                raise ValueError("Only the {reason} template field is supported") from exc
    for key, default in (("stop_speed_mps", 0.1), ("stop_duration_s", 0.5)):
        number = float(value.get(key, default))
        if not math.isfinite(number) or number <= 0:
            raise ValueError(f"{key} must be finite and positive")
        value[key] = number
    for key, default in (("sample_plot_count", 20), ("sample_plot_seed", 0)):
        number = value.get(key, default)
        if isinstance(number, bool) or not isinstance(number, int) or number < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
        value[key] = number
    return value


def object_type(metadata):
    """Resolve existing run/normalized actor metadata, never directory names."""
    actor = metadata.get("actor_type", "pedestrian")
    if actor == "car":
        return "car"
    if actor != "pedestrian":
        raise ValueError(f"Unsupported inserted actor: {actor!r}")
    behavior = metadata.get("behavior", metadata.get("actor_behavior",
                                                   metadata.get("pedestrian_behavior")))
    velocity = metadata.get("velocity_xy_mps")
    speed = np.linalg.norm(velocity) if velocity is not None else metadata.get("speed_mps")
    if speed is None or not np.isfinite(float(speed)) or float(speed) < 0:
        raise ValueError("Pedestrian metadata requires a finite nonnegative speed")
    stationary = float(speed) <= 1e-6
    if behavior not in (None, "standing", "stationary", "crossing", "moving"):
        raise ValueError(f"Unsupported pedestrian behavior: {behavior!r}")
    if behavior in ("standing", "stationary") and not stationary:
        raise ValueError("Standing pedestrian has nonzero speed")
    if behavior in ("crossing", "moving") and stationary:
        raise ValueError("Crossing pedestrian has zero speed")
    return "standing_pedestrian" if stationary else "crossing_pedestrian"


def resolve_conditions(config, kind):
    return {name: config["conditions"][name].format(reason=REASONS[kind])
            for name in CONDITIONS}


def add_arguments(parser):
    parser.add_argument("--fixed-cots", type=Path, metavar="PATH", nargs="?",
                        const=Path(__file__).resolve().with_name("fixed_cots.yaml"),
                        help="Evaluate five fixed CoTs; defaults to the bundled fixed_cots.yaml.")


def validate_arguments(parser, args):
    if args.fixed_cots and (args.original or getattr(args, "cot_swap", False)
                           or getattr(args, "replay_controls", False)):
        parser.error("--fixed-cots cannot be combined with --original or CoT-swap options")


def prepare_manifest(root, identity, tokens):
    path = root / "evaluation_manifest.json"
    if path.exists():
        previous = read_json(path)
        if previous.get("identity") != jsonable(identity):
            raise ValueError(f"Conflicting fixed-CoT output settings: {path}")
    elif root.exists() and any(root.iterdir()):
        raise ValueError(f"Refusing to reuse output without a manifest: {root}")
    save_json(path, {"mode": "fixed_cots", "schema_version": SCHEMA_VERSION,
                     "identity": identity, "selected_tokens": tokens})


def input_provenance(agent_input, metadata_path):
    """Fingerprint the supplied dynamics, metadata and actual camera files."""
    files = {}

    def visit(value):
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        elif isinstance(value, (str, Path)):
            p = Path(value)
            if p.suffix.lower() in (".png", ".jpg", ".jpeg") and p.is_file():
                files[str(p.resolve())] = file_sha256(p)

    visit(agent_input)
    return {"input_sha256": digest(agent_input), "image_sha256": files,
            "metadata_path": str(metadata_path),
            "metadata_sha256": file_sha256(metadata_path)}


def speed_metrics(poses, dt, initial_speed, config):
    """Segment-average planar speeds, including origin to the first pose."""
    xy = np.vstack(([0.0, 0.0], np.asarray(poses)[:, :2]))
    speeds = np.linalg.norm(np.diff(xy, axis=0), axis=1) / dt
    threshold = config["stop_speed_mps"]
    window = max(1, int(math.ceil(config["stop_duration_s"] / dt - 1e-9)))
    low = speeds <= threshold
    start = next((i for i in range(len(low) - window + 1)
                  if low[i:i + window].all()), None)
    return {
        "speed_mps": speeds.tolist(),
        "speed_time_s": ((np.arange(len(speeds)) + 0.5) * dt).tolist(),
        "final_speed_mps": float(speeds[-1]),
        "initial_speed_mps": float(initial_speed),
        "initially_stationary": bool(initial_speed <= threshold),
        "sustained_stop": start is not None,
        "stop_time_s": None if start is None else float(start * dt),
    }


def condition_metrics(poses, *, dt, initial_speed, config):
    return {
        "distance_driven_m": trajectory_distance_driven(poses),
        **speed_metrics(poses, dt, initial_speed, config),
    }


def contrast(before_poses, after_poses, before, after):
    return {
        "ade_m": average_displacement_error(before_poses, after_poses),
        **{f"{key}_delta": after[key] - before[key] for key in (
            "distance_driven_m", "final_speed_mps")},
    }


def aggregate(records, tokens):
    complete = [r for r in records if all(c in r["conditions"] for c in CONDITIONS)]
    paired = [r for r in complete if BASELINE in r["conditions"]]
    # All main bars/curves use precisely this cohort; fixed-only remains available
    # when natural caches are missing, with its own explicit denominator.
    def summarize(cohort, names):
        result = {}
        for name in names:
            values = [r["conditions"][name] for r in cohort]
            moving = [v for v in values if not v["initially_stationary"]]
            mean = lambda key, rows=values: (float(np.mean([v[key] for v in rows]))
                                           if rows else None)
            result[name] = {
                "count": len(values), "stopping_rate": mean("sustained_stop"),
                "moving_stopping_rate": mean("sustained_stop", moving),
                "moving_count": len(moving),
                "distance_driven_m": mean("distance_driven_m"),
                "final_speed_mps": mean("final_speed_mps"),
            }
        return result
    contrasts = {}
    for before in (*CONDITIONS, BASELINE, ORIGINAL):
        cohort = [r for r in complete if before in r["conditions"]]
        for after in CONDITIONS:
            if before == after:
                continue
            key = f"{before}->{after}"
            values = [r["contrasts"][key] for r in cohort if key in r["contrasts"]]
            contrasts[key] = {
                "count": len(values),
                **{field: float(np.mean([v[field] for v in values])) if values else None
                   for field in ("ade_m", "distance_driven_m_delta", "final_speed_mps_delta")},
            }
    return {
        "requested_count": len(tokens), "fixed_complete_count": len(complete),
        "comparison_count": len(paired),
        "comparison_tokens": [r["token"] for r in paired],
        "condition_counts": {name: {
            "attempted": len(tokens),
            "successful": sum(name in r["conditions"] for r in records),
            "failed": len(tokens) - sum(name in r["conditions"] for r in records),
        } for name in CONDITIONS},
        "conditions": summarize(paired, (*CONDITIONS, BASELINE)),
        "fixed_only_conditions": summarize(complete, CONDITIONS),
        "contrasts": contrasts,
    }


def load_completed(root, token, expected, dt):
    """Reasoning JSON is the commit marker, written after the trajectory."""
    reasoning_path = root / "reasoning" / f"{token}.json"
    trajectory_path = root / "trajectories" / f"{token}.npz"
    if not reasoning_path.exists():
        return None
    document = read_json(reasoning_path)
    if document.get("intervention") != jsonable(expected):
        raise ResumeConflict(f"Conflicting saved intervention: {reasoning_path}")
    if not trajectory_path.exists():
        raise ResumeConflict(f"Completed intervention has no trajectory: {trajectory_path}")
    if document.get("trajectory_sha256") != file_sha256(trajectory_path):
        raise ResumeConflict(f"Saved trajectory checksum mismatch: {trajectory_path}")
    return load_cached_trajectory(trajectory_path, expected_interval_length_s=dt)


def save_completed(root, token, poses, output, intervention, dt):
    directory = root / "trajectories"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{token}.npz"
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, trajectory=poses, interval_length_s=np.float32(dt))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    save_json(root / "reasoning" / f"{token}.json", {
        "intervention": intervention, "model_output": output,
        "trajectory_sha256": file_sha256(path),
    })
