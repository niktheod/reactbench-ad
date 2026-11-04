"""Shared contracts and metrics for Alpamayo image-by-CoT interventions.

Collision metrics come from protocol V2 scoring (``metrics/<dataset>_at_fault_*``), not from here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ...models.original_cache import load_cached_trajectory
from ...scoring.geometry import average_displacement_error, trajectory_distance_driven


NATURAL_ORIGINAL = "original_image_original_cot"
NATURAL_EDITED = "edited_image_edited_cot"
CROSSED_ORIGINAL_IMAGE = "original_image_edited_cot"
CROSSED_EDITED_IMAGE = "edited_image_original_cot"
REPLAY_ORIGINAL = "replay_original_image_original_cot"
REPLAY_EDITED = "replay_edited_image_edited_cot"

FACTORIAL_CONDITIONS = (
    NATURAL_ORIGINAL,
    NATURAL_EDITED,
    CROSSED_ORIGINAL_IMAGE,
    CROSSED_EDITED_IMAGE,
)
CROSSED_CONDITIONS = (
    CROSSED_ORIGINAL_IMAGE,
    CROSSED_EDITED_IMAGE,
)
REPLAY_CONDITIONS = (REPLAY_ORIGINAL, REPLAY_EDITED)

CONDITION_SOURCES = {
    NATURAL_ORIGINAL: {"image_source": "original", "cot_source": "original"},
    NATURAL_EDITED: {"image_source": "edited", "cot_source": "edited"},
    CROSSED_ORIGINAL_IMAGE: {
        "image_source": "original",
        "cot_source": "edited",
    },
    CROSSED_EDITED_IMAGE: {
        "image_source": "edited",
        "cot_source": "original",
    },
    REPLAY_ORIGINAL: {"image_source": "original", "cot_source": "original"},
    REPLAY_EDITED: {"image_source": "edited", "cot_source": "edited"},
}

CONTRASTS = {
    "natural_edit_effect": (NATURAL_ORIGINAL, NATURAL_EDITED),
    "cot_effect_on_original_image": (
        NATURAL_ORIGINAL,
        CROSSED_ORIGINAL_IMAGE,
    ),
    "image_effect_with_original_cot": (
        NATURAL_ORIGINAL,
        CROSSED_EDITED_IMAGE,
    ),
    "image_effect_with_edited_cot": (
        CROSSED_ORIGINAL_IMAGE,
        NATURAL_EDITED,
    ),
    "cot_effect_on_edited_image": (
        CROSSED_EDITED_IMAGE,
        NATURAL_EDITED,
    ),
}

RESERVED_COT_TOKENS = (
    "<|cot_start|>",
    "<|cot_end|>",
    "<|traj_future_start|>",
    "<|traj_future_end|>",
)


@dataclass(frozen=True)
class SavedCot:
    """One decoded natural CoT and its on-disk provenance."""

    text: str
    path: Path
    sha256: str


@dataclass(frozen=True)
class CotSwapSources:
    """Natural artifacts required before crossed inference can start."""

    original_cots: dict[str, SavedCot]
    edited_cots: dict[str, SavedCot]
    edited_trajectories: dict[str, np.ndarray]
    edited_summary_path: Path | None
    edited_summary_sha256: str | None


def forced_cot_prefix(cot: str) -> str:
    """Return the complete assistant prefix consumed by trajectory generation."""

    if not isinstance(cot, str):
        raise TypeError(f"forced_cot must be a string, got {type(cot).__name__}")
    contained = [token for token in RESERVED_COT_TOKENS if token in cot]
    if contained:
        raise ValueError(
            "forced_cot contains reserved Alpamayo token(s): "
            f"{contained}"
        )
    return f"<|cot_start|>{cot}<|cot_end|><|traj_future_start|>"


def _unwrap_singleton_cot(value: Any, path: Path) -> str:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    while isinstance(value, list):
        if len(value) != 1:
            raise ValueError(
                "Expected singleton Alpamayo CoT dimensions in "
                f"{path}, got length {len(value)}"
            )
        value = value[0]
    if not isinstance(value, str):
        raise ValueError(f"Expected a string CoT in {path}")
    forced_cot_prefix(value)
    return value


def load_saved_cot(path: Path) -> SavedCot:
    """Load the single decoded CoT saved by a one-trajectory Alpamayo run."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Saved Alpamayo reasoning is missing: {resolved}")
    with resolved.open(encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, Mapping) or "cot" not in document:
        raise ValueError(f"Saved reasoning has no 'cot' field: {resolved}")
    text = _unwrap_singleton_cot(document["cot"], resolved)
    digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
    return SavedCot(text=text, path=resolved, sha256=digest)


def load_cot_swap_sources(
    original_result_dir: Path,
    edited_result_dir: Path,
    *,
    tokens: Sequence[str],
    interval_length_s: float,
    expected_edited_summary: Mapping[str, Any] | None = None,
) -> CotSwapSources:
    """Fail fast unless all natural CoTs and edited trajectories are present."""

    summary_path: Path | None = None
    summary_sha256: str | None = None
    if expected_edited_summary is not None:
        summary_path = edited_result_dir / "metrics" / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(
                f"Natural edited summary is missing: {summary_path}"
            )
        with summary_path.open(encoding="utf-8") as handle:
            summary = json.load(handle)
        if not isinstance(summary, Mapping):
            raise ValueError(f"Expected a JSON object in {summary_path}")
        mismatches: list[str] = []
        for key, expected in expected_edited_summary.items():
            actual = (
                summary.get(key, False)
                if isinstance(expected, bool)
                else summary.get(key)
            )
            if isinstance(expected, float) and isinstance(actual, (int, float)):
                matches = bool(np.isclose(actual, expected, atol=1e-9))
            else:
                matches = actual == expected
            if not matches:
                mismatches.append(f"{key}: expected {expected!r}, got {actual!r}")
        if mismatches:
            raise ValueError(
                "Natural edited result is incompatible with this CoT-swap "
                f"evaluation ({summary_path}): {'; '.join(mismatches)}"
            )
        summary_path = summary_path.resolve()
        summary_sha256 = hashlib.sha256(summary_path.read_bytes()).hexdigest()

    original_cots: dict[str, SavedCot] = {}
    edited_cots: dict[str, SavedCot] = {}
    edited_trajectories: dict[str, np.ndarray] = {}
    for token in tokens:
        original_cots[token] = load_saved_cot(
            original_result_dir / "reasoning" / f"{token}.json"
        )
        edited_cots[token] = load_saved_cot(
            edited_result_dir / "reasoning" / f"{token}.json"
        )
        edited_trajectories[token] = load_cached_trajectory(
            edited_result_dir / "trajectories" / f"{token}.npz",
            expected_interval_length_s=interval_length_s,
        )
    return CotSwapSources(
        original_cots=original_cots,
        edited_cots=edited_cots,
        edited_trajectories=edited_trajectories,
        edited_summary_path=summary_path,
        edited_summary_sha256=summary_sha256,
    )


def intervention_reasoning(
    model_extra: Mapping[str, Any],
    *,
    condition: str,
    donor_cot: SavedCot,
    diffusion_seed: int | None = None,
    generation_seed: int | None = None,
) -> dict[str, Any]:
    """Attach explicit intervention provenance to Alpamayo text outputs."""

    if condition not in CONDITION_SOURCES:
        raise ValueError(f"Unknown CoT-swap condition: {condition}")
    if "cot" not in model_extra:
        raise ValueError("Forced Alpamayo output has no 'cot' field")
    replayed_cot = _unwrap_singleton_cot(
        model_extra["cot"], Path("<forced Alpamayo output>")
    )
    if replayed_cot != donor_cot.text:
        raise ValueError(
            "Forced Alpamayo output did not preserve the donor CoT: "
            f"expected {donor_cot.text!r}, got {replayed_cot!r}"
        )
    result = dict(model_extra)
    result["intervention"] = {
        "type": "forced_cot_token_replay",
        "condition": condition,
        **CONDITION_SOURCES[condition],
        "forced_cot": donor_cot.text,
        "donor_reasoning_path": str(donor_cot.path),
        "donor_reasoning_sha256": donor_cot.sha256,
        **_seed_metadata(diffusion_seed, generation_seed),
    }
    return result


def _seed_metadata(diffusion_seed: int | None, generation_seed: int | None) -> dict[str, int]:
    if (diffusion_seed is None) == (generation_seed is None):
        raise ValueError("Supply exactly one of diffusion_seed or generation_seed")
    if generation_seed is not None:
        return {"generation_seed": int(generation_seed)}
    return {"diffusion_seed": int(diffusion_seed)}


def _condition_metrics(trajectory: Any) -> dict[str, Any]:
    return {"distance_driven_m": trajectory_distance_driven(trajectory)}


def _contrast_metrics(
    before_name: str,
    after_name: str,
    trajectories: Mapping[str, Any],
    conditions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    before = conditions[before_name]
    after = conditions[after_name]
    return {
        "from_condition": before_name,
        "to_condition": after_name,
        "ade_m": average_displacement_error(
            trajectories[before_name], trajectories[after_name]
        ),
        "distance_driven_delta_m": (
            after["distance_driven_m"] - before["distance_driven_m"]
        ),
    }


def _interaction_metrics(
    trajectories: Mapping[str, Any],
    conditions: Mapping[str, Mapping[str, Any]],
) -> dict[str, float]:
    arrays = {
        condition: np.asarray(trajectories[condition], dtype=np.float64)
        for condition in FACTORIAL_CONDITIONS
    }
    shapes = {array.shape for array in arrays.values()}
    if len(shapes) != 1:
        raise ValueError(
            "CoT-swap interaction requires equal trajectory shapes, got "
            f"{sorted(shapes)}"
        )
    trajectory_interaction = (
        arrays[NATURAL_EDITED][:, :2]
        - arrays[CROSSED_ORIGINAL_IMAGE][:, :2]
        - arrays[CROSSED_EDITED_IMAGE][:, :2]
        + arrays[NATURAL_ORIGINAL][:, :2]
    )

    def difference_in_differences(field: str) -> float:
        return float(
            conditions[NATURAL_EDITED][field]
            - conditions[CROSSED_ORIGINAL_IMAGE][field]
            - conditions[CROSSED_EDITED_IMAGE][field]
            + conditions[NATURAL_ORIGINAL][field]
        )

    return {
        "trajectory_mean_norm_m": float(
            np.linalg.norm(trajectory_interaction, axis=1).mean()
        ),
        "distance_driven_m": difference_in_differences("distance_driven_m"),
    }


def build_cot_swap_record(
    *,
    token: str,
    trajectories: Mapping[str, Any],
    actor_motion: Mapping[str, Any],
    interval_length_s: float,
    speed_mps: float,
    diffusion_seed: int | None = None,
    generation_seed: int | None = None,
    longitudinal_acceleration_mps2: float | None = None,
    condition_provenance: Mapping[str, Mapping[str, Any]] | None = None,
    replay_trajectories: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build explicitly named four-condition trajectory comparisons for one token."""

    missing = [name for name in FACTORIAL_CONDITIONS if name not in trajectories]
    if missing:
        raise ValueError(f"Missing CoT-swap trajectories: {missing}")
    speed = float(speed_mps)
    if not np.isfinite(speed) or speed < 0:
        raise ValueError(f"speed_mps must be finite and non-negative, got {speed}")
    acceleration = longitudinal_acceleration_mps2
    if acceleration is not None:
        acceleration = float(acceleration)
        if not np.isfinite(acceleration):
            raise ValueError("longitudinal_acceleration_mps2 must be finite or None")

    condition_provenance = condition_provenance or {}
    condition_metrics: dict[str, dict[str, Any]] = {}
    for condition in FACTORIAL_CONDITIONS:
        condition_metrics[condition] = {
            **CONDITION_SOURCES[condition],
            **_condition_metrics(trajectories[condition]),
            **dict(condition_provenance.get(condition, {})),
        }

    contrasts = {
        name: _contrast_metrics(
            before,
            after,
            trajectories,
            condition_metrics,
        )
        for name, (before, after) in CONTRASTS.items()
    }
    record: dict[str, Any] = {
        "token": token,
        "output_stem": token,
        "speed_mps": speed,
        "speed_kmh": speed * 3.6,
        "longitudinal_acceleration_mps2": acceleration,
        **_seed_metadata(diffusion_seed, generation_seed),
        "actor_motion": dict(actor_motion),
        "metric_pose_count": int(
            np.asarray(trajectories[NATURAL_ORIGINAL]).shape[0]
        ),
        "metric_horizon_s": float(
            np.asarray(trajectories[NATURAL_ORIGINAL]).shape[0]
            * interval_length_s
        ),
        "conditions": condition_metrics,
        "contrasts": contrasts,
        "interaction_difference_in_differences": _interaction_metrics(
            trajectories, condition_metrics
        ),
    }

    if replay_trajectories:
        required_replays = [
            name for name in REPLAY_CONDITIONS if name not in replay_trajectories
        ]
        if required_replays:
            raise ValueError(f"Missing self-replay trajectories: {required_replays}")
        replay_controls: dict[str, Any] = {}
        for replay_name, natural_name in (
            (REPLAY_ORIGINAL, NATURAL_ORIGINAL),
            (REPLAY_EDITED, NATURAL_EDITED),
        ):
            replay_controls[replay_name] = {
                **CONDITION_SOURCES[replay_name],
                "reference_condition": natural_name,
                "ade_m": average_displacement_error(
                    trajectories[natural_name], replay_trajectories[replay_name]
                ),
                **_condition_metrics(replay_trajectories[replay_name]),
                **dict(condition_provenance.get(replay_name, {})),
            }
        record["replay_controls"] = replay_controls
    return record


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else float(np.mean(np.asarray(values, dtype=np.float64)))


def aggregate_cot_swap_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate four-condition and contrast metrics without ambiguous labels."""

    condition_summary: dict[str, Any] = {}
    for condition in FACTORIAL_CONDITIONS:
        values = [record["conditions"][condition] for record in records]
        condition_summary[condition] = {
            **CONDITION_SOURCES[condition],
            "count": len(values),
            "mean_distance_driven_m": _mean(
                [float(value["distance_driven_m"]) for value in values]
            ),
        }

    contrast_summary: dict[str, Any] = {}
    for contrast in CONTRASTS:
        values = [record["contrasts"][contrast] for record in records]
        contrast_summary[contrast] = {
            "from_condition": CONTRASTS[contrast][0],
            "to_condition": CONTRASTS[contrast][1],
            "count": len(values),
            "mean_ade_m": _mean([float(value["ade_m"]) for value in values]),
            "mean_distance_driven_delta_m": _mean(
                [float(value["distance_driven_delta_m"]) for value in values]
            ),
        }

    summary: dict[str, Any] = {
        "count": len(records),
        "conditions": condition_summary,
        "contrasts": contrast_summary,
        "interaction_difference_in_differences": {
            field: _mean(
                [
                    float(
                        record["interaction_difference_in_differences"][field]
                    )
                    for record in records
                ]
            )
            for field in ("trajectory_mean_norm_m", "distance_driven_m")
        },
    }
    if records and all("replay_controls" in record for record in records):
        summary["replay_controls"] = {
            replay: {
                "reference_condition": (
                    NATURAL_ORIGINAL if replay == REPLAY_ORIGINAL else NATURAL_EDITED
                ),
                "count": len(records),
                "mean_ade_m": _mean(
                    [
                        float(record["replay_controls"][replay]["ade_m"])
                        for record in records
                    ]
                ),
            }
            for replay in REPLAY_CONDITIONS
        }
    return summary
