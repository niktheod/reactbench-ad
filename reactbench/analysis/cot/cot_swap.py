"""CoT swap (paper section 4.3): original/edited images crossed with original/edited CoTs, for both Alpamayo models.

The natural conditions come from the natural runs (``natural.py``); the crossed ones force the other scene's CoT
onto an image, with the token's natural seeds: ``original_image_edited_cot`` and ``edited_image_original_cot``,
one directory each in ``<run>_cot_swap/``. Collision metrics come from protocol V2 scoring
(``metrics/paiav_at_fault_*``), not from here.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ...models import runner
from ...models.original_cache import save_manifest
from ...scoring.geometry import average_displacement_error, trajectory_distance_driven
from .natural import NaturalMismatch, natural_runs

NATURAL_ORIGINAL = "original_image_original_cot"
NATURAL_EDITED = "edited_image_edited_cot"
CROSSED_ORIGINAL_IMAGE = "original_image_edited_cot"
CROSSED_EDITED_IMAGE = "edited_image_original_cot"
FACTORIAL_CONDITIONS = (NATURAL_ORIGINAL, NATURAL_EDITED, CROSSED_ORIGINAL_IMAGE, CROSSED_EDITED_IMAGE)
CONDITION_SOURCES = {
    NATURAL_ORIGINAL: {"image_source": "original", "cot_source": "original"},
    NATURAL_EDITED: {"image_source": "edited", "cot_source": "edited"},
    CROSSED_ORIGINAL_IMAGE: {"image_source": "original", "cot_source": "edited"},
    CROSSED_EDITED_IMAGE: {"image_source": "edited", "cot_source": "original"},
}
CONTRASTS = {
    "natural_edit_effect": (NATURAL_ORIGINAL, NATURAL_EDITED),
    "cot_effect_on_original_image": (NATURAL_ORIGINAL, CROSSED_ORIGINAL_IMAGE),
    "image_effect_with_original_cot": (NATURAL_ORIGINAL, CROSSED_EDITED_IMAGE),
    "image_effect_with_edited_cot": (CROSSED_ORIGINAL_IMAGE, NATURAL_EDITED),
    "cot_effect_on_edited_image": (CROSSED_EDITED_IMAGE, NATURAL_EDITED),
}
INTERVENTION = ("the donor CoT decoded by the natural rollout is forced as the assistant prefix; hidden states are "
                "recomputed with the selected image and the unchanged diffusion expert reads the prompt KV cache")
METRIC_DEFINITIONS = {
    "condition_metrics": "distance driven over the first four seconds; collision metrics come from protocol V2 "
                         "scoring (metrics/paiav_at_fault_*)",
    "contrast_ade_m": "mean corresponding-pose x/y displacement from the named from_condition to to_condition",
    "contrast_delta": "to_condition value minus from_condition value",
    "interaction_difference_in_differences": "(edited-image effect with edited CoT) minus "
                                             "(edited-image effect with original CoT)",
}


def unwrap_singleton(value: Any, field: str = "cot") -> str:
    """Upstream text outputs are nested (batch, set, sample) singletons."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    while isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"Expected one {field} output, got {len(value)}")
        value = value[0]
    if not isinstance(value, str):
        raise ValueError(f"Expected a string {field} output, got {type(value).__name__}")
    return value


def _contrast_metrics(before_name, after_name, trajectories, conditions):
    before, after = conditions[before_name], conditions[after_name]
    return {"from_condition": before_name, "to_condition": after_name,
            "ade_m": average_displacement_error(trajectories[before_name], trajectories[after_name]),
            "distance_driven_delta_m": after["distance_driven_m"] - before["distance_driven_m"]}


def _interaction_metrics(trajectories, conditions):
    arrays = {name: np.asarray(trajectories[name], dtype=np.float64) for name in FACTORIAL_CONDITIONS}
    if len({array.shape for array in arrays.values()}) != 1:
        raise ValueError("CoT-swap interaction requires equal trajectory shapes")
    trajectory_interaction = (arrays[NATURAL_EDITED][:, :2] - arrays[CROSSED_ORIGINAL_IMAGE][:, :2]
                              - arrays[CROSSED_EDITED_IMAGE][:, :2] + arrays[NATURAL_ORIGINAL][:, :2])

    def difference_in_differences(field):
        return float(conditions[NATURAL_EDITED][field] - conditions[CROSSED_ORIGINAL_IMAGE][field]
                     - conditions[CROSSED_EDITED_IMAGE][field] + conditions[NATURAL_ORIGINAL][field])

    return {"trajectory_mean_norm_m": float(np.linalg.norm(trajectory_interaction, axis=1).mean()),
            "distance_driven_m": difference_in_differences("distance_driven_m")}


def build_cot_swap_record(*, token: str, trajectories: Mapping[str, Any], actor_motion: Mapping[str, Any],
                          interval_length_s: float, speed_mps: float, seeds: Mapping[str, int],
                          longitudinal_acceleration_mps2: float | None = None,
                          condition_provenance: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Explicitly named four-condition trajectory comparisons for one token (four-second trajectories)."""
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
    provenance = condition_provenance or {}
    conditions = {name: {**CONDITION_SOURCES[name], "distance_driven_m": trajectory_distance_driven(trajectories[name]),
                         **dict(provenance.get(name, {}))} for name in FACTORIAL_CONDITIONS}
    count = np.asarray(trajectories[NATURAL_ORIGINAL]).shape[0]
    return {"token": token, "output_stem": token, "speed_mps": speed, "speed_kmh": speed * 3.6,
            "longitudinal_acceleration_mps2": acceleration, **seeds, "actor_motion": dict(actor_motion),
            "metric_pose_count": int(count), "metric_horizon_s": float(count * interval_length_s),
            "conditions": conditions,
            "contrasts": {name: _contrast_metrics(before, after, trajectories, conditions)
                          for name, (before, after) in CONTRASTS.items()},
            "interaction_difference_in_differences": _interaction_metrics(trajectories, conditions)}


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else float(np.mean(np.asarray(values, dtype=np.float64)))


def aggregate_cot_swap_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate four-condition and contrast metrics without ambiguous labels."""
    conditions = {name: {**CONDITION_SOURCES[name], "count": len(records), "mean_distance_driven_m": _mean(
        [float(record["conditions"][name]["distance_driven_m"]) for record in records])}
        for name in FACTORIAL_CONDITIONS}
    contrasts = {}
    for name, (before, after) in CONTRASTS.items():
        values = [record["contrasts"][name] for record in records]
        contrasts[name] = {"from_condition": before, "to_condition": after, "count": len(values),
                           "mean_ade_m": _mean([float(value["ade_m"]) for value in values]),
                           "mean_distance_driven_delta_m": _mean(
                               [float(value["distance_driven_delta_m"]) for value in values])}
    return {"count": len(records), "conditions": conditions, "contrasts": contrasts,
            "interaction_difference_in_differences": {
                field: _mean([float(record["interaction_difference_in_differences"][field]) for record in records])
                for field in ("trajectory_mean_norm_m", "distance_driven_m")}}


def run(s) -> int:
    """The CoT swap of ``s.tokens`` (``natural.main`` prepares ``s``); 0 if any token succeeded."""
    model, dt = s.model, s.model.interval_length_s
    natural, status = natural_runs(s, required=True)
    unattempted = sorted({token for token in s.tokens for run in natural.values() if token not in run.selected})
    if unattempted:
        raise ValueError(f"The natural runs never attempted {len(unattempted)} requested token(s): "
                         f"{unattempted[:10]}. Select tokens that both natural runs attempted.")
    # Check every natural output before loading the model: seed or provenance mismatches are fatal; missing
    # outputs or CoTs that cannot be replayed exclude a token.
    failures, prepared = [], {}
    for index, token in enumerate(s.tokens, 1):
        reasons = {name: run.unavailable(token) for name, run in natural.items()}
        if any(reasons.values()):
            failures += [{"index": index, "token": token, "stage": f"natural_{name}", "error": reason}
                         for name, reason in reasons.items() if reason]
            continue
        stage, seeds = "source_contract", model.seeds(token)
        try:
            contract = model.loader.original_source_contract(token)
            stage = "natural_outputs"
            outputs = {name: run.load(token, contract, seeds) for name, run in natural.items()}
            stage = "donor_cot"
            donors = {name: model.donor_cot(output.document) for name, output in outputs.items()}
        except NaturalMismatch:
            raise
        except Exception as exc:
            failures.append(runner.failure(index, token, stage, exc, **seeds))
            continue
        prepared[token] = index, seeds, outputs, donors
    if not prepared:
        raise ValueError("No requested token has valid natural outputs; there is no CoT-swap cohort: " +
                         "; ".join(f"{row['token']} ({row['stage']}): {row['error']}" for row in failures[:5]))
    excluded = len(failures)
    runner.prepare_output(s.result_dir, s.args.overwrite, remove=True)
    model.load()
    plan = ((CROSSED_ORIGINAL_IMAGE, True, "edited"), (CROSSED_EDITED_IMAGE, False, "original"))
    records, successful = [], []
    for token, (index, seeds, outputs, donors) in prepared.items():
        stage = "edited_input_check"
        try:
            checks = model.check_edited_inputs(token)
            stage = "metric_context"
            context = model.metric_context(token)
            trajectories = {NATURAL_ORIGINAL: outputs["original"].poses, NATURAL_EDITED: outputs["edited"].poses}
            provenance = {NATURAL_ORIGINAL: outputs["original"].provenance(),
                          NATURAL_EDITED: outputs["edited"].provenance()}
            for condition, original_image, cot_source in plan:
                stage = f"{condition}_inference"
                with model.inputs(token, original_image) as model_input:
                    prediction = model.force(model_input, donors[cot_source], seeds)
                stage = f"{condition}_save"
                donor = outputs[cot_source]
                root = s.result_dir / condition
                path = root / "trajectories" / f"{token}.npz"
                trajectories[condition] = runner.trajectory_poses(prediction)
                runner.save_trajectory(path, trajectories[condition], dt, **model.arrays(prediction))
                reasoning_path = root / "reasoning" / f"{token}.json"
                runner.save_json(reasoning_path, {**prediction.extra, "intervention": {
                    "type": "forced_cot_token_replay", "condition": condition, **CONDITION_SOURCES[condition],
                    "forced_cot": donors[cot_source], "assistant_prefix": model.forced_prefix(donors[cot_source]),
                    "donor_reasoning_path": str(donor.reasoning_path),
                    "donor_reasoning_sha256": donor.reasoning_sha256, **seeds, **model.cot_extras()}})
                provenance[condition] = {"inference": "forced_cot_token_replay", "trajectory_path": str(path),
                                         "reasoning_path": str(reasoning_path),
                                         "donor_reasoning_path": str(donor.reasoning_path),
                                         "donor_reasoning_sha256": donor.reasoning_sha256}
            stage = "cot_swap_metrics"
            record = build_cot_swap_record(
                token=token, trajectories={name: runner.metric_poses(poses, dt) for name, poses in trajectories.items()},
                actor_motion=context["actor_motion"], interval_length_s=dt, speed_mps=context["speed_mps"],
                seeds=seeds, longitudinal_acceleration_mps2=context.get("longitudinal_acceleration_mps2"),
                condition_provenance=provenance)
            record.update(native_pose_counts={name: len(poses) for name, poses in trajectories.items()}, **checks)
            for key, value in context.items():
                record.setdefault(key, value)
            records.append(record)
            successful.append(token)
            print(f"Completed {index}/{len(s.tokens)} token={token}", flush=True)
        except Exception as exc:
            failures.append(runner.failure(index, token, stage, exc, **seeds))
            runner.save_json(s.result_dir / "failed_samples.json", failures)
            print(f"Failed {index}/{len(s.tokens)} token={token} stage={stage}: {exc}", flush=True)
    runner.save_json(s.result_dir / "failed_samples.json", failures)
    runner.save_json(s.result_dir / "metrics" / "per_sample.json", records)
    summary = aggregate_cot_swap_records(records)
    summary.update({
        "mode": "cot_swap", "model": model.model, "dataset": model.dataset, "run_dir": str(model.run_dir),
        "result_dir": str(s.result_dir), "natural_original_result_dir": str(natural["original"].path),
        "natural_edited_result_dir": str(natural["edited"].path), "natural_runs": status,
        "evaluation_config_path": str(model.config_path), "identity": {"signature": s.signature, **model.identity()},
        "intervention": INTERVENTION, "seed_policy": "each token's natural seeds, shared by all four conditions",
        "condition_order": list(FACTORIAL_CONDITIONS), "interval_length_s": dt,
        "metric_horizon_s": runner.METRIC_HORIZON_S, "selected_token_count": len(s.tokens),
        "successful_token_count": len(records), "failed_sample_count": len({row["token"] for row in failures}),
        "excluded_before_inference_count": excluded, "gpus": runner.gpu_names(),
        "metric_definitions": METRIC_DEFINITIONS, **model.summary_extras(), **model.cot_extras()})
    runner.save_json(s.result_dir / "metrics" / "summary.json", summary)
    save_manifest(s.result_dir, mode="cot_swap", signature=s.signature, source_run_dir=model.run_dir,
                  selected_tokens=s.tokens, successful_tokens=successful, natural_runs=status, **model.cot_extras())
    print(f"Saved {len(records)}/{len(s.tokens)} CoT-swap samples to {s.result_dir}; {len(failures)} failures.",
          flush=True)
    return 0 if records else 1
