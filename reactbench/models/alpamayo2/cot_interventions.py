"""Alpamayo 2 CoT-swap and fixed-CoT evaluation on PAI-AV and NAVSIM.

Both modes build on completed natural runs from ``runner.py``: ``original/``
and the paired edited run, which must carry this configuration's exact cache
signature (including the GPU architecture class) and source contracts. Their
decoded reasoning is the CoT-swap donor text and their trajectories are the
natural reference conditions. Forced rollouts reuse each token's natural text
and diffusion seeds (``forced_cot.py``).

Output layouts follow the Alpamayo 1.5 experiments, so the shared CPU scorers,
inventories and plots apply unchanged: ``<run>_cot_swap/``
(a fresh directory, like natural runs) and ``<run>_fixed_cots/`` (resumable,
following the shared fixed-CoT contract).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ...analysis.cot.cot_swap import (
    CROSSED_EDITED_IMAGE, CROSSED_ORIGINAL_IMAGE, FACTORIAL_CONDITIONS, NATURAL_EDITED,
    NATURAL_ORIGINAL, REPLAY_CONDITIONS, REPLAY_EDITED, REPLAY_ORIGINAL, SavedCot,
    aggregate_cot_swap_records, build_cot_swap_record, intervention_reasoning,
)
from ...io import read_json, write_json
from ..common import predictions_dir
from ..original_cache import _source_contract_identity, file_sha256, load_cached_trajectory
from ...analysis.cot.fixed_cot import (
    BASELINE, CONDITIONS, ORIGINAL, SCHEMA_VERSION, ResumeConflict, aggregate, condition_metrics,
    contrast, digest, load_completed, load_config, object_type, prepare_manifest,
    resolve_conditions, save_json,
)
from ...analysis.cot.fixed_cot import add_arguments as add_fixed_cot_arguments
from . import forced_cot, runner
from .inference import paired_seed


DT, NATIVE_POSES, METRIC_POSES, METRIC_HORIZON_S = 0.1, 64, 40, 4.0
COT_SWAP_SUFFIX, REPLAY_CONTROLS_SUFFIX, FIXED_COTS_SUFFIX = "_cot_swap", "_replay_controls", "_fixed_cots"
INTERVENTION = ("decoded donor CoT forced as the assistant prefix {cot}<|traj_future_start|>; "
                "hidden states are recomputed with the selected image and the unchanged diffusion "
                "expert reads the prompt KV cache")
class NaturalMismatch(ValueError):
    """A completed natural run disagrees with this sample's inputs or seeds."""


class ModelInitializationError(RuntimeError):
    """Loading the model failed; no further sample can run."""


def parse_args(dataset, argv=None):
    parser = argparse.ArgumentParser(
        description=f"Alpamayo2 {dataset} CoT-swap or fixed-CoT evaluation", allow_abbrev=False)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--config", type=Path, default=runner.CONFIGS[dataset])
    parser.add_argument("--dataset", default=runner.DATASET_FOLDERS[dataset], help="Dataset folder of the predictions")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--cot-swap", action="store_true",
                      help="Force each natural CoT onto the other image: original image with the edited "
                           "CoT and edited image with the original CoT")
    add_fixed_cot_arguments(mode)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--max-samples", type=int)
    selection.add_argument("--tokens-file", type=Path, help="JSON object with an exact ordered 'tokens' list")
    parser.add_argument("--original-result-dir", type=Path,
                        help="Completed natural --original run (default: <results>/<input-parent>/original)")
    parser.add_argument("--edited-result-dir", type=Path,
                        help="Completed natural edited run (default: <results>/<input-parent>/<run>)")
    parser.add_argument("--scoring", choices=("score", "off"), default="score")
    parser.add_argument("--scoring-workers", type=int, default=1)
    args = parser.parse_args(argv)
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    return args


def natural_signatures(dataset, args, agent, loader, hardware, signature):
    """Signatures a compatible natural run may carry: this configuration's."""
    return {"current": runner.jsonable(signature)}


def token_seeds(agent, token):
    """The natural runs' per-token seeds, shared by every forced condition."""
    return paired_seed(token, agent.seed, "text"), paired_seed(token, agent.diffusion_seed_base, "diffusion")


@dataclass(frozen=True)
class NaturalOutput:
    poses: np.ndarray
    document: dict
    reasoning_path: Path
    reasoning_sha256: str
    trajectory_path: Path
    trajectory_sha256: str

    def provenance(self):
        return {"inference": "natural", "trajectory_path": str(self.trajectory_path),
                "trajectory_sha256": self.trajectory_sha256, "reasoning_path": str(self.reasoning_path),
                "reasoning_sha256": self.reasoning_sha256}


class NaturalRun:
    """A completed natural ``runner.py`` result: reference trajectories and donor CoTs."""

    def __init__(self, path, *, mode, signatures, hardware_class, run_dir=None):
        self.path, self.mode = Path(path).expanduser().resolve(), mode
        manifest_path = self.path / "evaluation_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Completed natural {mode} run is missing: {manifest_path}")
        manifest = read_json(manifest_path)
        if (not isinstance(manifest, dict) or manifest.get("schema_version") != 1
                or manifest.get("mode") != mode):
            raise ValueError(f"{manifest_path} is not a completed natural {mode} run")
        cached_class = ((manifest.get("signature") or {}).get("inference_options") or {}).get("hardware_class")
        if cached_class != hardware_class:
            raise ValueError(
                f"The natural {mode} run in {self.path} was computed on {cached_class} GPUs, but this job "
                f"runs on {hardware_class}; forced and natural rollouts must share a GPU architecture.")
        signature = next((name for name, value in signatures.items() if manifest.get("signature") == value), None)
        if signature is None:
            raise ValueError(f"Natural {mode} run settings do not match this evaluation: {manifest_path}")
        if run_dir is not None and manifest.get("source_run_dir") != str(run_dir):
            raise ValueError(f"{manifest_path} was computed from {manifest.get('source_run_dir')}, not {run_dir}")
        lists = {}
        for field in ("selected_tokens", "successful_tokens"):
            value = manifest.get(field)
            if (not isinstance(value, list) or not all(isinstance(token, str) and token for token in value)
                    or len(value) != len(set(value))):
                raise ValueError(f"Invalid {field} in {manifest_path}")
            lists[field] = set(value)
        self.selected, self.successful = lists["selected_tokens"], lists["successful_tokens"]
        if not self.successful <= self.selected:
            raise ValueError(f"{manifest_path} marks unselected tokens as successful")
        self.contracts = manifest.get("source_contracts")
        if not isinstance(self.contracts, dict):
            raise ValueError(f"Invalid source_contracts in {manifest_path}")
        self.status = {"path": str(self.path), "manifest_sha256": file_sha256(manifest_path),
                       "signature": signature}

    def unavailable(self, token):
        """Why this token has no natural output, or None."""
        if token not in self.selected:
            return f"Never attempted by the natural {self.mode} run in {self.path}"
        if token not in self.successful:
            return f"Natural {self.mode} inference failed or the sample was excluded"
        return None

    def load(self, token, contract, *, text_seed, diffusion_seed):
        """Saved outputs of a successful token; input or seed mismatches are fatal."""
        if (_source_contract_identity(self.contracts.get(token))
                != _source_contract_identity(runner.jsonable(contract))):
            raise NaturalMismatch(f"Source provenance differs from the natural {self.mode} run for {token}")
        reasoning_path = self.path / "reasoning" / f"{token}.json"
        document = read_json(reasoning_path)
        if not isinstance(document, dict):
            raise FileNotFoundError(f"Missing natural {self.mode} reasoning: {reasoning_path}")
        if (document.get("text_seed"), document.get("diffusion_seed")) != (text_seed, diffusion_seed):
            raise NaturalMismatch(f"Natural {self.mode} seeds differ for {token}: {reasoning_path}")
        trajectory_path = self.path / "trajectories" / f"{token}.npz"
        poses = load_cached_trajectory(trajectory_path, expected_interval_length_s=DT)
        if poses.shape != (NATIVE_POSES, 3):
            raise ValueError(f"Natural {self.mode} trajectory must hold {NATIVE_POSES} poses: {trajectory_path}")
        return NaturalOutput(poses, document, reasoning_path, file_sha256(reasoning_path),
                             trajectory_path, file_sha256(trajectory_path))


def load_model(agent):
    """Load the weights, then re-check the pinned upstream source.

    An edit made while the model was being imported (2026-10-13: a debugging
    breakpoint in the shared checkout) would otherwise run silently.
    """
    agent.load()
    agent.identity()


def save_archive(condition_root, token, prediction):
    """``runner.py``'s archive contract: 64 rear-axle x/y/yaw poses plus raw XYZ and rotations."""
    poses = np.asarray(prediction.poses, dtype=np.float32)
    xyz, rotations = np.asarray(prediction.xyz), np.asarray(prediction.rotations)
    if (poses.shape != (NATIVE_POSES, 3) or xyz.shape != (NATIVE_POSES, 3)
            or rotations.shape != (NATIVE_POSES, 3, 3)):
        raise ValueError("Prediction must contain all 64 poses, XYZ positions, and rotation matrices")
    if not all(np.isfinite(value).all() for value in (poses, xyz, rotations)):
        raise ValueError("Prediction contains non-finite values")
    path = condition_root / "trajectories" / f"{token}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, trajectory=poses, pred_xyz=xyz, pred_rot=rotations,
                                interval_length_s=np.float32(DT), ego_origin="rear_axle")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return poses, path


def input_provenance(loader, token, model_input, contract, metadata_path):
    """Stable fingerprint of the edited inputs.

    PAI-AV rear frames are decoded into temporary paths, so archive frames are
    identified by the source contract and edited frames by their file hashes.
    """
    images = {str(camera): [file_sha256(Path(path)) for path in paths]
              for camera, paths in loader.edited_camera_paths(token).items()}
    inputs = {key: value for key, value in model_input.items() if key != "camera_paths"}
    return {"model_input_sha256": digest(inputs), "edited_image_sha256": images,
            "source_contract_sha256": digest(_source_contract_identity(runner.jsonable(contract))),
            "metadata_path": str(metadata_path), "metadata_sha256": file_sha256(Path(metadata_path))}


def metric_poses(poses):
    metric = np.asarray(poses)[:METRIC_POSES]
    if metric.shape != (METRIC_POSES, 3):
        raise ValueError(f"Expected at least {METRIC_POSES} poses for the four-second metrics")
    return metric


def run_cot_swap(s):
    agent, loader, replay = s.agent, s.loader, False
    natural = {"original": NaturalRun(s.original_dir, mode="original", signatures=s.natural_signatures,
                                      hardware_class=s.hardware["class"]),
               "edited": NaturalRun(s.edited_dir, mode="paired", signatures=s.natural_signatures,
                                    hardware_class=s.hardware["class"], run_dir=s.run_dir)}
    unattempted = sorted({token for token in s.tokens for run in natural.values() if token not in run.selected})
    if unattempted:
        raise ValueError(f"The natural runs never attempted {len(unattempted)} requested token(s): "
                         f"{unattempted[:10]}. Select tokens that both natural runs attempted.")
    # Validate every natural output before claiming the destination: seed or
    # provenance mismatches are fatal; missing outputs or unclosed CoTs exclude a token.
    failures, prepared = [], {}
    for token in s.tokens:
        reasons = {name: run.unavailable(token) for name, run in natural.items()}
        if any(reasons.values()):
            failures.extend({"token": token, "stage": f"natural_{name}", "error": reason}
                            for name, reason in reasons.items() if reason)
            continue
        stage = "source_contract"
        text_seed, diffusion_seed = token_seeds(agent, token)
        try:
            contract = loader.original_source_contract(token)
            stage = "natural_outputs"
            outputs = {name: run.load(token, contract, text_seed=text_seed, diffusion_seed=diffusion_seed)
                       for name, run in natural.items()}
            stage = "donor_cot"
            donors = {name: SavedCot(text=forced_cot.natural_cot(output.document), path=output.reasoning_path,
                                     sha256=output.reasoning_sha256) for name, output in outputs.items()}
        except NaturalMismatch:
            raise
        except Exception as exc:
            failures.append(runner.failure(token, stage, exc))
            continue
        prepared[token] = SimpleNamespace(
            text_seed=text_seed, diffusion_seed=diffusion_seed, donors=donors,
            poses={NATURAL_ORIGINAL: outputs["original"].poses, NATURAL_EDITED: outputs["edited"].poses},
            provenance={NATURAL_ORIGINAL: outputs["original"].provenance(),
                        NATURAL_EDITED: outputs["edited"].provenance()})
    if not prepared:
        raise ValueError("No requested token has valid natural outputs; there is no CoT-swap cohort: "
                         + "; ".join(f"{row['token']} ({row['stage']}): {row['error']}" for row in failures[:5]))
    excluded_count = len(failures)
    # As in runner.py: check the GPU allocation before claiming the destination,
    # claim it atomically before loading weights, and release it if loading fails.
    agent.prepare()
    s.result_dir.parent.mkdir(parents=True, exist_ok=True)
    s.result_dir.mkdir(exist_ok=False)
    try:
        load_model(agent)
    except BaseException:
        try:
            s.result_dir.rmdir()
        except OSError:
            pass
        raise
    plan = (("original", ((CROSSED_ORIGINAL_IMAGE, "edited"),) + (((REPLAY_ORIGINAL, "original"),) if replay else ())),
            ("edited", ((CROSSED_EDITED_IMAGE, "original"),) + (((REPLAY_EDITED, "edited"),) if replay else ())))
    adapter = forced_cot.adapter_identity()
    records, successful = [], []
    for index, (token, sample) in enumerate(prepared.items(), 1):
        text_seed, diffusion_seed, donors = sample.text_seed, sample.diffusion_seed, sample.donors
        stage = "edited_input_check"
        try:
            input_check = loader.check_edited_inputs(token)
            stage = "metric_context"
            context = loader.metric_context(token)
            forced = {}
            for image_source, conditions in plan:
                stage = f"{image_source}_inputs"
                with loader.materialize_inputs(token, original=image_source == "original") as paths:
                    model_input = loader.model_input(token, paths)
                    for condition, cot_source in conditions:
                        stage = f"{condition}_inference"
                        forced[condition] = cot_source, forced_cot.compute_forced_trajectory(
                            agent, model_input, donors[cot_source].text,
                            text_seed=text_seed, diffusion_seed=diffusion_seed)
            stage = "save_predictions"
            trajectories, provenance = dict(sample.poses), dict(sample.provenance)
            for condition, (cot_source, prediction) in forced.items():
                donor = donors[cot_source]
                reasoning = intervention_reasoning(prediction.extra, condition=condition, donor_cot=donor,
                                                   diffusion_seed=diffusion_seed)
                reasoning["intervention"].update(
                    text_seed=text_seed, assistant_prefix=forced_cot.forced_cot_prefix(donor.text),
                    forced_cot_adapter=adapter)
                poses, trajectory_path = save_archive(s.result_dir / condition, token, prediction)
                reasoning_path = s.result_dir / condition / "reasoning" / f"{token}.json"
                write_json(reasoning_path, runner.jsonable(reasoning))
                trajectories[condition] = poses
                provenance[condition] = {
                    "inference": ("forced_cot_token_replay_control" if condition in REPLAY_CONDITIONS
                                  else "forced_cot_token_replay"),
                    "trajectory_path": str(trajectory_path), "reasoning_path": str(reasoning_path),
                    "donor_reasoning_path": str(donor.path), "donor_reasoning_sha256": donor.sha256}
            stage = "cot_swap_metrics"
            record = build_cot_swap_record(
                token=token,
                trajectories={name: metric_poses(trajectories[name]) for name in FACTORIAL_CONDITIONS},
                replay_trajectories=({name: metric_poses(trajectories[name]) for name in REPLAY_CONDITIONS}
                                     if replay else None),
                actor_motion=context["actor_motion"], interval_length_s=DT, speed_mps=context["speed_mps"],
                diffusion_seed=diffusion_seed,
                longitudinal_acceleration_mps2=context.get("longitudinal_acceleration_mps2"),
                condition_provenance=provenance)
            record.update(text_seed=text_seed, dataset=s.dataset, native_pose_count=NATIVE_POSES,
                          image_timestamps_us=model_input["image_timestamps_us"],
                          navigation_command=model_input.get("navigation_command"), **input_check)
            for key, value in context.items():
                record.setdefault(key, value)
            records.append(record)
            successful.append(token)
        except Exception as exc:
            failures.append(runner.failure(token, stage, exc))
        print(f"[{index}/{len(prepared)}] {token}: {stage}", flush=True)

    metrics_dir = s.result_dir / "metrics"
    write_json(s.result_dir / "failed_samples.json", runner.jsonable(failures))
    write_json(metrics_dir / "per_sample.json", runner.jsonable(records))
    summary = aggregate_cot_swap_records(records)
    summary.update({
        "mode": "cot_swap", "dataset": s.dataset, "run_dir": str(s.run_dir),
        "run_config_path": str(loader.run_config_path), "annotations_path": str(loader.annotations_path),
        "result_dir": str(s.result_dir), "natural_original_result_dir": str(natural["original"].path),
        "natural_edited_result_dir": str(natural["edited"].path),
        "natural_runs": {name: run.status for name, run in natural.items()},
        "evaluation_config_path": str(s.args.config.resolve()), "model_name_or_path": s.cfg["model_name_or_path"],
        "identity": {"signature": s.signature, **s.identity}, "forced_cot_adapter": adapter,
        "intervention": INTERVENTION,
        "seed_policy": "each token's natural text and diffusion seeds, shared by all natural, crossed "
                       "and replay conditions; diffusion noise is seeded at the sampler boundary",
        "condition_order": list(FACTORIAL_CONDITIONS), "replay_controls_enabled": bool(replay),
        "interval_length_s": DT, "metric_horizon_s": METRIC_HORIZON_S, "native_pose_count": NATIVE_POSES,
        "input_contract": loader.camera_profile, "selected_token_count": len(s.tokens),
        "successful_token_count": len(records), "failed_sample_count": len({row["token"] for row in failures}),
        "excluded_before_inference_count": excluded_count,
        "compact_run_preflight": loader.preflight_report(), "hardware": s.hardware,
        "gpu_placement": agent.placement_report(),
        "metric_definitions": {
            "condition_metrics": "distance driven over the first four seconds; collision metrics come "
                                 "from protocol V2 scoring (metrics/<dataset>_at_fault_*)",
            "contrast_ade_m": "mean corresponding-pose x/y displacement from the named from_condition "
                              "to to_condition",
            "contrast_delta": "to_condition value minus from_condition value",
            "interaction_difference_in_differences": "(edited-image effect with edited CoT) minus "
                                                     "(edited-image effect with original CoT)",
        },
    })
    write_json(metrics_dir / "summary.json", runner.jsonable(summary))
    write_json(s.result_dir / "evaluation_manifest.json", runner.jsonable({
        "schema_version": 1, "mode": "cot_swap", "signature": s.signature, "source_run_dir": str(s.run_dir),
        "selected_tokens": s.tokens, "successful_tokens": successful,
        "natural_runs": {name: run.status for name, run in natural.items()},
        "replay_controls": bool(replay), "forced_cot_adapter": adapter}))
    print(f"Saved {len(records)}/{len(s.tokens)} CoT-swap samples to {s.result_dir}; "
          f"{len(failures)} failures.", flush=True)
    return 0 if records else 1


def save_fixed(condition_root, token, prediction, intervention):
    """Trajectory first; the reasoning JSON is the commit marker read by load_completed."""
    _, path = save_archive(condition_root, token, prediction)
    save_json(condition_root / "reasoning" / f"{token}.json", {
        "intervention": intervention, "model_output": runner.jsonable(prediction.extra),
        "trajectory_sha256": file_sha256(path)})
    return load_cached_trajectory(path, expected_interval_length_s=DT)


def run_fixed_cots(s):
    agent, loader, root = s.agent, s.loader, s.result_dir
    fixed_path = s.args.fixed_cots.expanduser().resolve()
    fixed = load_config(fixed_path)
    if fixed["stop_duration_s"] > METRIC_HORIZON_S:
        raise ValueError("stop_duration_s exceeds the metric horizon")
    kind = object_type(loader.run_config)
    texts = resolve_conditions(fixed, kind)
    for text in texts.values():
        forced_cot.forced_cot_prefix(text)
    # Missing natural runs are reported, as in Alpamayo 1.5; incompatible ones are fatal.
    natural, baselines = {}, {}
    for name, path, mode, run_dir in ((ORIGINAL, s.original_dir, "original", None),
                                      (BASELINE, s.edited_dir, "paired", s.run_dir)):
        try:
            natural[name] = NaturalRun(path, mode=mode, signatures=s.natural_signatures,
                                       hardware_class=s.hardware["class"], run_dir=run_dir)
            baselines[name] = natural[name].status
        except FileNotFoundError as exc:
            natural[name] = None
            baselines[name] = {"missing": str(exc)}
    identity = {
        "schema_version": SCHEMA_VERSION, "backend": f"alpamayo2-{s.dataset}", "dataset": s.dataset,
        "signature": s.signature, "model_identity": s.identity, "forced_cot_adapter": forced_cot.adapter_identity(),
        "run_dir": str(s.run_dir), "run_config_sha256": file_sha256(loader.run_config_path),
        "fixed_config": fixed, "fixed_config_sha256": file_sha256(fixed_path), "conditions": texts,
        "object_type": kind, "insertion_timestep": loader.run_config.get("future_timestep"),
        "metric_horizon_s": METRIC_HORIZON_S,
        "baseline_dirs": {BASELINE: str(s.edited_dir), ORIGINAL: str(s.original_dir)},
    }
    agent.prepare()  # Check the GPU allocation before writing any output.
    prepare_manifest(root, identity, s.tokens)
    print(f"Fixed-CoT alpamayo2-{s.dataset}: {len(s.tokens)} samples x {len(texts)} conditions; {root}", flush=True)
    print(f"Baseline availability: {baselines}", flush=True)
    loaded = False  # A completely resumed run loads no weights.
    records, failures = [], []
    for index, token in enumerate(s.tokens, 1):
        text_seed, diffusion_seed = token_seeds(agent, token)
        record = {"token": token, "output_stem": token, "conditions": {}, "contrasts": {},
                  "text_seed": text_seed, "diffusion_seed": diffusion_seed}
        stage = "source_contract"
        try:
            contract = loader.original_source_contract(token)
            stage = "edited_input_check"
            record.update(loader.check_edited_inputs(token))
            stage = "metric_context"
            context = loader.metric_context(token)
            actor, geometry, speed = context["actor_motion"], context.get("vehicle_geometry"), context["speed_mps"]
            if object_type(actor) != kind:
                raise ValueError(f"Actor metadata differs from run object type for {token}")
            record.update(dataset=s.dataset, actor_motion=actor, ego_geometry=geometry, speed_mps=speed,
                          longitudinal_acceleration_mps2=context.get("longitudinal_acceleration_mps2"),
                          placement_metadata_path=context["placement_metadata_path"], ego_origin="rear_axle")
            stage = "natural_baselines"
            trajectories, issues, references = {}, {}, {}
            for name, run in natural.items():
                reason = "No compatible natural run" if run is None else run.unavailable(token)
                if reason:
                    issues[name] = reason
                    continue
                try:
                    output = run.load(token, contract, text_seed=text_seed, diffusion_seed=diffusion_seed)
                except NaturalMismatch:
                    raise
                except (FileNotFoundError, KeyError, ValueError) as exc:
                    issues[name] = f"{type(exc).__name__}: {exc}"
                    continue
                trajectories[name] = output.poses
                references[name] = {"path": str(output.trajectory_path), "sha256": output.trajectory_sha256}
            record.update(baseline_issues=issues, baseline_provenance=references)
            stage = "edited_inputs"
            with loader.materialize_inputs(token, original=False) as paths:
                model_input = loader.model_input(token, paths)
                provenance = input_provenance(loader, token, model_input, contract,
                                              context["placement_metadata_path"])
                record["provenance"] = provenance
                for name, text in texts.items():
                    intervention = {"condition": name, "forced_cot": text,
                                    "assistant_prefix": forced_cot.forced_cot_prefix(text),
                                    "text_sha256": digest(text), "text_seed": text_seed,
                                    "diffusion_seed": diffusion_seed, "input_provenance": provenance,
                                    "identity_sha256": digest(identity)}
                    condition_root = root / name
                    # A conflicting committed result is a hard error, not a failed sample.
                    poses = load_completed(condition_root, token, intervention, DT)
                    try:
                        if poses is None:
                            if not loaded:
                                try:
                                    load_model(agent)
                                except Exception as exc:
                                    raise ModelInitializationError(str(exc)) from exc
                                loaded = True
                            prediction = forced_cot.compute_forced_trajectory(
                                agent, model_input, text, text_seed=text_seed, diffusion_seed=diffusion_seed)
                            poses = save_fixed(condition_root, token, prediction, intervention)
                        trajectories[name] = poses
                    except ModelInitializationError:
                        raise
                    except Exception as exc:
                        failures.append({"token": token, "condition": name, "stage": "inference",
                                         "error": str(exc)})
                        print(f"{token} {name}: {exc}", flush=True)
            stage = "metrics"
            metrics = {}
            for name, poses in trajectories.items():
                try:
                    poses = metric_poses(poses)
                    record["conditions"][name] = condition_metrics(
                        poses, dt=DT, initial_speed=speed, config=fixed)
                    metrics[name] = poses
                except Exception as exc:
                    failures.append({"token": token, "condition": name, "stage": "metrics", "error": str(exc)})
            for before in metrics:
                for after in CONDITIONS:
                    if after in metrics and before != after:
                        record["contrasts"][f"{before}->{after}"] = contrast(
                            metrics[before], metrics[after],
                            record["conditions"][before], record["conditions"][after])
            # Small arrays permit standalone CPU sample plots with no source caches.
            record["trajectories"] = runner.jsonable(metrics)
        except (ResumeConflict, NaturalMismatch, ModelInitializationError):
            raise
        except Exception as exc:
            failures.append({"token": token, "condition": None, "stage": stage, "error": str(exc)})
        records.append(record)
        save_json(root / "metrics" / "samples" / f"{token}.json", record)
        save_json(root / "failed_samples.json", failures)
        print(f"[{index}/{len(s.tokens)}] {token}: "
              f"{sum(name in record['conditions'] for name in CONDITIONS)}/{len(CONDITIONS)} conditions", flush=True)
    save_json(root / "metrics" / "per_sample.json", records)
    summary = {**aggregate(records, s.tokens), "mode": "fixed_cots", "dataset": s.dataset,
               "run_dir": str(s.run_dir), "identity": identity, "baselines": baselines,
               "failure_count": len(failures), "result_dir": str(root), "interval_length_s": DT,
               "metric_horizon_s": METRIC_HORIZON_S, "native_pose_count": NATIVE_POSES,
               "hardware": s.hardware, "gpu_placement": agent.placement_report(),
               "metric_definitions": {
                   "speed": "Planar segment distance / dt, origin included; times are segment midpoints",
                   "stop": "First low-speed segment start with at least stop_duration_s consecutive low-speed segments",
                   "stop_time_null": "No sustained stop within the four-second horizon",
                   "initially_stationary": "Recorded initial speed <= stop_speed_mps; moving_stopping_rate excludes these samples",
               }}
    save_json(root / "metrics" / "summary.json", summary)
    print(f"Saved fixed-CoT results: {root}; paired comparison count={summary['comparison_count']}", flush=True)
    return 0 if any(record["conditions"].keys() & set(CONDITIONS) for record in records) else 1


def main(dataset, argv=None):
    args = parse_args(dataset, argv)
    cfg, agent = runner.load_agent(dataset, args.config)
    run_dir = args.run_dir.expanduser().resolve()
    group = predictions_dir(runner.MODEL_NAME, args.dataset, "original").parent
    original_dir = (args.original_result_dir or group / "original").expanduser().resolve()
    edited_dir = (args.edited_result_dir or group / run_dir.name).expanduser().resolve()
    suffix = COT_SWAP_SUFFIX if args.cot_swap else FIXED_COTS_SUFFIX
    result_dir = group / f"{run_dir.name}{suffix}"
    if result_dir in (original_dir, edited_dir):
        raise ValueError("Intervention results must not be written into a natural run directory")
    if args.cot_swap and result_dir.exists():
        raise FileExistsError(f"Result directory already exists: {result_dir}. Use a fresh results root.")
    with runner.load_run(dataset, run_dir) as loader:
        tokens = runner.selected_tokens(args, loader.tokens)
        hardware = agent.hardware()
        identity, signature = runner.signature_for(dataset, args, agent, loader, hardware)
        setup = SimpleNamespace(
            dataset=dataset, args=args, cfg=cfg, agent=agent, loader=loader, run_dir=run_dir, tokens=tokens,
            hardware=hardware, identity=identity, signature=signature, result_dir=result_dir,
            original_dir=original_dir, edited_dir=edited_dir,
            natural_signatures=natural_signatures(dataset, args, agent, loader, hardware, signature))
        status = run_cot_swap(setup) if args.cot_swap else run_fixed_cots(setup)

    if dataset == "navsim":
        from ...scoring.navsim import postprocess_evaluation
        postprocess_evaluation(result_dir, args.scoring, args.scoring_workers)
    else:
        from ...scoring.paiav import postprocess_evaluation
        postprocess_evaluation(result_dir, args.scoring)
    return status
