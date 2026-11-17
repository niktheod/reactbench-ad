"""Shared core of the model runners: command line, output layout, per-token loop, bookkeeping and scoring.

Every runner evaluates one model on one benchmark variant in two passes::

    python -m reactbench.models.<model>.eval VARIANT_DIR --original   # the original scenes, cached once per model
    python -m reactbench.models.<model>.eval VARIANT_DIR              # the edited scenes of VARIANT_DIR, then scoring

Predictions go to ``<outputs>/predictions/<model>/<reactnav|reactpai>/<variant>`` (``original`` for the original
scenes): ``trajectories/<token>.npz`` (``trajectory``: N x 3 rear-axle x/y/heading poses in the current ego frame,
the first at t = dt, at least 4 s; ``interval_length_s``), ``metrics/per_sample.json`` and
``metrics/summary.json`` (the bookkeeping the scorer reads), ``evaluation_manifest.json`` (``original_cache.py``)
and ``failed_samples.json``. An edited run pairs each token with the cached original prediction of the same
configuration; both members of a pair use the same per-token seeds (``seeds.py``). An existing, non-empty output
directory is only replaced with ``--overwrite``.

A runner subclasses ``Runner`` with the model-specific parts: the agent (``setup``), the cache signature, the seeded
prediction of one token (``predict``) and its extra outputs. ``drivewam/eval.py`` is the shortest example.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from .. import paths
from ..io import file_digest, finite_json, jsonable, read_json, write_json
from .original_cache import cache_signature, load_original_cache, save_manifest

DATASETS = {"reactnav": "navsim", "reactpai": "pai-av"}
METRIC_HORIZON_S = 4.0
# The sample_record arguments a loader's metric context provides.
CONTEXT_KEYS = ("actor_motion", "speed_mps", "longitudinal_acceleration_mps2", "vehicle_geometry")


# --- helpers ---------------------------------------------------------------------------------------------------

def predictions_dir(model: str, dataset: str, variant_name: str) -> Path:
    """Where a model's predictions for one variant go: <outputs>/predictions/<model>/<dataset>/<variant>."""
    return paths.outputs("predictions", model, dataset, variant_name)


def expand_config(text: str) -> str:
    """Expand the location placeholders of the model configurations."""
    def location(key, default=None, function=None):
        try:
            return str(function() if function else paths.get(key, default))
        except KeyError:
            return "{" + key + "}"
    replacements = {"{checkpoints}": location("checkpoints", "checkpoints"),
                    "{third_party}": str(paths.REPO_ROOT / "third_party"),
                    "{sensor_blobs_test}": location("sensor_blobs_test", function=lambda: paths.navsim_sensor_blobs("test"))}
    for placeholder, value in replacements.items():
        text = text.replace(placeholder, value)
    return text


def config_path(value, directory: Path) -> Path:
    """A configuration NAME (``<directory>/<NAME>.yaml``) or PATH."""
    named = Path(directory) / f"{value}.yaml"
    path = (named if named.is_file() else Path(value)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration {value!r} is neither a name in {directory} nor a file")
    return path


def read_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(expand_config(Path(path).read_text()))
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return config


def select_tokens(available, *, tokens_file=None, max_samples=None) -> list[str]:
    """All tokens of the variant, its first ``max_samples``, or the exact ordered ``{"tokens": [...]}`` of a file."""
    available = list(available)
    if tokens_file is not None:
        document = read_json(tokens_file)
        tokens = document.get("tokens") if isinstance(document, dict) else None
        if (not isinstance(tokens, list) or not tokens or len(tokens) != len(set(tokens))
                or not all(isinstance(token, str) and token for token in tokens)):
            raise ValueError(f"{tokens_file} needs a non-empty 'tokens' list of unique strings")
        missing = sorted(set(tokens) - set(available))
        if missing:
            raise ValueError(f"Requested tokens are not in the variant: {missing[:10]}")
        return tokens
    return available if max_samples is None else available[:max_samples]


def trajectory_poses(prediction: Any) -> np.ndarray:
    """The (N, 3) float32 poses of a prediction (an array or an object with ``poses``)."""
    poses = np.asarray(getattr(prediction, "poses", prediction), dtype=np.float32)
    if poses.ndim != 2 or poses.shape[1] != 3 or poses.shape[0] == 0 or not np.isfinite(poses).all():
        raise ValueError(f"Expected finite (N, 3) predicted poses, got shape {poses.shape}")
    return poses


def metric_poses(poses: np.ndarray, interval_length_s: float) -> np.ndarray:
    """The first four seconds of a trajectory, the part that is scored."""
    count = int(round(METRIC_HORIZON_S / interval_length_s))
    if not np.isclose(count * interval_length_s, METRIC_HORIZON_S, atol=1e-6) or len(poses) < count:
        raise ValueError(f"A trajectory of {len(poses)} poses at {interval_length_s} s does not cover "
                         f"{METRIC_HORIZON_S} s")
    return np.asarray(poses)[:count]


def save_trajectory(path: Path, poses: np.ndarray, interval_length_s: float, **arrays) -> None:
    """``trajectory`` and ``interval_length_s`` (plus any model arrays), written atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, trajectory=poses, interval_length_s=np.float32(interval_length_s), **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_json(path: Path, value: Any) -> None:
    """Runner outputs: numpy and tensor values made plain, non-finite floats written as null."""
    write_json(path, finite_json(jsonable(value)))


def failure(index, token, stage, exc, **fields) -> dict[str, Any]:
    """The one failure schema of the runners."""
    return {"index": index, "token": token, "stage": stage, **fields, "error_type": type(exc).__name__,
            "error": str(exc), "traceback": traceback.format_exc()}


def prepare_output(path: Path, overwrite: bool, *, remove: bool) -> None:
    """Refuse an existing non-empty output directory unless ``overwrite``; ``remove`` deletes it then."""
    if path.is_dir() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{path} already holds results; pass --overwrite to replace them")
        if remove:
            shutil.rmtree(path)
    if remove:
        path.mkdir(parents=True, exist_ok=True)


def paiav_metric_context(loader, token) -> dict[str, Any]:
    """A ReactPAI sample's bookkeeping from ``PAIAVCompactVariant`` (cf. ``NAVSIMVariant.metric_context``)."""
    from ..scoring.actor_motion import actor_motion_from_metadata
    metadata, metadata_path = loader.metadata(token)
    geometry, geometry_source = loader.ego_vehicle_geometry(token)
    return {"actor_motion": actor_motion_from_metadata(metadata, camera_name=None),
            "vehicle_geometry": geometry, "speed_mps": loader.current_speed_mps(token),
            "longitudinal_acceleration_mps2": loader.current_longitudinal_acceleration_mps2(token),
            "placement_metadata_path": str(metadata_path), "ego_geometry_source": geometry_source,
            "ego_origin": "rear_axle"}


def score(result_dir: Path, dataset: str, workers: int = 1) -> bool:
    """Score an edited run (``scoring.navsim``/``scoring.paiav``); False if scoring failed."""
    if dataset == "reactnav":
        from ..scoring.navsim import PREFIX, postprocess_evaluation
        arguments = ("score", workers)
    else:
        from ..scoring.paiav import PREFIX, postprocess_evaluation
        arguments = ("score",)
    error = Path(result_dir) / "metrics" / f"{PREFIX}_error.json"
    error.unlink(missing_ok=True)
    postprocess_evaluation(result_dir, *arguments)
    return not error.is_file()


def gpu_names() -> list[str]:
    try:
        import torch
        return [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] \
            if torch.cuda.is_available() else []
    except Exception:
        return []


# --- the runner ------------------------------------------------------------------------------------------------

class Runner:
    """One model on one benchmark variant. Subclasses fill in the model-specific parts below."""

    model = ""                 # predictions folder; a configuration's ``name`` replaces it (e.g. recogdrive_il)
    datasets = ("reactnav",)   # supported datasets; the first is the default
    config_dir: Path | None = None   # the model's NAME.yaml configurations
    default_config = ""               # a NAME in config_dir

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.dataset = args.dataset
        self.original = bool(getattr(args, "original", False))
        self.config_path = config_path(args.config or self.default_config, self.config_dir)
        self.config = read_config(self.config_path)
        self.model = self.config.pop("name", type(self).model)
        self.variant_dir = Path(args.variant_dir).expanduser().resolve()
        self.loader = None

    # --- model-specific parts ----------------------------------------------------------------------------------

    def open_variant(self):
        """The dataset loader (a context manager): ``NAVSIMVariant`` or ``PAIAVCompactVariant`` (or a model subclass)."""
        if self.dataset == "reactnav":
            from ..datasets.navsim import NAVSIMVariant
            return NAVSIMVariant(self.variant_dir)
        from ..datasets.paiav import PAIAVCompactVariant
        return PAIAVCompactVariant(self.variant_dir)

    def setup(self) -> None:
        """Build the agent from ``self.config`` (no weights yet); set ``self.interval_length_s``."""
        raise NotImplementedError

    def cache_fields(self) -> dict[str, Any]:
        """evaluator, seed_name and seed_base of the cache signature; optionally inference_options and source."""
        raise NotImplementedError

    def identity(self) -> dict[str, Any]:
        """Model provenance recorded in the summary next to the signature."""
        return {}

    def load(self) -> None:
        """Load the weights; called once the original cache is known to fit."""
        self.agent.initialize()

    def seeds(self, token: str) -> dict[str, int]:
        return {}

    def predict(self, token: str, seeds: Mapping[str, int]) -> Any:
        """The prediction for one token (``self.original`` selects the scene); seed right before inference."""
        raise NotImplementedError

    def arrays(self, prediction) -> dict[str, Any]:
        """Extra arrays saved with the trajectory."""
        return {}

    def save_extras(self, token: str, prediction) -> dict[str, Any]:
        """Save extra outputs (texts, reasoning); return record fields."""
        return {}

    def check_edited_inputs(self, token: str) -> dict[str, Any]:
        """Reject an edited sample the model cannot be shown consistently; return record fields."""
        return {}

    def excluded_tokens(self) -> dict[str, str]:
        """{token: reason} of the edited run's samples this model is not evaluated on (not failures)."""
        return {}

    def summary_extras(self) -> dict[str, Any]:
        return {}

    # --- shared parts -------------------------------------------------------------------------------------------

    def source(self) -> dict[str, Any]:
        """The inputs an original cache depends on, besides the per-token source contracts."""
        if self.dataset == "reactnav":
            return {"dataset": self.dataset, "annotations_sha256": file_digest(self.loader.annotations_path)}
        return {"dataset": self.dataset, "dataset_group": self.variant_dir.parent.name}

    def signature(self) -> dict[str, Any]:
        if self.dataset == "reactnav":
            self.loader.annotations  # noqa: B018  (reads the annotations and sets annotations_path)
        fields = dict(self.cache_fields())
        source = {**self.source(), **fields.pop("source", {})}
        return cache_signature(config_path=self.config_path, interval_length_s=self.interval_length_s,
                               inference_options=fields.pop("inference_options", {}), source=source, **fields)

    def metric_context(self, token: str) -> dict[str, Any]:
        if self.dataset == "reactnav":
            return self.loader.metric_context(token)
        return paiav_metric_context(self.loader, token)

    def save(self, token: str, prediction) -> np.ndarray:
        poses = trajectory_poses(prediction)
        metric_poses(poses, self.interval_length_s)
        save_trajectory(self.result_dir / "trajectories" / f"{token}.npz", poses, self.interval_length_s,
                        **self.arrays(prediction))
        return poses

    def run(self) -> int:
        os.environ.setdefault("MPLBACKEND", "Agg")
        args, mode = self.args, "original" if self.original else "edited"
        self.result_dir = predictions_dir(self.model, self.dataset,
                                          "original" if self.original else self.variant_dir.name)
        self.original_dir = Path(args.original_result_dir or predictions_dir(self.model, self.dataset, "original"))
        self.original_dir = self.original_dir.expanduser().resolve()
        prepare_output(self.result_dir, args.overwrite, remove=False)
        with self.open_variant() as loader:
            self.loader = loader
            tokens = select_tokens(loader.tokens, tokens_file=args.tokens_file, max_samples=args.max_samples)
            self.setup()
            self.excluded_inputs = {} if self.original else {
                token: reason for token, reason in self.excluded_tokens().items() if token in tokens}
            tokens = [token for token in tokens if token not in self.excluded_inputs]
            signature = self.signature()
            # Per-token provenance of the original inputs, where the loader records it (ReactPAI, SimWAM, Alpamayo 2).
            failures, contracts, eligible = [], {}, list(tokens)
            with_contracts = hasattr(loader, "original_source_contract")
            if with_contracts:
                for index, token in enumerate(tokens, 1):
                    try:
                        contracts[token] = loader.original_source_contract(token)
                    except Exception as exc:
                        failures.append(failure(index, token, "source_contract", exc))
                eligible = [token for token in tokens if token in contracts]
            excluded = []
            if not self.original:
                cohort = load_original_cache(self.original_dir, tokens=eligible, signature=signature,
                                             source_contracts=contracts if with_contracts else None)
                excluded = cohort.excluded_failed_tokens
                failures += [{"index": tokens.index(token) + 1, "token": token, "stage": "original_cache",
                              "error_type": "ExcludedPair", "error": "The original prediction failed; no pair"}
                             for token in excluded]
                eligible = cohort.eligible_tokens
            eligible = set(eligible)
            prepare_output(self.result_dir, args.overwrite, remove=True)
            print(f"{self.model}: {len(eligible)} {mode} samples of {self.variant_dir}", flush=True)
            if eligible:
                self.load()
            records, successful = [], []
            for index, token in enumerate(tokens, 1):
                if token not in eligible:
                    continue
                stage, seeds = "input", {}
                try:
                    seeds = self.seeds(token)
                    record = {"token": token, "output_stem": token, **seeds}
                    if not self.original:
                        stage = "edited_input_check"
                        record.update(self.check_edited_inputs(token))
                        stage = "metric_context"
                        from ..scoring.records import sample_record
                        context = self.metric_context(token)
                        record = {**context, **sample_record(token=token, output_stem=token, **{
                            k: context[k] for k in CONTEXT_KEYS if k in context}), **record}
                    stage = f"{mode}_inference"
                    prediction = self.predict(token, seeds)
                    stage = "save_prediction"
                    poses = self.save(token, prediction)
                    record.update(native_pose_count=len(poses), metric_pose_count=len(
                        metric_poses(poses, self.interval_length_s)), metric_horizon_s=METRIC_HORIZON_S)
                    record.update(self.save_extras(token, prediction))
                    successful.append(token)
                    records.append(record)
                    print(f"Completed {index}/{len(tokens)} token={token}", flush=True)
                except Exception as exc:
                    failures.append(failure(index, token, stage, exc, **seeds))
                    save_json(self.result_dir / "failed_samples.json", failures)
                    print(f"Failed {index}/{len(tokens)} token={token} stage={stage}: {type(exc).__name__}: {exc}",
                          file=sys.stderr, flush=True)
            return self.finish(tokens, records, successful, failures, contracts, signature, excluded)

    def finish(self, tokens, records, successful, failures, contracts, signature, excluded) -> int:
        """Write the bookkeeping and the manifest, then score an edited run; 0 if it all worked."""
        from ..scoring.records import run_summary
        save_json(self.result_dir / "failed_samples.json", failures)
        save_json(self.result_dir / "metrics" / "per_sample.json", records)
        summary = run_summary([] if self.original else records)
        summary.update({
            "model": self.model, "dataset": self.dataset, "mode": "original" if self.original else "paired",
            "run_dir": str(self.variant_dir), "run_config_path": str(self.loader.run_config_path),
            "annotations_path": str(self.loader.annotations_path), "result_dir": str(self.result_dir),
            "original_result_dir": str(self.original_dir), "evaluation_config_path": str(self.config_path),
            "interval_length_s": self.interval_length_s, "metric_horizon_s": METRIC_HORIZON_S,
            "identity": {"signature": signature, **self.identity()},
            "selected_token_count": len(tokens), "successful_inference_count": len(successful),
            "successful_pair_count": 0 if self.original else len(records),
            "failed_sample_count": len({row["token"] for row in failures}),
            "excluded_original_tokens": excluded, "excluded_input_tokens": self.excluded_inputs,
            "gpus": gpu_names(), **self.summary_extras()})
        save_json(self.result_dir / "metrics" / "summary.json", summary)
        save_manifest(self.result_dir, mode=summary["mode"], signature=signature, source_run_dir=self.variant_dir,
                      selected_tokens=tokens, successful_tokens=successful, source_contracts=contracts)
        print(f"Saved {len(successful)}/{len(tokens)} predictions to {self.result_dir}; "
              f"{len(failures)} failures.", flush=True)
        scored = True
        if not self.original and self.args.scoring == "score" and successful:
            scored = score(self.result_dir, self.dataset, self.args.scoring_workers)
        return 0 if successful and scored else 1

    # --- command line -------------------------------------------------------------------------------------------

    @classmethod
    def parser(cls, description: str, dataset: str | None = None) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description=description, allow_abbrev=False,
                                         formatter_class=argparse.RawDescriptionHelpFormatter)
        parser.add_argument("variant_dir", type=Path,
                            help="Benchmark variant (final_images/, metadata/, run_config.json)")
        names = sorted(path.stem for path in Path(cls.config_dir).glob("*.yaml"))
        parser.add_argument("--config", help=f"Configuration: {', '.join(names)} (in {Path(cls.config_dir).name}/) "
                                             "or a path")
        parser.add_argument("--dataset", choices=cls.datasets, default=dataset or cls.datasets[0],
                            help=f"Dataset of VARIANT_DIR (default {dataset or cls.datasets[0]})")
        parser.add_argument("--original", action="store_true",
                            help="Predict the original scenes (cached in <dataset>/original for the edited runs)")
        selection = parser.add_mutually_exclusive_group()
        selection.add_argument("--max-samples", type=int, help="Only the first N tokens (a smoke test)")
        selection.add_argument("--tokens-file", type=Path, help="JSON with the exact ordered 'tokens' to evaluate")
        parser.add_argument("--original-result-dir", type=Path,
                            help="Original predictions of an edited run (default <dataset>/original)")
        parser.add_argument("--scoring", choices=("score", "off"), default="score",
                            help="Score the edited run on CPU after inference (default) or not")
        parser.add_argument("--scoring-workers", type=int, default=1, help="ReactNAV scoring processes (default 1)")
        parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory")
        return parser

    @classmethod
    def main(cls, description: str, argv=None, *, dataset: str | None = None) -> int:
        parser = cls.parser(description, dataset)
        args = parser.parse_args(argv)
        if args.max_samples is not None and args.max_samples <= 0:
            parser.error("--max-samples must be positive")
        if args.scoring_workers < 1:
            parser.error("--scoring-workers must be positive")
        if args.original and args.original_result_dir is not None:
            parser.error("--original-result-dir is for edited runs")
        return cls(args).run()

