"""Shared cached inference and clean/edited scoring for stage-3 validation.

Scoring subprocesses only receive CPU trajectories and immutable input references.
The initialized training agent and standalone agent use the same single-rollout path.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from contextlib import closing, contextmanager
from dataclasses import asdict
import gzip
import hashlib
import logging
import lzma
import multiprocessing
import os
from pathlib import Path
import pickle
import random
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

from .recogdrive_checkpoint_selection import aggregate, aggregate_contract, digest, read_json
from .recogdrive_paired_validation import PAIRED_PROTOCOL, is_paired, source_shard, aggregate_pairs, evaluate_pairs

TRAIN_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = TRAIN_ROOT.parents[1]
LOGGER = logging.getLogger(__name__)

# Bump the affected version (and config.yaml) when evaluation semantics change.
# Training losses/regularizers and source refactors do not change these versions.
PROTOCOLS = {"clean": "recogdrive-clean-pdms-v1", "edited": "recogdrive-edited-nc-dac-v1",
             "features": "recogdrive-internvl-features-v1", "world": "recogdrive-raw-world-v1"}
ARTIFACT_VERSION = 2
ONLINE_FEATURE_PROTOCOL = "recogdrive-internvl-online-inputs-v1"


def pin_imports():
    import navsim
    if Path(navsim.__file__).resolve().parent != TRAIN_ROOT / "navsim":
        raise RuntimeError("Validation must import navsim from train/recogdrive")
    for path in (REPO_ROOT, TRAIN_ROOT):
        if str(path) in sys.path:
            sys.path.remove(str(path))
        sys.path.insert(0, str(path))


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def fingerprint(path, *, content=False):
    path = Path(path).resolve()
    stat = path.stat()
    result = {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if content:
        result["sha256"] = file_sha256(path)
    return result


def live_checkpoint_identity(agent, epoch, step):
    """Identify evaluated planner weights before reward-based retention runs."""
    planner = getattr(agent, "action_head", agent)
    hasher, size = hashlib.sha256(), 0
    for name, tensor in sorted(planner.state_dict().items()):
        if name.startswith("old_policy."):
            continue
        hasher.update(digest({"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}).encode())
        raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        hasher.update(memoryview(raw))
        size += raw.nbytes
    return {"kind": "live_planner_state", "sha256": hasher.hexdigest(), "tensor_bytes": size,
            "epoch": epoch, "step": step}


def verify_file(reference):
    actual = fingerprint(reference["path"], content="sha256" in reference)
    if actual != reference:
        raise ValueError(f"Frozen input changed: {reference['path']}")
    return Path(reference["path"])


def load_config(path, *, training_batch_size=None, batch_size=None, edited_protocol=None):
    from omegaconf import OmegaConf
    config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    from .recogdrive_triplet_validation import is_triplet, validate_config
    if is_triplet(config):
        if edited_protocol not in (None, "triplet"):
            raise ValueError("Triplet config cannot use paired/legacy reporting")
        return validate_config(config, batch_size=batch_size)
    if edited_protocol == "triplet":
        raise ValueError("Use validation/config_edited_only_braking.yaml for triplet reporting")
    if config["version"] != ARTIFACT_VERSION or config["clean_samples"] != 18179:
        raise ValueError("Validation requires schema v2 and all 18,179 clean samples")
    if edited_protocol is not None:
        if edited_protocol not in ("paired", "legacy"):
            raise ValueError("edited_protocol must be paired or legacy")
        config["protocols"]["edited"] = PAIRED_PROTOCOL if edited_protocol == "paired" else PROTOCOLS["edited"]
    expected = {**PROTOCOLS, "edited": PAIRED_PROTOCOL} if is_paired(config) else PROTOCOLS
    if config.get("feature_mode", "cached_hidden_states") == "online":
        expected = {**expected, "features": ONLINE_FEATURE_PROTOCOL}
    elif config.get("feature_mode", "cached_hidden_states") != "cached_hidden_states":
        raise ValueError("Unknown feature_mode")
    if config.get("protocols") != expected:
        raise ValueError("Validation protocol versions do not match this implementation")
    if config["tolerance"] != 3.0:
        raise ValueError("Validation tolerance must be 3 absolute PDMS points")
    if batch_size is not None:
        config["batch_size"] = batch_size
    elif config["batch_size"] is None:
        config["batch_size"] = (training_batch_size if training_batch_size is not None
                                else int(os.environ.get("RECOGDRIVE_BATCH_SIZE", "4")))
    config.setdefault("feature_workers", 0)
    config.setdefault("prefetch_factor", 2)
    if is_paired(config):
        config.setdefault("paired_feature_length", 2800)
        if type(config["paired_feature_length"]) is not int or config["paired_feature_length"] < 1:
            raise ValueError("paired_feature_length must be a positive integer")
    for key in ("batch_size", "scoring_workers", "prefetch_factor"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"Validation {key} must be a positive integer")
    if type(config["feature_workers"]) is not int or config["feature_workers"] < 0:
        raise ValueError("Validation feature_workers must be a non-negative integer")
    if config["inference"]["precision"] not in ("32-true", "16-mixed", "bf16-mixed"):
        raise ValueError("Unsupported validation inference precision")
    if not 0 <= config["inference"]["seed"] < 2**63:
        raise ValueError("Inference seed must be in [0, 2**63)")
    return config


def code_identity():
    files = [
        *sorted((TRAIN_ROOT / "navsim/agents/recogdrive").rglob("*.py")),
        *sorted((TRAIN_ROOT / "navsim/planning/simulation/planner/pdm_planner").rglob("*.py")),
        TRAIN_ROOT / "navsim/evaluate/pdm_score.py", Path(__file__),
        Path(__file__).with_name("recogdrive_paired_validation.py"),
        Path(__file__).with_name("recogdrive_checkpoint_selection.py"),
        Path(__file__).parent / "callbacks/recogdrive_validation_callback.py",
        TRAIN_ROOT / "validation/prepare_data.py", TRAIN_ROOT / "validation/evaluate_checkpoints.py",
        REPO_ROOT / "shared/evaluation/navsim_raw_metrics.py",
        REPO_ROOT / "shared/evaluation/navsim_raw_world.py",
        REPO_ROOT / "shared/evaluation/paiav_collision_metrics.py",
        REPO_ROOT / "autovla/evaluation/trajectory_metrics.py",
        REPO_ROOT / "shared/geometry/pedestrian_geometry.py",
    ]
    return digest({str(path.relative_to(REPO_ROOT)): file_sha256(path) for path in files})


def feature_code_identity():
    paths = [TRAIN_ROOT / "navsim/agents/recogdrive" / name for name in (
        "recogdrive_features.py", "recogdrive_backbone.py", "utils/internvl_preprocess.py", "utils/conversation.py")]
    return digest({str(p.relative_to(TRAIN_ROOT)): file_sha256(p) for p in paths})


def world_code_identity():
    paths = [REPO_ROOT / "shared/evaluation/navsim_raw_world.py",
             TRAIN_ROOT / "navsim/common/dataclasses.py",
             TRAIN_ROOT / "navsim/planning/scenario_builder/navsim_scenario.py"]
    return digest({str(p.relative_to(REPO_ROOT)): file_sha256(p) for p in paths})


def provenance():
    """Audit metadata, deliberately excluded from compatibility decisions."""
    return {"code": code_identity(), "feature_code": feature_code_identity(),
            "world_code": world_code_identity(), "torch": torch.__version__, "numpy": np.__version__}


def require_artifact_version(value):
    if value.get("version") != ARTIFACT_VERSION:
        raise ValueError("Legacy/unsupported validation artifact: schema v2 with separate clean/edited "
                         "contracts is required. Preserve old artifacts; use a new validation root. "
                         "Old whole-code hashes cannot establish semantic compatibility automatically.")


def cohort_identity(manifest, kind):
    """Ordered declared inputs, independent of the other cohort and audit metadata."""
    fields = ("sample_id", "token", "kind", "scene_token", "log_name", "source", "frame_index",
              "feature_path", "variant_id", "image", "original_image", "metadata")
    samples = [{**{key: s[key] for key in fields if key in s},
                "source_failed": bool(s.get("source_error"))}
               for s in manifest["samples"] if s["kind"] == kind]
    return digest({"version": ARTIFACT_VERSION, "samples": samples,
                   "features": manifest["feature_contract"],
                   **({"metric_cache": manifest["metric_cache"]} if kind == "clean" else {}),
                   **({"world": manifest["world_contract"]} if kind == "edited" else {})})


def input_contract(config, prepared, kind):
    fields = ("sample_id", "token", "kind", "features", "metric_cache", "world", "actor_motion", "variant_id")
    inputs = [{**{key: s[key] for key in fields if key in s},
               "preparation_failed": bool(s.get("preparation_error"))}
              for s in prepared["samples"] if s["kind"] == kind]
    contract = {"version": ARTIFACT_VERSION, "protocol": config["protocols"][kind],
            "cohort": prepared["cohort_identities"][kind], "inputs": digest(inputs),
            "inference": config["inference"],
            "runtime": {"torch": torch.__version__, "numpy": np.__version__},
            "batching": {"batch_size": config["batch_size"],
                         "protocol": "fixed per-kind manifest batches; per-batch seed v1; right zero padding"},
            "model": {key: config["model"][key] for key in (
                "cam_type", "vlm_type", "dit_type", "vlm_size", "sampling_method")},
            "aggregation": ("successful sample mean; clean PDMS 5/5/2; scale 100" if kind == "clean"
                            else "successful sample mean; raw binary NC-all x DAC; scale 100")}
    if kind == "edited" and is_paired(config):
        tokens = {s["token"] for s in prepared["samples"] if s["kind"] == "edited"}
        originals = [{k: s[k] for k in ("sample_id", "token", "features") if k in s}
                     for s in prepared["samples"] if s["kind"] == "clean" and s["token"] in tokens]
        contract["original_inputs"] = digest(originals)
        contract["batching"] = {"protocol": "source-token CPU float32 noise; fixed right zero padding",
                                "feature_length": config.get("paired_feature_length", 2800)}
        contract["aggregation"] += "; paired complete-case gain/rescue/harm; trajectory diagnostics"
    if config.get("feature_mode") == "online":
        contract["model"] = {**config["model"], "conditioning": config.get("conditioning_contract")}
        contract["batching"] = {"batch_size": config["batch_size"],
                                "protocol": "online VLM left padding to at least 2800, no truncation",
                                "noise": "source-token CPU float32" if kind == "edited" and is_paired(config)
                                         else "fixed per-kind manifest batches; per-batch seed v1",
                                "pooling": "unmasked_mean"}
    return contract


def input_identity(config, prepared, kind=None):
    if kind is not None:
        return digest(input_contract(config, prepared, kind))
    return digest({k: input_identity(config, prepared, k) for k in prepared["cohort_identities"]})


def validate_result(result, evaluator, checkpoint, *, clean_only=False, edited_only=False):
    """Reuse scores only for the requested cohorts, protocol and checkpoint bytes."""
    require_artifact_version(result)
    from .recogdrive_triplet_validation import is_triplet
    if is_triplet(evaluator.config):
        return evaluator.validate_result(result, checkpoint)
    if not result.get("finalized"):
        raise ValueError("Cannot reuse unfinished validation results")
    for kind in evaluation_kinds(clean_only=clean_only, edited_only=edited_only):
        contract = result.get("evaluation_contracts", {}).get(kind)
        if (contract != evaluator.contracts[kind]
                or result.get("input_identities", {}).get(kind) != digest(contract)):
            raise ValueError(f"Incompatible {kind} evaluation inputs/protocol; measure this cohort again")
        expected = [s["sample_id"] for s in evaluator.samples if s["kind"] == kind]
        measured = aggregate([r for r in result["records"] if r["kind"] == kind], expected)
        if aggregate_contract(result[kind]) != aggregate_contract(measured):
            raise ValueError(f"Saved {kind} aggregate does not match its sample records")
        if kind == "edited" and is_paired(evaluator.config):
            if result.get("paired") != aggregate_pairs(result["records"], expected):
                raise ValueError("Saved paired aggregate does not match its sample records")
    # A copied checkpoint with identical bytes is still the same model.
    saved = result.get("checkpoint_identity", {})
    if not saved.get("sha256") or any(saved.get(k) != checkpoint[k] for k in ("sha256", "size")):
        raise ValueError("Baseline/checkpoint identity mismatch")


def seed_for_batch(sample_ids, seed):
    """One reproducible random stream per declared batch, independent of epoch/rank."""
    material = {"protocol": "recogdrive-validation-batch-v1", "seed": seed, "samples": list(sample_ids)}
    return int(digest(material)[:16], 16) >> 1


@contextmanager
def preserve_inference_state(agent):
    modes = [(module, module.training) for module in agent.modules()]
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices), torch.inference_mode():
        try:
            agent.eval()
            yield
        finally:
            # Restore each child exactly, including independently frozen references.
            for module, training in modes:
                module.training = training
            random.setstate(python_state)
            np.random.set_state(numpy_state)


def load_features(reference):
    with gzip.open(verify_file(reference), "rb") as handle:
        features = pickle.load(handle)
    required = {"history_trajectory": (4, 3), "high_command_one_hot": (4,),
                "status_feature": (8,)}
    for key, shape in required.items():
        if tuple(features[key].shape) != shape or not torch.isfinite(features[key]).all():
            raise ValueError(f"Invalid cached {key}")
    if "image_path_tensor" in features:
        path = features["image_path_tensor"]
        if path.ndim != 1 or path.dtype != torch.long or path.numel() == 0:
            raise ValueError("Invalid online image path tensor")
        image = verify_file(features["image_reference"])
        if Path(''.join(chr(int(c)) for c in path)).resolve() != image:
            raise ValueError("Online image path differs from frozen image reference")
        return {key: features[key] for key in (*required, "image_path_tensor")}
    hidden = features["last_hidden_state"]
    if hidden.ndim != 2 or hidden.shape[0] < 1 or hidden.shape[1] != 1536 or not torch.isfinite(hidden).all():
        raise ValueError("Invalid 2B cached VLM hidden state")
    result = {key: features[key] for key in (*required, "last_hidden_state")}
    if "context_valid_mask" in features:
        result["context_valid_mask"] = features["context_valid_mask"]
    return result


class _FeatureBatchDataset(Dataset):
    """Load declared batches without replacing failed samples or changing seeds.

    ``load_images`` (online inputs) also preprocesses each image here, in the loader workers, exactly
    as the agent's forward would; image failures then count as that sample's input error.
    """
    def __init__(self, batches, load_images=False):
        self.batches, self.load_images = batches, load_images

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):
        start = time.perf_counter()
        batch = self.batches[index]
        ready, features, errors = [], [], []
        for sample in batch:
            try:
                if sample.get("preparation_error"):
                    raise ValueError(sample["preparation_error"])
                loaded = load_features(sample["features"])
                if self.load_images and "image_path_tensor" in loaded:
                    from navsim.agents.recogdrive.utils import internvl_preprocess
                    loaded["pixel_values"] = internvl_preprocess.load_image(
                        "".join(map(chr, loaded["image_path_tensor"].tolist())))
                features.append(loaded)
                ready.append(sample)
            except Exception as exc:
                errors.append((sample, f"input: {type(exc).__name__}: {exc}"))
        return {"batch": batch, "ready": ready, "features": features, "errors": errors,
                "loading_seconds": time.perf_counter() - start}


def _identity_batch(batch):
    return batch


def iter_feature_batches(batches, config):
    """Bounded, ordered CPU prefetch; worker seeds never consume training RNG."""
    if not batches:
        return
    workers = min(config.get("feature_workers", 0), len(batches))
    options = ({"multiprocessing_context": "spawn",
                "prefetch_factor": config.get("prefetch_factor", 2)} if workers else {})
    loader = DataLoader(_FeatureBatchDataset(batches, load_images=config.get("worker_image_loading", False)),
                        batch_size=None, shuffle=False,
                        collate_fn=_identity_batch, num_workers=workers,
                        generator=torch.Generator().manual_seed(config["inference"]["seed"]),
                        **options)
    yield from loader


def evaluation_kinds(*, clean_only=False, edited_only=False):
    if clean_only and edited_only:
        raise ValueError("clean_only and edited_only are mutually exclusive")
    return ("clean",) if clean_only else ("edited",) if edited_only else ("clean", "edited")


def infer_batch(agent, features, seed, inference, *, source_seeds=None, fixed_length=None):
    """Use the agent's ordinary batched forward with the training collation rules."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed % 2**32)
    device = next(agent.parameters()).device
    precision = inference["precision"]
    dtype = torch.float16 if precision == "16-mixed" else torch.bfloat16
    from navsim.agents.recogdrive.recogdrive_conditioning import collate_features
    batch = {key: value.to(device) if key != "image_path_tensor" else value
             for key, value in collate_features(features).items()}
    if fixed_length is not None and "last_hidden_state" in batch:
        hidden = batch["last_hidden_state"]
        if hidden.shape[1] > fixed_length:
            raise ValueError("Cached features exceed paired protocol sequence length")
        batch["last_hidden_state"] = torch.nn.functional.pad(hidden, (0, 0, 0, fixed_length - hidden.shape[1]))
        if "context_valid_mask" in batch:
            batch["context_valid_mask"] = torch.nn.functional.pad(
                batch["context_valid_mask"], (0, fixed_length - hidden.shape[1]), value=False)
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=precision != "32-true"):
        prediction = agent.forward(batch, **({"inference_seeds": source_seeds} if source_seeds is not None else {}))
    poses = prediction["pred_traj"].float().cpu().numpy()
    if poses.shape != (len(features), 8, 3):
        raise ValueError(f"Inference must return one (8, 3) trajectory per scene, got {poses.shape}")
    # Check finiteness per sample in the evaluator so a bad row does not discard
    # valid predictions from the rest of its batch.
    return poses


def score_sample(sample, poses):
    """Absolute scoring; edited predictions do not require original predictions."""
    pin_imports()
    if sample["kind"] == "clean":
        from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
        from navsim.common.dataclasses import Trajectory
        from navsim.evaluate.pdm_score import pdm_score
        from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
        from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
        with lzma.open(verify_file(sample["metric_cache"]), "rb") as handle:
            cache = pickle.load(handle)
        sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
        result = asdict(pdm_score(cache, Trajectory(poses), sampling,
                                 PDMSimulator(sampling), PDMScorer(sampling, PDMScorerConfig(
                                     progress_weight=5., ttc_weight=5., comfortable_weight=2.,
                                     driving_direction_weight=0.))))
        result = {key: float(value) for key, value in result.items()}
        return 100. * result["score"], result
    from reactbench.scoring.navsim_metrics import score_prediction_any_contact as score_prediction
    world = read_json(verify_file(sample["world"]))
    result = score_prediction(poses, .5, sample["actor_motion"], world)
    if result["nc_all"] not in (0, 1) or result["dac"] not in (0, 1):
        return None, result
    return 100. * result["nc_all"] * result["dac"], result


def _score_record(sample, poses):
    start = time.perf_counter()
    record = {"sample_id": sample["sample_id"], "token": sample["token"], "kind": sample["kind"]}
    try:
        score, components = score_sample(sample, poses)
        record["components"] = components
        if score is None:
            raise ValueError(f"Edited NC-all/DAC unavailable: {components.get('dac_reason', components)}")
        if not np.isfinite(score) or not 0 <= score <= 100:
            raise ValueError("Invalid primary score")
        record.update(score=score, components=components, error=None)
    except Exception as exc:
        record.update(score=None, error=f"scoring: {type(exc).__name__}: {exc}")
    record["scoring_seconds"] = time.perf_counter() - start
    return record


def _worker_init():
    torch.set_num_threads(1)
    pin_imports()


def shard_batches(samples, batch_size, rank, world_size):
    """Freeze batches before distributing them; never pad shards or drop a tail."""
    if not 0 <= rank < world_size:
        raise ValueError("Invalid validation rank/world size")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Validation batch_size must be a positive integer")
    for kind in ("clean", "edited"):
        group = [sample for sample in samples if sample["kind"] == kind]
        for start in range(rank * batch_size, len(group), world_size * batch_size):
            yield group[start:start + batch_size]


def shard_samples(samples, rank, world_size, batch_size=1):
    return [sample for batch in shard_batches(samples, batch_size, rank, world_size) for sample in batch]


def combine_shards(shards, samples, kinds=("clean", "edited")):
    records = [record for shard in shards for record in shard["records"]]
    groups = {kind: aggregate([r for r in records if r["kind"] == kind],
                             [s["sample_id"] for s in samples if s["kind"] == kind])
              for kind in kinds}
    return {"finalized": True, **groups, "records": sorted(records, key=lambda r: r["sample_id"]),
            "rank_timings": [shard["timings"] for shard in shards]}


class ValidationEvaluator:
    def __init__(self, config, *, clean_only=False, edited_only=False):
        pin_imports()
        kinds = evaluation_kinds(clean_only=clean_only, edited_only=edited_only)
        self.config = config
        self.prepared = read_json(config["clean_prepared_inputs" if clean_only else "prepared_inputs"])
        manifest = read_json(config["clean_manifest" if clean_only else "manifest"])
        require_artifact_version(self.prepared)
        require_artifact_version(manifest)
        cohorts = {kind: cohort_identity(manifest, kind) for kind in kinds}
        if {k: self.prepared.get("cohort_identities", {}).get(k) for k in kinds} != cohorts:
            raise ValueError("Prepared inputs do not match the frozen cohorts")
        if manifest["feature_contract"]["protocol"] != config["protocols"]["features"]:
            raise ValueError("Cached VLM preprocessing/prompt protocol changed; prepare new inputs")
        if not clean_only and manifest["world_contract"]["protocol"] != config["protocols"]["world"]:
            raise ValueError("Raw-world construction protocol changed; prepare new inputs")
        self.samples = [s for s in self.prepared["samples"] if s["kind"] in kinds]
        members = lambda rows: [(s["sample_id"], s["kind"], s["token"]) for s in rows]
        self.originals = {}
        if "edited" in kinds and is_paired(config):
            if self.prepared.get("cohort_identities", {}).get("clean") != cohort_identity(manifest, "clean"):
                raise ValueError("Paired original cohort differs from the frozen clean cohort")
            if members([s for s in self.prepared["samples"] if s["kind"] == "clean"]) != members(
                    [s for s in manifest["samples"] if s["kind"] == "clean"]):
                raise ValueError("Preparation silently changed the paired original cohort")
            for sample in self.prepared["samples"]:
                if sample["kind"] == "clean":
                    if sample["token"] in self.originals:
                        raise ValueError("Duplicate original token in paired inputs")
                    # Clean PDMS preparation failures need not prevent feature-only inference.
                    self.originals[sample["token"]] = {k: sample[k] for k in
                                                       ("sample_id", "token", "kind", "features") if k in sample}
        if members(self.samples) != members([s for s in manifest["samples"] if s["kind"] in kinds]):
            raise ValueError("Preparation silently changed the declared cohort")
        if len({s["sample_id"] for s in self.samples}) != len(self.samples):
            raise ValueError("Duplicate frozen validation sample IDs")
        if any(s["kind"] not in ("clean", "edited") for s in self.prepared["samples"]):
            raise ValueError("Unknown validation sample kind")
        if "clean" in kinds and sum(s["kind"] == "clean" for s in self.samples) != config["clean_samples"]:
            raise ValueError("Clean cohort must contain all 18,179 held-out samples")
        if clean_only and any(s["kind"] != "clean" for s in self.prepared["samples"]):
            raise ValueError("Clean-only preparation must contain only clean samples")
        if not clean_only and not any(s["kind"] == "edited" for s in self.samples):
            raise ValueError("Edited cohort is empty; finalize and prepare it first")
        self.contracts = {kind: input_contract(config, self.prepared, kind) for kind in kinds}
        self.identities = {kind: digest(contract) for kind, contract in self.contracts.items()}
        self.identity = digest(self.identities)
        self.provenance = provenance()

    def artifact_context(self, *, clean_only=False, edited_only=False):
        kinds = evaluation_kinds(clean_only=clean_only, edited_only=edited_only)
        contracts = {k: v for k, v in self.contracts.items() if k in kinds}
        identities = {k: digest(v) for k, v in contracts.items()}
        return {"version": ARTIFACT_VERSION, "evaluation_contracts": contracts,
                "input_identities": identities, "input_identity": digest(identities),
                "provenance": self.provenance, "manifest_identity": self.prepared["manifest_identity"]}

    def check_agent_config(self, agent_config):
        for key, value in self.config["model"].items():
            actual = agent_config.get(key)
            if key == "vlm_path" and actual and value:  # equivalent spellings, e.g. the launcher's ../..
                actual, value = Path(actual).resolve(), Path(value).resolve()
            if actual != value:
                raise ValueError(f"Validation model configuration mismatch: {key}")
        online = self.config.get("feature_mode") == "online"
        if agent_config.get("train_backbone") or bool(agent_config.get("cache_hidden_state")) == online:
            raise ValueError("Validation requires a frozen VLM and the configured online/cached feature mode")
        if online and not agent_config.get("mask_vlm_padding"):
            raise ValueError("Online stage-2 validation requires VLM padding masks")
        sampling = agent_config["trajectory_sampling"]
        if sampling.get("time_horizon") != 4 or sampling.get("interval_length") != .5:
            raise ValueError("Validation requires native eight-pose/four-second trajectories")

    def evaluate_shard(self, agent, rank=0, world_size=1, clean_only=False, edited_only=False):
        kinds = evaluation_kinds(clean_only=clean_only, edited_only=edited_only)
        samples = [s for s in self.samples if s["kind"] in kinds]
        batches = list(shard_batches(samples, self.config["batch_size"], rank, world_size))
        sample_count = sum(len(batch) for batch in batches)
        records, timings = [], {}
        workers = self.config["scoring_workers"]
        pool = ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"),
                                   initializer=_worker_init) if workers > 1 else None
        try:
            with preserve_inference_state(agent):
                for kind in kinds:
                    if kind == "edited" and is_paired(self.config):
                        paired_records, timings[kind] = evaluate_pairs(self, agent, rank, world_size, pool)
                        records.extend(paired_records)
                        continue
                    start, pending = time.perf_counter(), []
                    inference_seconds = 0.
                    input_wait_seconds, feature_loading_seconds, forward_seconds = 0., 0., 0.

                    def collect():
                        future, sample, context = pending.pop(0)
                        try:
                            record = future.result()  # never finalize still-running scores
                        except Exception as exc:
                            record = {"sample_id": sample["sample_id"], "token": sample["token"],
                                      "kind": kind, "score": None,
                                      "error": f"worker: {type(exc).__name__}: {exc}"}
                        record.update(context)
                        records.append(record)

                    def fail(sample, error, context):
                        records.append({"sample_id": sample["sample_id"], "token": sample["token"],
                                        "kind": kind, "score": None, "error": error, **context})

                    kind_batches = [b for b in batches if b[0]["kind"] == kind]
                    with closing(iter_feature_batches(kind_batches, self.config)) as loaded_batches:
                        while True:
                            before = time.perf_counter()
                            try:
                                loaded = next(loaded_batches)
                            except StopIteration:
                                break
                            wait = time.perf_counter() - before
                            input_wait_seconds += wait
                            feature_loading_seconds += loaded["loading_seconds"]
                            batch, ready, features = loaded["batch"], loaded["ready"], loaded["features"]
                            seed = seed_for_batch([s["sample_id"] for s in batch], self.config["inference"]["seed"])
                            for sample, error in loaded["errors"]:
                                fail(sample, error, {"batch_seed": seed})
                            inference_seconds += wait
                            if not ready:
                                continue
                            forward_start = time.perf_counter()
                            try:
                                poses_batch = infer_batch(agent, features, seed, self.config["inference"])
                            except Exception as exc:
                                for sample in ready:
                                    fail(sample, f"batch inference: {type(exc).__name__}: {exc}", {"batch_seed": seed})
                                continue
                            finally:
                                elapsed = time.perf_counter() - forward_start
                                forward_seconds += elapsed
                                inference_seconds += elapsed
                            duration = time.perf_counter() - before
                            context = {"batch_seed": seed, "inference_batch_size": len(ready),
                                       "inference_seconds": duration / len(ready)}  # amortized input/forward time
                            for sample, poses in zip(ready, poses_batch):
                                if not np.isfinite(poses).all():
                                    fail(sample, "inference: non-finite trajectory", context)
                                    continue
                                try:
                                    if pool:
                                        pending.append((pool.submit(_score_record, sample, poses), sample, context))
                                        if len(pending) >= workers * 2:
                                            collect()
                                    else:
                                        record = _score_record(sample, poses)
                                        record.update(context)
                                        records.append(record)
                                except Exception as exc:
                                    fail(sample, f"scoring worker: {type(exc).__name__}: {exc}", context)
                            if len(records) and len(records) % 100 == 0:
                                LOGGER.info("Validation rank %s: %s/%s attempts completed", rank, len(records), sample_count)
                    while pending:
                        collect()
                    timings[kind] = {"wall_seconds": time.perf_counter() - start,
                                     "inference_seconds": inference_seconds,
                                     "input_wait_seconds": input_wait_seconds,
                                     "feature_loading_seconds": feature_loading_seconds,
                                     "forward_seconds": forward_seconds}
        finally:
            if pool:
                pool.shutdown(wait=True)
        return {"records": records, "timings": timings}

    def finalize(self, shards, clean_only=False, edited_only=False):
        kinds = evaluation_kinds(clean_only=clean_only, edited_only=edited_only)
        samples = [s for s in self.samples if s["kind"] in kinds]
        result = combine_shards(shards, samples, kinds=kinds if edited_only else ("clean", "edited"))
        result.update(self.artifact_context(clean_only=clean_only, edited_only=edited_only))
        if "edited" in kinds and is_paired(self.config):
            result["paired"] = aggregate_pairs(result["records"], [s["sample_id"] for s in samples if s["kind"] == "edited"])
        return result


def load_baseline(config, evaluator):
    result = read_json(config["baseline_results"])
    checkpoint = fingerprint(config["baseline_checkpoint"], content=True)
    validate_result(result, evaluator, checkpoint, clean_only=True)
    if result["clean"]["mean"] is None:
        raise ValueError("Baseline clean PDMS is undefined")
    identity = digest({"inputs": evaluator.identities,
                       "baseline": {"clean": aggregate_contract(result["clean"]),
                                    "checkpoint_sha256": checkpoint["sha256"]},
                       "tolerance": config["tolerance"], "keep": 5,
                       "tie_breaks": ["edited descending", "clean descending", "epoch ascending"]})
    return result, identity


def evaluate_distributed(evaluator, agent, rank=0, world_size=1, clean_only=False, edited_only=False):
    """One completion/error protocol for standalone evaluation and the callback."""
    kinds = evaluation_kinds(clean_only=clean_only, edited_only=edited_only)
    options = {"clean_only": True} if clean_only else {"edited_only": True} if edited_only else {}
    try:
        local = evaluator.evaluate_shard(agent, rank, world_size, **options)
    except Exception as exc:
        from .recogdrive_triplet_validation import is_triplet
        if is_triplet(evaluator.config):
            local = evaluator.failed_shard(rank, world_size, f"rank failure: {type(exc).__name__}: {exc}")
            if world_size > 1:
                shards = [None] * world_size
                dist.all_gather_object(shards, local)
            else:
                shards = [local]
            return evaluator.finalize(shards, **options)
        samples = [s for s in evaluator.samples if s["kind"] in kinds]
        assigned = shard_samples(samples, rank, world_size, evaluator.config["batch_size"])
        if "edited" in kinds and is_paired(evaluator.config):
            assigned = [s for s in assigned if s["kind"] == "clean"] + source_shard(samples, rank, world_size)
        local = {"records": [{"sample_id": s["sample_id"], "token": s["token"], "kind": s["kind"],
                              "score": None, "error": f"rank failure: {type(exc).__name__}: {exc}"}
                             for s in assigned],
                 "timings": {}}
    if world_size > 1:
        shards = [None] * world_size
        dist.all_gather_object(shards, local)
    else:
        shards = [local]
    return evaluator.finalize(shards, **options)


def load_inference_checkpoint(agent, path):
    """Reject partial planner loads instead of silently retaining random inference weights."""
    from navsim.agents.recogdrive.recogdrive_diffusion_planner import planner_state_from_checkpoint
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    agent.action_head.load_state_dict(planner_state_from_checkpoint(agent.action_head, checkpoint), strict=False)
    return checkpoint
