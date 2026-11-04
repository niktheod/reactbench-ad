"""Opt-in held-out original/benign/hazardous evaluation of native model poses.

This reporter never calls the PDM simulator/controller and defines no reward.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
from functools import lru_cache
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np

from .recogdrive_mixed_data import HAZARD_SFT_PROTOCOL, HOLDOUT_PROTOCOLS, check_file, load_manifest
from .recogdrive_paired_validation import source_seed, trajectory_diagnostics
from .recogdrive_checkpoint_selection import aggregate, digest

TRIPLET_PROTOCOL = "recogdrive-edited-triplet-nc-dac-v1"


def is_triplet(config):
    return config.get("protocols", {}).get("edited") == TRIPLET_PROTOCOL


def validate_config(config, batch_size=None):
    if config.get("version") != 2 or not is_triplet(config):
        raise ValueError("Triplet validation requires schema v2 and its dedicated protocol")
    if config.get("feature_mode", "cached_hidden_states") != "cached_hidden_states":
        raise ValueError("Triplet validation uses existing cached hidden states")
    if batch_size is not None:
        config["batch_size"] = batch_size
    for key in ("batch_size", "scoring_workers", "prefetch_factor", "paired_feature_length"):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError(f"Triplet {key} must be a positive integer")
    if type(config.get("feature_workers")) is not int or config["feature_workers"] < 0:
        raise ValueError("Triplet feature_workers must be a nonnegative integer")
    if (type(config["inference"]["seed"]) is not int or not 0 <= config["inference"]["seed"] < 2**63
            or config["inference"]["precision"] not in ("32-true", "16-mixed", "bf16-mixed")):
        raise ValueError("Invalid triplet inference seed/precision")
    for key in ("training_manifest", "raw_world_manifest"):
        if not config.get(key):
            raise ValueError(f"Triplet validation requires {key}")
    if type(config.get("hazard_sft_metrics", False)) is not bool:
        raise ValueError("hazard_sft_metrics must be a boolean")
    return config


@lru_cache(maxsize=128)
def _world(path, sha256, size):
    import hashlib
    data = Path(path).read_bytes()
    if len(data) != size or hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError(f"Frozen raw world changed: {path}")
    return json.loads(data)


def predicted_braking(poses, initial_speed):
    """Finite differences at native .5 s resolution; no execution assumptions."""
    nodes = np.vstack((np.zeros((1, 2)), np.asarray(poses)[:, :2]))
    speeds = np.linalg.norm(np.diff(nodes, axis=0), axis=1) / .5
    decel = np.diff(np.r_[initial_speed, speeds]) / .5
    braking = np.flatnonzero(decel <= -1.)
    stopped = np.flatnonzero(speeds <= .5)
    # Earliest low-speed segment after which all remaining segments stay low.
    stop = next((int(i) for i in stopped if np.all(speeds[i:] <= .5)), None)
    return {**trajectory_diagnostics(poses), "early_speed_mps": float(speeds[:3].mean()),
            "final_speed_mps": float(speeds[-1]), "peak_predicted_deceleration_mps2": float(max(0., -decel.min())),
            "braking_onset_s": float(braking[0] * .5) if len(braking) else None,
            "stopping_time_s": float((stop + 1) * .5) if stop is not None else None,
            "emergency_braking": bool(np.any(decel <= -3.))}


def score_triplet(sample, predictions):
    from reactbench.scoring.actor_motion import actor_footprint_corners
    from reactbench.scoring.geometry import footprint_signed_clearance, vehicle_footprint_corners
    from reactbench.scoring.navsim_metrics import score_prediction_any_contact as score_prediction
    record = {k: sample[k] for k in ("sample_id", "token", "kind", "variant", "log_name")}
    record.update(score=None, error=None, attempted=True, outcomes={}, diagnostics={}, input_errors={}, scoring_errors={})
    for side, (poses, error) in predictions.items():
        if error:
            record["input_errors"][side] = error
    try:
        world = _world(**sample["world"])
        initial_speed = float(np.linalg.norm(world["initial_velocity"]))
    except Exception as exc:
        record["error"] = f"world: {type(exc).__name__}: {exc}"
        return record
    for side, (poses, error) in predictions.items():
        if error:
            continue
        try:
            poses = np.asarray(poses, dtype=float)
            if poses.shape != (8, 3) or not np.isfinite(poses).all():
                raise ValueError("invalid native predicted trajectory")
            diagnostics = predicted_braking(poses, initial_speed)
            stop = diagnostics["stopping_time_s"]
            if stop is not None:
                pose = poses[int(round(stop / .5)) - 1]
                diagnostics["stopping_clearance_m"] = float(footprint_signed_clearance(
                    vehicle_footprint_corners(pose, world["geometry"]),
                    actor_footprint_corners(sample["hazardous_motion"], stop)))
            else:
                diagnostics["stopping_clearance_m"] = None
            record["diagnostics"][side] = diagnostics
            record["outcomes"][f"{side}_hazardous"] = score_prediction(poses, .5, sample["hazardous_motion"], world)
            if side == "benign":
                record["outcomes"]["benign_own"] = score_prediction(poses, .5, sample["benign_motion"], world)
            elif side == "original":
                record["outcomes"]["original_own"] = score_prediction(poses, .5, None, world)
        except Exception as exc:
            record["scoring_errors"][side] = f"{type(exc).__name__}: {exc}"
    hazardous = record["outcomes"].get("hazardous_hazardous", {})
    if hazardous.get("nc_dac") in (0, 1):
        record["score"] = 100. * hazardous["nc_dac"]
    errors = [*record["input_errors"].values(), *record["scoring_errors"].values()]
    if record["score"] is None:
        errors.append("hazardous raw safety unavailable")
    record["error"] = "; ".join(errors) or None
    # aggregate() treats error as a failure. Independent component coverage is
    # kept below even when another side or DAC failed.
    return record


def _summary(records):
    def values(key, component="nc_dac"):
        return [r["outcomes"].get(key, {}).get(component) for r in records
                if r.get("outcomes", {}).get(key, {}).get(component) in (0, 1)]
    components = {key: {c: {"mean": 100. * float(np.mean(v)) if v else None, "complete": len(v),
                           "expected": len(records)}
                        for c in ("nc_all", "dac", "nc_dac") for v in [values(key, c)]}
                  for key in ("benign_hazardous", "hazardous_hazardous", "original_hazardous", "benign_own", "original_own")}
    pairs = [r for r in records if all(r.get("outcomes", {}).get(f"{s}_hazardous", {}).get("nc_dac") in (0, 1)
                                      for s in ("benign", "hazardous"))]
    b = [r["outcomes"]["benign_hazardous"]["nc_dac"] for r in pairs]
    h = [r["outcomes"]["hazardous_hazardous"]["nc_dac"] for r in pairs]
    rescued = sum(x == 0 and y == 1 for x, y in zip(b, h))
    harmed = sum(x == 1 and y == 0 for x, y in zip(b, h))
    failures = b.count(0)
    original_pairs = [r for r in records if all(r.get("outcomes", {}).get(f"{s}_hazardous", {}).get("nc_dac") in (0, 1)
                                              for s in ("original", "hazardous"))]
    complete = sum(all(r.get("outcomes", {}).get(f"{s}_hazardous", {}).get("nc_dac") in (0, 1)
                       for s in ("original", "benign", "hazardous")) for r in records)
    diagnostic_pairs = [r for r in records if all(s in r.get("diagnostics", {}) for s in ("original", "benign", "hazardous"))]
    diagnostics = {}
    for side in ("original", "benign", "hazardous"):
        keys = ("path_length_m", "speed_proxy_mps", "longitudinal_displacement_m", "early_speed_mps", "final_speed_mps",
                "peak_predicted_deceleration_mps2", "braking_onset_s", "stopping_time_s", "stopping_clearance_m", "emergency_braking")
        diagnostics[side] = {}
        for key in keys:
            vals = [r["diagnostics"][side][key] for r in diagnostic_pairs if r["diagnostics"][side].get(key) is not None]
            diagnostics[side][key] = {"mean": float(np.mean(vals)) if vals else None, "complete": len(vals)}
    # First-contact speed availability has its own denominator (clear trajectories
    # and contacts at t=0 have no measurable incoming contact speed).
    contact = {}
    for side in ("original", "benign", "hazardous"):
        vals = [(r.get("outcomes", {}).get(f"{side}_hazardous", {}).get("first_contact") or {}).get("ego_speed_mps")
                for r in records]
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        contact[side] = {"mean_mps": float(np.mean(vals)) if vals else None, "complete": len(vals)}
    benign_change = {key: {"mean": float(np.mean([r["diagnostics"]["benign"][key] - r["diagnostics"]["original"][key]
                                                   for r in diagnostic_pairs])) if diagnostic_pairs else None,
                           "complete": len(diagnostic_pairs)}
                    for key in ("path_length_m", "speed_proxy_mps", "longitudinal_displacement_m", "early_speed_mps")}
    return {"expected": len(records), "attempted": sum(r.get("attempted", True) for r in records), "complete": int(complete),
            "failed": len(records) - int(complete), "complete_pairs": len(pairs),
            "benign_mean": 100. * float(np.mean(b)) if b else None,
            "hazardous_mean": 100. * float(np.mean(h)) if h else None,
            "hazard_gain_pp": 100. * float(np.mean(np.subtract(h, b))) if b else None,
            "rescued": rescued, "harmed": harmed, "rescue_fraction": rescued / len(pairs) if pairs else None,
            "harm_fraction": harmed / len(pairs) if pairs else None, "benign_failures": failures,
            "rescue_rate": rescued / failures if failures else None, "components": components,
            "original_pairs": len(original_pairs), "original_gain_pp": float(np.mean([
                100. * (r["outcomes"]["hazardous_hazardous"]["nc_dac"] - r["outcomes"]["original_hazardous"]["nc_dac"])
                for r in original_pairs])) if original_pairs else None,
            "diagnostics_complete": len(diagnostic_pairs), "predicted_trajectories": diagnostics,
            "benign_minus_original": benign_change,
            "raw_contact_speed": contact,
            "early_speed_reduction_mps": float(np.mean([r["diagnostics"]["benign"]["early_speed_mps"] -
                r["diagnostics"]["hazardous"]["early_speed_mps"] for r in diagnostic_pairs])) if diagnostic_pairs else None,
            "benign_emergency_only": sum(r["diagnostics"]["benign"]["emergency_braking"] and
                                         not r["diagnostics"]["original"]["emergency_braking"] for r in diagnostic_pairs)}


def aggregate_triplets(records, samples):
    expected = {s["sample_id"]: s for s in samples}
    ids = [r["sample_id"] for r in records]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("Triplet coverage contains missing, duplicate or unexpected records")
    for record in records:
        if any(record.get(k) != expected[record["sample_id"]][k] for k in ("token", "variant", "log_name", "kind")):
            raise ValueError("Triplet result membership mismatch")
        for outcome in record.get("outcomes", {}).values():
            nc, dac = outcome.get("nc_all"), outcome.get("dac")
            if nc not in (0, 1) or dac not in (None, 0, 1):
                raise ValueError("Triplet safety components must be binary or unavailable")
            product = nc * dac if dac is not None else None
            if outcome.get("nc_dac") != product:
                raise ValueError("Triplet safety product differs from NC-all x DAC")
        safety = record.get("outcomes", {}).get("hazardous_hazardous", {}).get("nc_dac")
        if record.get("score") != (100. * safety if safety is not None else None):
            raise ValueError("Triplet hazardous score differs from its raw components")
    summary = _summary(records)
    summary["by_variant"] = {variant: _summary([r for r in records if r["variant"] == variant])
                             for variant in sorted({r["variant"] for r in records})}
    # Resample whole logs, retaining their sources and variants as clusters.
    groups = [[r for r in records if r["log_name"] == log] for log in sorted({r["log_name"] for r in records})]
    summary["source_count"] = len({r["token"] for r in records})
    summary["log_count"] = len(groups)
    gains = [np.array([100. * (r["outcomes"]["hazardous_hazardous"]["nc_dac"] - r["outcomes"]["benign_hazardous"]["nc_dac"])
                       for r in group if all(r.get("outcomes", {}).get(f"{side}_hazardous", {}).get("nc_dac") in (0, 1)
                                            for side in ("benign", "hazardous"))]) for group in groups]
    gains = [g for g in gains if len(g)]
    interval = None
    if len(gains) >= 2:
        rng = np.random.default_rng(0)
        boot = [float(np.concatenate([gains[i] for i in rng.integers(len(gains), size=len(gains))]).mean()) for _ in range(1000)]
        interval = np.percentile(boot, [2.5, 97.5]).tolist()
    summary["hazard_gain_log_cluster_ci95_pp"] = interval
    return summary


class TripletValidationEvaluator:
    def __init__(self, config, *, clean_only=False, edited_only=False):
        import torch
        raise NotImplementedError("Triplet monitoring (held-out hazard SFT sources) is not part of this release")
        if clean_only:
            raise ValueError("Triplet reporting has no clean PDMS scope; clean L1 remains in training")
        self.config = validate_config(config)
        self.manifest = load_manifest(config["training_manifest"])
        if self.manifest["protocol"] not in HOLDOUT_PROTOCOLS:
            raise ValueError("Triplet reporting requires an edited-only or hazard SFT training manifest")
        worlds = RawSafetyInputs(config["raw_world_manifest"])
        self.samples, self.inputs = [], {}
        for triplet in self.manifest["triplets"]:
            token = triplet["source_token"]
            source = self.manifest["original_sources"][token]
            world = worlds.manifest["samples"].get(token)
            if world is None or Path(world["feature_path"]).resolve() != Path(source["feature_path"]):
                raise ValueError(f"Missing/mismatched cached original world: {token}")
            if world["metric_file"] != next(r for r in source["files"] if r["path"] == source["metric_path"]):
                raise ValueError(f"Raw/PDM source identity mismatch: {token}")
            record = {"sample_id": f"triplet/{token}/{triplet['variant']}", "token": token, "kind": "edited",
                      "variant": triplet["variant"], "log_name": source["log_name"],
                      "world": {"path": str((worlds.path.parent / world["world_file"]).resolve()),
                                "sha256": world["world_sha256"], "size": world["world_size"]}}
            for side, sample in (("original", source), ("benign", self.manifest["heldout_samples"][triplet["benign"]]),
                                 ("hazardous", self.manifest["heldout_samples"][triplet["hazardous"]])):
                sample_id = token if side == "original" else triplet[side]
                reference = next(r for r in sample["files"] if Path(r["path"]).name == "internvl_feature.gz")
                # Keep frozen preparation stamps; load_features resolves named links.
                check_file(reference)
                frozen = {**reference, "path": str(Path(reference["path"]).resolve())}
                self.inputs[sample_id] = {"sample_id": sample_id, "token": token, "kind": "edited", "features": frozen}
                record[side] = sample_id
                if side != "original":
                    record[f"{side}_motion"] = sample["actor_motion"]
            self.samples.append(record)
        self.contracts = {"edited": {"version": 2, "protocol": TRIPLET_PROTOCOL,
            "dataset": self.manifest["artifact_sha256"], "raw_world_manifest": fingerprint(worlds.path, content=True),
            "samples": digest(self.samples), "inputs": digest(self.inputs), "inference": config["inference"],
            "runtime": {"torch": torch.__version__, "numpy": np.__version__},
            "feature_length": config["paired_feature_length"], "model": config["model"],
            "braking": "native-0.5s; onset=-1m/s2; emergency=-3m/s2; stop=0.5m/s-v1"}}
        # Opt-in hazard SFT monitoring: L1 to each input's training target and V0 in its own world.
        self.sft_inputs = None
        if config.get("hazard_sft_metrics", False):
            from .recogdrive_hazard_sft import load_targets, monitoring_contract, monitoring_inputs
            if self.manifest["protocol"] != HAZARD_SFT_PROTOCOL:
                raise ValueError("Hazard SFT metrics require a hazard SFT training manifest")
            self.sft_inputs = monitoring_inputs(self.manifest, load_targets(self.manifest))
            if set(self.sft_inputs) != set(self.inputs):
                raise ValueError("Hazard SFT monitoring inputs differ from the triplet inputs")
            self.contracts["edited"]["hazard_sft"] = monitoring_contract(self.manifest, self.sft_inputs)
        self.identities = {k: digest(v) for k, v in self.contracts.items()}
        self.identity = digest(self.identities)
        self.provenance = {**provenance(), "triplet_code_sha256": fingerprint(Path(__file__), content=True)["sha256"]}

    def check_agent_config(self, agent_config):
        from .recogdrive_validation import ValidationEvaluator
        return ValidationEvaluator.check_agent_config(self, agent_config)

    def artifact_context(self, **options):
        return {"version": 2, "evaluation_contracts": self.contracts, "input_identities": self.identities,
                "input_identity": self.identity, "provenance": self.provenance}

    def assigned(self, rank, world_size):
        if type(rank) is not int or type(world_size) is not int or not 0 <= rank < world_size:
            raise ValueError("Invalid triplet rank/world size")
        sources = sorted({s["token"] for s in self.samples})[rank::world_size]
        return [s for s in self.samples if s["token"] in sources]

    def failed_shard(self, rank, world_size, error):
        samples = self.assigned(rank, world_size)
        shard = {"records": [{**{k: s[k] for k in ("sample_id", "token", "kind", "variant", "log_name")},
                              "score": None, "error": error, "attempted": False, "outcomes": {}} for s in samples], "timings": {}}
        if self.sft_inputs is not None:
            from .recogdrive_hazard_sft import failed_monitoring_record
            inputs = sorted({s[side] for s in samples for side in ("original", "benign", "hazardous")})
            shard["sft_records"] = [failed_monitoring_record(self.sft_inputs[i], error) for i in inputs]
        return shard

    def _score_sft_inputs(self, inputs, predictions, pool, workers):
        """Per-input hazard SFT monitoring, bounded like the triplet scoring queue."""
        from .recogdrive_hazard_sft import failed_monitoring_record, score_monitoring_input
        records, pending = [], []

        def collect():
            future, spec = pending.pop(0)
            try:
                return future.result()
            except Exception as exc:
                return failed_monitoring_record(spec, f"scoring: {type(exc).__name__}: {exc}")

        for item in inputs:
            spec = self.sft_inputs[item["sample_id"]]
            poses, error = predictions.get(item["sample_id"], (None, "missing prediction"))
            if error:
                records.append(failed_monitoring_record(spec, error))
            elif pool:
                pending.append((pool.submit(score_monitoring_input, spec, poses), spec))
                if len(pending) >= 2 * workers:
                    records.append(collect())
            else:
                records.append(score_monitoring_input(spec, poses))
        while pending:
            records.append(collect())
        return records

    def evaluate_shard(self, agent, rank=0, world_size=1, **options):
        from .recogdrive_validation import iter_feature_batches, infer_batch, preserve_inference_state, _worker_init
        samples = self.assigned(rank, world_size)
        inputs = [self.inputs[k] for k in sorted({s[side] for s in samples for side in ("original", "benign", "hazardous")})]
        predictions, records, pending = {}, [], []
        timing = {k: 0. for k in ("forward_seconds", "feature_loading_seconds", "input_wait_seconds")}
        start = time.perf_counter()
        batches = [inputs[i:i + self.config["batch_size"]] for i in range(0, len(inputs), self.config["batch_size"])]
        workers = self.config["scoring_workers"]
        pool = ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"), initializer=_worker_init) if workers > 1 and samples else None
        try:
            with preserve_inference_state(agent), closing(iter_feature_batches(batches, self.config)) as loaded_batches:
                while True:
                    before = time.perf_counter()
                    try:
                        batch = next(loaded_batches)
                    except StopIteration:
                        break
                    timing["input_wait_seconds"] += time.perf_counter() - before
                    timing["feature_loading_seconds"] += batch["loading_seconds"]
                    for sample, error in batch["errors"]:
                        predictions[sample["sample_id"]] = (None, error)
                    ready = batch["ready"]
                    if ready:
                        before = time.perf_counter()
                        try:
                            poses = infer_batch(agent, batch["features"], self.config["inference"]["seed"], self.config["inference"],
                                                source_seeds=[source_seed(s["token"], self.config["inference"]["seed"]) for s in ready],
                                                fixed_length=self.config["paired_feature_length"])
                            for sample, pose in zip(ready, poses):
                                predictions[sample["sample_id"]] = (pose, None)
                        except Exception as exc:
                            for sample in ready:
                                predictions[sample["sample_id"]] = (None, f"inference: {type(exc).__name__}: {exc}")
                        timing["forward_seconds"] += time.perf_counter() - before
            def collect(future, sample):
                try:
                    record = future.result()
                except Exception as exc:
                    record = {**{k: sample[k] for k in ("sample_id", "token", "kind", "variant", "log_name")},
                              "score": None, "error": f"scoring: {type(exc).__name__}: {exc}", "outcomes": {}}
                return record
            for sample in samples:
                args = (sample, {side: predictions.get(sample[side], (None, "missing prediction")) for side in ("original", "benign", "hazardous")})
                if pool:
                    pending.append((pool.submit(score_triplet, *args), sample))
                    if len(pending) >= 2 * workers:
                        records.append(collect(*pending.pop(0)))
                else:
                    records.append(score_triplet(*args))
            while pending:
                records.append(collect(*pending.pop(0)))
            for record in records:
                record["source_seed"] = source_seed(record["token"], self.config["inference"]["seed"])
            if self.sft_inputs is not None:
                before = time.perf_counter()
                sft_records = self._score_sft_inputs(inputs, predictions, pool, workers)
                timing["hazard_sft_scoring_seconds"] = time.perf_counter() - before
        finally:
            if pool:
                pool.shutdown(wait=True)
        timing.update(wall_seconds=time.perf_counter() - start, predictions=len(inputs), triplets=len(samples))
        shard = {"records": records, "timings": timing}
        if self.sft_inputs is not None:
            shard["sft_records"] = sft_records
        return shard

    def finalize(self, shards, **options):
        records = sorted([r for shard in shards for r in shard["records"]], key=lambda r: r["sample_id"])
        triplets = aggregate_triplets(records, self.samples)
        # Hazardous absolute safety has independent coverage from triplet failures.
        edited = aggregate([{**r, "error": None if r.get("score") is not None else r.get("error")} for r in records],
                           [s["sample_id"] for s in self.samples])
        result = {"finalized": True, "edited": edited, "triplets": triplets, "records": records,
                  "rank_timings": [s["timings"] for s in shards], **self.artifact_context()}
        if self.sft_inputs is not None:
            from .recogdrive_hazard_sft import aggregate_monitoring
            sft_records = sorted([r for shard in shards for r in shard.get("sft_records", [])],
                                 key=lambda r: r["sample_id"])
            result.update(hazard_sft=aggregate_monitoring(sft_records, self.sft_inputs), hazard_sft_records=sft_records)
        return result

    def validate_result(self, result, checkpoint):
        if (not result.get("finalized") or result.get("evaluation_contracts") != self.contracts
                or result.get("input_identity") != self.identity or result.get("input_identities") != self.identities):
            raise ValueError("Incompatible triplet evaluation inputs/protocol")
        measured = self.finalize([{"records": result["records"], "sft_records": result.get("hazard_sft_records", []),
                                   "timings": {}}])
        if result.get("triplets") != measured["triplets"] or result.get("edited") != measured["edited"]:
            raise ValueError("Saved triplet aggregates do not match records")
        if self.sft_inputs is not None and result.get("hazard_sft") != measured["hazard_sft"]:
            raise ValueError("Saved hazard SFT aggregates do not match records")
        if any(result.get("checkpoint_identity", {}).get(k) != checkpoint[k] for k in ("sha256", "size")):
            raise ValueError("Triplet checkpoint identity mismatch")
