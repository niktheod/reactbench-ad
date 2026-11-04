"""Paired reaction reporting, independent of training dropout and reward selection."""
from contextlib import closing
import time

import numpy as np

from .recogdrive_checkpoint_selection import aggregate, digest

PAIRED_PROTOCOL = "recogdrive-edited-paired-nc-dac-v1"


def is_paired(config):
    return config.get("protocols", {}).get("edited") == PAIRED_PROTOCOL


def source_seed(token, seed):
    return int(digest({"protocol": PAIRED_PROTOCOL, "source": token, "seed": seed})[:16], 16) >> 1


def source_shard(samples, rank, world_size):
    if not 0 <= rank < world_size:
        raise ValueError("Invalid validation rank/world size")
    tokens = sorted({s["token"] for s in samples if s["kind"] == "edited"})[rank::world_size]
    tokens = set(tokens)
    return [s for s in samples if s["kind"] == "edited" and s["token"] in tokens]


def trajectory_diagnostics(poses):
    xy = np.concatenate([np.zeros((1, 2)), np.asarray(poses)[:, :2]], axis=0)
    length = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
    return {"path_length_m": length, "speed_proxy_mps": length / 4.,
            "longitudinal_displacement_m": float(xy[-1, 0])}


def score_pair(sample, edited, original, edited_error=None, original_error=None):
    from .recogdrive_validation import _score_record
    record = (_score_record(sample, edited) if edited_error is None else
              {"sample_id": sample["sample_id"], "token": sample["token"], "kind": "edited",
               "score": None, "error": edited_error})
    reference = (_score_record(sample, original) if original_error is None else
                 {"score": None, "error": original_error})
    record.update(variant_id=sample.get("variant_id"), original_score=reference["score"],
                  original_components=reference.get("components"), original_error=reference["error"],
                  original_scoring_seconds=reference.get("scoring_seconds", 0.))
    record["pair_error"] = record["error"] or reference["error"]
    record["reaction_gain_pp"] = (record["score"] - reference["score"]
                                  if record["pair_error"] is None else None)
    if edited_error is None and original_error is None:
        a, b = trajectory_diagnostics(original), trajectory_diagnostics(edited)
        record["trajectory_diagnostics"] = {"original": a, "edited": b,
                                             "change": {k: b[k] - a[k] for k in a}}
    return record


def aggregate_pairs(records, expected):
    """Signed gains are separate from the existing [0,100] safety aggregator."""
    rows = [r for r in records if r["kind"] == "edited"]
    for row in rows:
        gain = row.get("reaction_gain_pp")
        if gain is not None and (row.get("score") not in (0, 100) or
                                 row.get("original_score") not in (0, 100) or
                                 row.get("error") or row.get("original_error") or row.get("pair_error") or
                                 gain != row["score"] - row["original_score"]):
            raise ValueError("Paired gain must equal the difference of two valid binary safety scores")
    coverage = aggregate([{"sample_id": r["sample_id"],
                           "score": r.get("score") if r.get("reaction_gain_pp") is not None else None,
                           "error": r.get("pair_error") or r.get("error") or
                           (None if r.get("reaction_gain_pp") is not None else "incomplete pair")}
                          for r in rows], expected)
    valid = [r for r in rows if r.get("reaction_gain_pp") is not None and not r.get("pair_error")]
    count = len(valid)
    rescued = sum(r["original_score"] == 0 and r["score"] == 100 for r in valid)
    harmed = sum(r["original_score"] == 100 and r["score"] == 0 for r in valid)
    failures = sum(r["original_score"] == 0 for r in valid)
    mean = lambda key: float(np.mean([r[key] for r in valid])) if count else None
    result = {**coverage, "complete": count, "original_mean": mean("original_score"),
              "edited_mean": mean("score"), "reaction_gain_pp": mean("reaction_gain_pp"),
              "rescued": rescued, "harmed": harmed, "original_failures": failures,
              "rescue_fraction": rescued / count if count else None,
              "harm_fraction": harmed / count if count else None,
              "rescue_rate": rescued / failures if failures else None}
    result.pop("mean")
    result.pop("sum")
    diagnostic_rows = [r["trajectory_diagnostics"] for r in rows if "trajectory_diagnostics" in r]
    result["diagnostics_complete"] = len(diagnostic_rows)
    result["trajectory_diagnostics"] = {
        side: {key: float(np.mean([r[side][key] for r in diagnostic_rows])) if diagnostic_rows else None
               for key in ("path_length_m", "speed_proxy_mps", "longitudinal_displacement_m")}
        for side in ("original", "edited", "change")}
    return result


def evaluate_pairs(evaluator, agent, rank, world_size, pool):
    from .recogdrive_validation import iter_feature_batches, infer_batch
    config = evaluator.config
    samples = source_shard(evaluator.samples, rank, world_size)
    tokens = sorted({s["token"] for s in samples})
    originals = [evaluator.originals.get(token, {"token": token, "sample_id": f"clean/{token}",
                  "kind": "clean", "preparation_error": "missing original feature reference"}) for token in tokens]
    predictions, records, pending = {}, [], []
    start = time.perf_counter()
    timing = {k: 0. for k in ("input_wait_seconds", "feature_loading_seconds", "forward_seconds",
                               "original_forward_seconds", "edited_forward_seconds")}
    workers = config["scoring_workers"]

    def collect():
        future, sample, context = pending.pop(0)
        try:
            record = future.result()
        except Exception as exc:
            error = f"scoring worker: {type(exc).__name__}: {exc}"
            record = {"sample_id": sample["sample_id"], "token": sample["token"], "kind": "edited",
                      "score": None, "error": error, "pair_error": error, "reaction_gain_pp": None}
        records.append({**record, **context})

    for side, cohort in (("original", originals), ("edited", samples)):
        batches = [cohort[i:i + config["batch_size"]] for i in range(0, len(cohort), config["batch_size"])]
        with closing(iter_feature_batches(batches, config)) as loaded_batches:
            while True:
                before = time.perf_counter()
                try:
                    loaded = next(loaded_batches)
                except StopIteration:
                    break
                timing["input_wait_seconds"] += time.perf_counter() - before
                timing["feature_loading_seconds"] += loaded["loading_seconds"]
                ready, features = loaded["ready"], loaded["features"]
                outcomes = [(s, None, e) for s, e in loaded["errors"]]
                if ready:
                    before = time.perf_counter()
                    try:
                        seeds = [source_seed(s["token"], config["inference"]["seed"]) for s in ready]
                        poses = infer_batch(agent, features, config["inference"]["seed"], config["inference"],
                                            source_seeds=seeds,
                                            fixed_length=config.get("paired_feature_length", 2800))
                        outcomes.extend((s, p, None if np.isfinite(p).all() else "non-finite trajectory")
                                        for s, p in zip(ready, poses))
                    except Exception as exc:
                        outcomes.extend((s, None, f"inference: {type(exc).__name__}: {exc}") for s in ready)
                    elapsed = time.perf_counter() - before
                    timing["forward_seconds"] += elapsed
                    timing[f"{side}_forward_seconds"] += elapsed
                for sample, poses, error in outcomes:
                    if side == "original":
                        predictions[sample["token"]] = (poses, error)
                        continue
                    original, original_error = predictions[sample["token"]]
                    args = (sample, poses, original, error, original_error)
                    context = {"source_seed": source_seed(sample["token"], config["inference"]["seed"])}
                    if pool:
                        pending.append((pool.submit(score_pair, *args), sample, context))
                        if len(pending) >= 2 * workers:
                            collect()
                    else:
                        records.append({**score_pair(*args), **context})
    while pending:
        collect()
    timing.update(wall_seconds=time.perf_counter() - start,
                  inference_seconds=timing["input_wait_seconds"] + timing["forward_seconds"],
                  original_predictions=len(originals), edited_predictions=len(samples))
    return records, timing
