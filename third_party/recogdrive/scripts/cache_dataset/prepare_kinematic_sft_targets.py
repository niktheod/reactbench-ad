#!/usr/bin/env python
"""Freeze kinematic stop targets for ReactTrain fine-tuning.

For every hazardous edit (t1-t4: the complete mixed manifest's t3/t4 edits plus the t1/t2 cache
root) the target keeps the logged GT path and brakes at a constant deceleration from t = 0 to
stop the margin short of the object's footprint (capped at the maximum deceleration), then
holds. CPU only; caches are read and never modified.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path

from edited_training_cache import EXP, atomic_write, read_json
from navsim.planning.training import recogdrive_kinematic_sft as kin
from navsim.planning.training.recogdrive_mixed_data import KINEMATIC_SFT_TIMESTEPS, digest, edit_key, load_manifest
from prepare_mixed_training_dataset import kinematic_edits

DEFAULT_MIXED = EXP / "recogdrive_mixed_training/navtrain_selected_t34_v1/manifest.json"
DEFAULT_EXTRA = EXP / "recogdrive_edited_training/navtrain_selected_t12_v1"
DEFAULT_OUTPUT = EXP / "recogdrive_kinematic_sft/targets_v1/targets.json"


def _worker_init():
    import torch
    torch.set_num_threads(1)


def edit_record(item):
    """(token, record): the kinematic target of a hazardous edit."""
    token, sample, settings = item
    try:
        gt_file, metric_file = kin.sample_file(sample, "trajectory_target.gz"), kin.sample_file(sample, "metric")
        gt, cache = kin.load_gt(gt_file), kin.load_metric_cache(metric_file)
        common = {"source_token": sample["source_token"], "variant": edit_key(token, sample, KINEMATIC_SFT_TIMESTEPS)}
        # The augmented metric caches track the inserted object under this token.
        target, record = kin.kinematic_target(cache, gt, f"inserted_{token}", kin.KinematicSettings(**settings))
        return token, {**common, "target": target.tolist(), "files": {"gt": gt_file, "metric": metric_file},
                       **record}
    except Exception as exc:
        raise RuntimeError(f"{token}: {type(exc).__name__}: {exc}") from exc


def _records(items, workers, label):
    if workers <= 1:
        return dict(edit_record(item) for item in items)
    results = []
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"), initializer=_worker_init) as pool:
        for index, result in enumerate(pool.map(edit_record, items, chunksize=8), 1):
            results.append(result)
            if index % 1000 == 0:
                print(f"{label}: {index}/{len(items)}", flush=True)
    return dict(results)


def build(mixed_path=DEFAULT_MIXED, extra_root=DEFAULT_EXTRA, workers=1, limit=None, settings=None):
    """The artifact for every hazardous edit (or the first ``limit`` of them for a dry run)."""
    settings = kin.KinematicSettings() if settings is None else settings
    mixed_path = Path(mixed_path).expanduser().resolve()
    extra_root = Path(extra_root).expanduser().resolve()
    mixed = load_manifest(mixed_path)
    edits, _, extra_identity = kinematic_edits(mixed, extra_root)
    hazardous = sorted(t for t, s in edits.items() if s["kind"] == "hazardous")
    items = [(t, edits[t], settings.as_dict()) for t in hazardous[:limit]]
    samples = _records(items, workers, "Kinematic targets")
    value = {"protocol": kin.TARGETS_PROTOCOL, "settings": settings.as_dict(),
             "conventions": kin.target_conventions(settings),
             "mixed_manifest": mixed["artifact_sha256"], "mixed_manifest_path": str(mixed_path),
             "extra_edited_root": str(extra_root), "extra_edited_manifest": extra_identity,
             "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "complete": limit is None or len(items) == len(hazardous),
             "summary": kin.target_summary(samples), "samples": samples}
    return {**value, "artifact_sha256": digest(value)}


def publish(value, output):
    """Write once; an existing artifact must be identical (re-running is read-only)."""
    output = Path(output).expanduser().resolve()
    if not value["complete"]:
        raise ValueError("Refusing to publish a partial target artifact; use --dry-run with --limit")
    if output.exists():
        if read_json(output).get("artifact_sha256") != value["artifact_sha256"]:
            raise FileExistsError(f"{output} exists with different targets; use a new --output")
        return output
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    atomic_write(output, lambda handle: handle.write(payload))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mixed-manifest", type=Path, default=DEFAULT_MIXED, help="Complete mixed manifest (t3/t4 edits)")
    parser.add_argument("--extra-edited-root", type=Path, default=DEFAULT_EXTRA, help="Prepared t1/t2 edited cache root")
    parser.add_argument("--output", type=Path, help=f"Defaults to {DEFAULT_OUTPUT} (default settings only)")
    parser.add_argument("--margin", type=float, default=kin.KinematicSettings.margin_m, help="Stop distance before the object (m)")
    parser.add_argument("--max-deceleration", type=float, default=kin.KinematicSettings.max_deceleration_mps2,
                        help="Deceleration cap (m/s^2)")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    parser.add_argument("--limit", type=int, help="First N hazardous edits only; requires --dry-run")
    parser.add_argument("--dry-run", action="store_true", help="Compute and summarize without writing")
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--workers and --limit must be positive")
    if args.limit is not None and not args.dry_run:
        parser.error("--limit is for dry runs; the published artifact covers every hazardous edit")
    settings = kin.KinematicSettings(margin_m=args.margin, max_deceleration_mps2=args.max_deceleration)
    if args.output is None and settings != kin.KinematicSettings():
        parser.error("Non-default --margin/--max-deceleration require their own --output")
    output = (args.output or DEFAULT_OUTPUT).expanduser().resolve()
    value = build(args.mixed_manifest, args.extra_edited_root, args.workers, args.limit, settings)
    if not args.dry_run:
        publish(value, output)
    print(json.dumps({"output": None if args.dry_run else str(output), "artifact_sha256": value["artifact_sha256"],
                      "samples": len(value["samples"]), "settings": value["settings"], "summary": value["summary"]}, indent=2))


if __name__ == "__main__":
    main()
