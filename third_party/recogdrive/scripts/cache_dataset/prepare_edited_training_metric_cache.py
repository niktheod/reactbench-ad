#!/usr/bin/env python
"""Build PDM reference/observation caches in each edited training world (CPU only)."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import lzma
import multiprocessing
import os
from pathlib import Path
import pickle

from edited_training_cache import (
    TRAIN_ROOT, argument_parser, atomic_write, code_identity, completed_record,
    digest, discover, ensure_contract, finish, freeze_manifest, locked, record_result,
    navsim_default, release_paths, scene_from_source, verify_file, verify_sample,
)


def metric_contract(args):
    from navsim.planning.metric_caching.edited_scenario import backend_provenance, map_identity, numerical_preflight
    numerical_preflight()
    return {"protocol": "recogdrive-edited-training-metrics-v1",
            "maps": map_identity(args.maps_root),
            "backend": backend_provenance(TRAIN_ROOT, args.map_version),
            "code": code_identity([Path(__file__), Path(__file__).with_name("edited_training_cache.py")])}


@lru_cache(maxsize=1)
def base_scenario(source_json, maps_root, map_version):
    import json
    from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
    source = json.loads(source_json)
    scene = scene_from_source(source, future_frames=10)
    return NavSimScenario(scene, map_root=maps_root, map_version=map_version)


def validate_metric_cache(cache, scenario, track_token):
    """Check the inserted footprint throughout the complete five-second TTC world."""
    import numpy as np
    for index in range(51):
        actor = scenario.actor_at(index * .1, cache.ego_state.time_point.time_us + index * 100000)
        if track_token not in cache.observation[index].tokens:
            raise ValueError(f"Inserted actor missing from metric observation {index}")
        if cache.observation[index][track_token].hausdorff_distance(actor.box.geometry) > 1e-6:
            raise ValueError(f"Inserted actor geometry/motion differs at observation {index}")
    if not np.isfinite(cache.ego_state.rear_axle.array).all():
        raise ValueError("Invalid metric-cache ego state")


def build_sample_metric(sample, source, path, maps_root, map_version):
    import json
    from navsim.planning.metric_caching.edited_scenario import AugmentedScenario, build_metric_cache, reference_diagnostics

    base = base_scenario(json.dumps(source, sort_keys=True), str(maps_root), map_version)
    track_token = f"inserted_{sample['sample_id']}"
    scenario = AugmentedScenario(base, sample["actor_motion"], track_token)
    cache = build_metric_cache(scenario, path)
    validate_metric_cache(cache, scenario, track_token)
    diagnostics = reference_diagnostics(cache)
    def write(handle):
        with lzma.LZMAFile(handle, "wb", preset=0) as compressed:
            pickle.dump(cache, compressed, protocol=pickle.HIGHEST_PROTOCOL)
    atomic_write(path, write)
    with lzma.open(path, "rb") as handle:
        restored = pickle.load(handle)
    validate_metric_cache(restored, scenario, track_token)
    return diagnostics


def prepare_one(task):
    sample, source, output, identity, maps_root, map_version = task
    output = Path(output)
    try:
        verify_sample(sample, source)
        record = completed_record(output, "metrics", sample, identity)
        if record is not None:
            return record
        path = output / sample["metric_cache"]
        diagnostics = build_sample_metric(sample, source, path, maps_root, map_version)
        verify_sample(sample, source)
        return record_result(output, "metrics", sample, identity, [path], reference=diagnostics)
    except Exception as exc:
        print(f"FAILED {sample['sample_id']}: {exc}", flush=True)
        return record_result(output, "metrics", sample, identity, error=exc)


def publish_metric_index(output, manifest, records):
    if len(records) != len(manifest["samples"]) or any(r.get("error") for r in records):
        raise ValueError("Cannot publish an incomplete metric-cache index")
    if [r["sample_id"] for r in records] != [s["sample_id"] for s in manifest["samples"]]:
        raise ValueError("Metric-cache records do not match manifest sample identities")
    if any(p.name != "metric_cache_metadata.csv" for p in (output / "metrics/metadata").glob("*.csv")):
        raise ValueError("Unexpected additional metric-cache metadata CSV")
    paths = [str(output / sample["metric_cache"]) for sample in manifest["samples"]]
    if len({Path(p).parent.name for p in paths}) != len(paths):
        raise ValueError("Metric-cache sample IDs are not unique")
    for path, record in zip(paths, records):
        if len(record["files"]) != 1 or record["files"][0]["path"] != path:
            raise ValueError("Metric-cache record points to a different sample")
        verify_file(record["files"][0])
    payload = ("file_name\n" + "\n".join(paths) + "\n").encode()
    atomic_write(output / "metrics/metadata/metric_cache_metadata.csv", lambda handle: handle.write(payload))


def prepare_metrics(args, manifest, contract):
    with locked(args.output / ".metrics.lock"):
        contract_id = ensure_contract(args.output, "metrics", contract)
        # Publish only after the whole cohort has valid caches. Failed retries
        # must not leave a previously published index pointing at partial data.
        (args.output / "metrics/metadata/metric_cache_metadata.csv").unlink(missing_ok=True)
        tasks = [(sample, manifest["sources"][sample["source_token"]], str(args.output),
                  digest([manifest["identity"], contract_id, sample["sample_id"]]),
                  str(args.maps_root), args.map_version) for sample in manifest["samples"]]
        executor = (ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"))
                    if args.workers > 1 else None)
        records = []
        try:
            results = executor.map(prepare_one, tasks, chunksize=len(manifest["variants"])) if executor else map(prepare_one, tasks)
            for index, record in enumerate(results, 1):
                records.append(record)
                if index % 100 == 0:
                    print(f"Metrics: {index}/{len(tasks)}", flush=True)
        finally:
            if executor:
                executor.shutdown(cancel_futures=True)
        if len(records) == len(tasks) and not any(r.get("error") for r in records):
            publish_metric_index(args.output, manifest, records)
        finish(args.output, "metrics", manifest, contract_id, records)


def main():
    parser = argument_parser(__doc__)
    parser.add_argument("--maps-root", type=Path, default=navsim_default(release_paths.nuplan_maps))
    parser.add_argument("--map-version", default="nuplan-maps-v1.0", choices=["nuplan-maps-v1.0"])
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.maps_root is None:
        parser.error("Set navsim_data (or nuplan_maps) in configs/paths.yaml, or pass --maps-root")
    args.maps_root = args.maps_root.expanduser().resolve()
    os.environ["NUPLAN_MAPS_ROOT"] = str(args.maps_root)
    os.environ["NUPLAN_MAP_VERSION"] = args.map_version
    manifest = discover(args)
    manifest = freeze_manifest(args, manifest)
    if args.dry_run:
        return
    contract = metric_contract(args)
    prepare_metrics(args, manifest, contract)


if __name__ == "__main__":
    main()
