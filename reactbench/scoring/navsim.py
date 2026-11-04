"""Score saved ReactNAV predictions (protocol V2: at-fault NC, NCins, DAC, contact speed, ADE).

Usage::

    python -m reactbench.scoring.navsim RESULT_DIR [RESULT_DIR ...] [--workers 8] [--output-root DIR]

Each run gets ``metrics/navsim_at_fault_{per_sample,summary,manifest}.json`` (under ``--output-root``
when given, otherwise next to the predictions). Scoring runs in a CPU subprocess whose Python must
provide the NAVSIM devkit and nuplan-devkit (``scoring_python`` in ``configs/paths.yaml``).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

from .. import paths
from ..io import digest, file_digest, read_json, write_json
from . import results as saved
from .collisions import paired_outcome
from .navsim_metrics import AT_FAULT_SCORING_CONTRACT
from .reporting import summarize, summarize_original
from .results import load_prediction

PREFIX = "navsim_at_fault"
PRIMARY_METRIC = "nc_dac"
FINGERPRINTED = ("navsim_metrics.py", "navsim_batch.py", "navsim_world.py", "collisions.py", "at_fault.py",
                 "navsim_lanes.py", "ade.py", "geometry.py", "actor_motion.py", "pedestrian_geometry.py")


def _optional(location):
    try:
        return location()
    except KeyError:
        return ""


def load_config(overrides=None):
    """Data locations come from configs/paths.yaml; ``overrides`` replaces individual keys."""
    config = {"version": 1, "map_version": "nuplan-maps-v1.0", "workers": 1,
              "backend_python": str(paths.get("scoring_python", sys.executable)),
              "navsim_root": str(paths.get("navsim_devkit", "third_party/recogdrive")),
              "log_path": str(_optional(lambda: paths.navsim_logs("test"))),
              "maps_root": str(_optional(paths.nuplan_maps) or os.environ.get("NUPLAN_MAPS_ROOT", "")),
              "work_root": str(paths.get("scoring_work", paths.outputs("work", "navsim")))}
    config.update({k: v for k, v in (overrides or {}).items() if v is not None})
    for key in ("navsim_root", "log_path", "maps_root", "work_root", "backend_python"):
        if config.get(key):
            config[key] = str(paths.resolve(config[key]))
    workers = config["workers"]
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    return config


def backend_call(inventory, config):
    """Score an inventory in the configured CPU Python (never a scheduler)."""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join((config["navsim_root"], str(paths.REPO_ROOT)))
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["NUPLAN_MAPS_ROOT"] = config["maps_root"]
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = "1"
    with tempfile.TemporaryDirectory(prefix="navsim-scoring-") as scratch:
        environment["MPLCONFIGDIR"] = str(Path(scratch) / "matplotlib")
        environment["XDG_CACHE_HOME"] = str(Path(scratch) / "cache")
        environment["NUMBA_CACHE_DIR"] = str(Path(scratch) / "numba")
        output = Path(scratch) / "response.json"
        command = [config["backend_python"], "-m", "reactbench.scoring.navsim", "--backend", "--response", str(output)]
        completed = subprocess.run(command, input=json.dumps({"inventory": inventory, "config": config}),
                                   text=True, env=environment, cwd=paths.REPO_ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, check=False)
        if completed.returncode or not output.is_file():
            raise RuntimeError(f"NAVSIM scoring backend failed ({completed.returncode}): {completed.stderr[-4000:]}")
        return read_json(output)


def postprocess_evaluation(result_dir, mode="score", workers=1):
    """Called by the runners after inference; a scoring failure never hides saved predictions."""
    if mode == "off":
        return
    try:
        summary = process_result(result_dir, load_config({"workers": workers}))
        print(f"ReactNAV scoring: {result_dir}; coverage={summary.get('overall', {}).get('coverage', {})}", flush=True)
    except Exception as exc:
        print(f"ReactNAV scoring failed; saved predictions remain available: {exc}", flush=True)
        write_json(Path(result_dir) / f"metrics/{PREFIX}_error.json", {"status": "failed", "error": str(exc)})


def inventory_result(root):
    inventory = saved.inventory_result(root)
    records = {str(r["token"]): r for r in read_json(Path(root) / "metrics/per_sample.json", [])}
    for task in inventory["tasks"]:
        if task["dataset"] != "navsim":
            task.update(status="inapplicable", reason="ReactNAV scoring requires a NAVSIM scene")
        if task["world"] == "clean":
            task["actor_motion"] = None
        row = records.get(task["token"], {})
        if inventory["mode"] == "fixed_cots":
            from ..analysis.cot.fixed_cot import CONDITIONS
            for name in CONDITIONS:
                task["pairs"][f"intervention/{name}"] = ["natural_edited", name]
        for name, c in row.get("contrasts", {}).items():
            if "from_condition" in c and "to_condition" in c:
                task["pairs"][f"contrast/{name}"] = [c["from_condition"], c["to_condition"]]
        for name, c in row.get("replay_controls", {}).items():
            if c.get("reference_condition"):
                task["pairs"][f"replay/{name}"] = [c["reference_condition"], name]
        task["input_fingerprint"] = digest({k: v for k, v in task.items() if k != "input_fingerprint"})
    return inventory


def implementation_fingerprint(config):
    from .navsim_lanes import lanes_identity
    here = Path(__file__).parent
    return digest({"code": {name: file_digest(here / name) for name in FINGERPRINTED},
                   "lanes": lanes_identity(config["work_root"])})


def score_task(task, worlds, config, contract):
    from collections import defaultdict
    from .ade import paired_ade
    from .motion_bins import acceleration_bin_label, speed_bin_label
    from .navsim_batch import score_predictions
    from .navsim_metrics import MapContext, at_fault_from_physical, score_prediction_at_fault
    row = {"token": task["token"], "world": task["world"], "status": task["status"], "reason": task.get("reason"),
           "groups": dict(task["groups"]), "scores": {}, "pairs": {}, "errors": {},
           "primary_pair": task.get("primary_pair", task["primary_condition"]), "prediction_sources": task["predictions"],
           "input_fingerprint": task["input_fingerprint"]}
    if task["status"] != "ready":
        return row
    try:
        predictions = {}
        for name, source in task["predictions"].items():
            try:
                predictions[name] = load_prediction(source)
            except Exception as exc:
                row["scores"][name] = None
                row["errors"][name] = f"{type(exc).__name__}: {exc}"
        radius = max([100.] + [100. * math.ceil((float(np.linalg.norm(p[:, :2], axis=1).max()) + 10) / 100)
                               for p, dt in predictions.values()])
        world = worlds.load(task["token"], radius)
        row.update(world_fingerprint=digest({"source": world["source"], "actor": task["actor_motion"], "contract": world["contract"]}),
                   actor_motion=task["actor_motion"], geometry=world["geometry"],
                   map_status=world["map_status"], map_error=world.get("map_error"),
                   world_source=world["source"], timestamp_warning=world["source"].get("timestamp_warning"))
        velocity, acceleration = np.array(world["initial_velocity"]), np.array(world["initial_acceleration"])
        speed = float(np.linalg.norm(velocity))
        longitudinal = float(velocity @ acceleration / speed) if speed >= .01 else None
        row["groups"].update(speed_mps=speed, longitudinal_acceleration_mps2=longitudinal)
        # Original-scene means do not use the insertion cohort's frozen motion bins.
        if task["world"] == "augmented":
            row["groups"].update(speed_bin_kmh=speed_bin_label(speed * 3.6, bin_spec="navsim"),
                                 acceleration_bin_mps2=acceleration_bin_label(longitudinal, bin_spec="navsim") if longitudinal is not None else None)
        map_context = MapContext(task["token"], world, config["work_root"])
        cache_root = Path(config["work_root"]) / "scores_at_fault"
        scores, pending = {}, defaultdict(list)
        for name, (poses, dt) in predictions.items():
            key = digest({"world": row["world_fingerprint"], "poses": poses.tolist(), "dt": dt, "contract": contract})
            previous = read_json(cache_root / f"{key}.json")
            if previous and previous.get("payload_sha256") == digest(previous["score"]):
                scores[name] = previous["score"]
            else:
                pending[(poses.shape, dt)].append((name, key))
        for (shape, dt), members in pending.items():
            physical = {}
            try:
                # Batched physical contacts equal the scalar reference; a failure falls back per prediction.
                batch = score_predictions(np.stack([predictions[n][0] for n, _ in members]), dt, task["actor_motion"], world)
                physical = {n: value for (n, _), value in zip(members, batch)}
            except Exception:
                physical = {}
            for name, key in members:
                poses = predictions[name][0]
                try:
                    score = (at_fault_from_physical(physical[name], poses, dt, task["actor_motion"], world, map_context)
                             if name in physical else
                             score_prediction_at_fault(poses, dt, task["actor_motion"], world, map_context))
                    if score["dac"] is not None:
                        write_json(cache_root / f"{key}.json", {"score": score, "payload_sha256": digest(score)})
                    scores[name] = score
                except Exception as exc:
                    row["scores"][name] = None
                    row["errors"][name] = f"{type(exc).__name__}: {exc}"
        for name in predictions:
            if name not in scores:
                continue
            score = scores[name]
            row["scores"][name] = score
            if score["dac"] is None:
                row["errors"][f"{name}/dac"] = score.get("dac_reason", "Missing DAC")
        for label, (before, after) in task["pairs"].items():
            a, b = row["scores"].get(before), row["scores"].get(after)
            row["pairs"][label] = ({"from_condition": before, "to_condition": after, **paired_outcome(a, b)}
                                   if a is not None and b is not None else None)
        if task["world"] == "augmented" and {"original", "edited"} <= set(task["predictions"]):
            try:
                row["ade_m"] = paired_ade({k: task["predictions"][k] for k in ("original", "edited")})
            except Exception as exc:
                row["ade_error"] = f"{type(exc).__name__}: {exc}"
        row["status"] = "scored" if not row["errors"] else "partial" if any(row["scores"].values()) else "failed"
    except Exception as exc:
        row.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
    return row


def process_result(root, config, *, output_root=None, backend=backend_call):
    """Score one run; write metrics under ``output_root`` (default: the run itself)."""
    from .rows import read_rows
    root = Path(root).resolve()
    output_root = Path(output_root).resolve() if output_root else root
    inventory = inventory_result(root)
    if inventory["dataset"] != "navsim":
        return {"status": "inapplicable", "result_dir": str(root), "dataset": inventory["dataset"]}
    if not any(t["status"] == "ready" for t in inventory["tasks"]):
        raise ValueError(f"No scoreable NAVSIM predictions: {root}; inspect the saved-result inventory")
    metrics = output_root / "metrics"
    try:
        previous, manifest = read_rows(output_root, PREFIX) if (metrics / f"{PREFIX}_per_sample.json").is_file() else (None, {})
    except ValueError:
        # Changed inference inputs require rescoring.
        previous, manifest = None, {}
    contract = implementation_fingerprint(config)
    fingerprint = digest(inventory)
    reusable = (previous and manifest.get("inventory_fingerprint") == fingerprint
                and manifest.get("implementation_fingerprint") == contract
                and manifest.get("complete") and manifest.get("data_config") == {k: config[k] for k in ("log_path", "maps_root", "map_version")})
    if reusable:
        rows = previous
    else:
        response = backend(inventory, config)
        rows = response["rows"]
        if not any(any(r.get("scores", {}).values()) for r in rows):
            write_json(metrics / f"{PREFIX}_error.json", {"errors": rows, "config": config})
            raise RuntimeError(f"No predictions could be scored: {root}; see metrics/{PREFIX}_error.json")
        write_json(metrics / f"{PREFIX}_per_sample.json", rows)
        manifest = {"version": 2, "mode": inventory["mode"], "result_dir": str(root), "config": config,
                    "data_config": {k: config[k] for k in ("log_path", "maps_root", "map_version")},
                    "source_documents": inventory["sources"], "inventory_fingerprint": fingerprint,
                    "implementation_fingerprint": contract, "contract": AT_FAULT_SCORING_CONTRACT,
                    "backend_provenance": response.get("backend_provenance", {}),
                    "per_sample_sha256": file_digest(metrics / f"{PREFIX}_per_sample.json"),
                    "complete": all(r["status"] in ("scored", "inapplicable") for r in rows)}
        write_json(metrics / f"{PREFIX}_manifest.json", manifest)
    statistics = (summarize_original(rows, PRIMARY_METRIC) if inventory["mode"] == "original"
                  else summarize(rows, PRIMARY_METRIC))
    summary = {**statistics, "dataset": "navsim", "result_dir": str(root)}
    write_json(metrics / f"{PREFIX}_summary.json", summary)
    (metrics / f"{PREFIX}_error.json").unlink(missing_ok=True)
    return summary


_WORKER = None


def initialize_worker(config):
    global _WORKER
    from .navsim_world import RawWorlds
    _WORKER = (RawWorlds(config), config, implementation_fingerprint(config))


def worker_task(task):
    return score_task(task, *_WORKER)


def backend_main(response_path):
    """Subprocess entry: read {inventory, config} from stdin, score every task, write the rows."""
    from concurrent.futures import ProcessPoolExecutor
    import importlib.metadata
    request = json.load(sys.stdin)
    config, tasks = request["config"], request["inventory"]["tasks"]
    if config["workers"] == 1:
        initialize_worker(config)
        rows = [worker_task(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=config["workers"], initializer=initialize_worker, initargs=(config,)) as pool:
            rows = list(pool.map(worker_task, tasks, chunksize=8))
    provenance = {"contract": AT_FAULT_SCORING_CONTRACT, "implementation_fingerprint": implementation_fingerprint(config),
                  "python": sys.version, "versions": {n: importlib.metadata.version(n) for n in ("numpy", "shapely", "nuplan-devkit")}}
    write_json(response_path, {"rows": rows, "backend_provenance": provenance})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("roots", nargs="*", type=Path, help="Run directories, or directories containing runs")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--output-root", type=Path,
                        help="Write metrics/ for each run under OUTPUT_ROOT/<run path relative to its root>")
    parser.add_argument("--dry-run", action="store_true", help="Inventory only; writes nothing")
    parser.add_argument("--backend", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--response", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.backend:
        return backend_main(args.response)
    config = load_config({"workers": args.workers})
    failures = 0
    for base in args.roots:
        base = base.resolve()
        for run in saved.discover_result_dirs([base]):
            try:
                if args.dry_run:
                    inventory = inventory_result(run)
                    print(run, inventory["mode"], dict(Counter(t["status"] for t in inventory["tasks"])), flush=True)
                    continue
                output = args.output_root / run.relative_to(base) if args.output_root else None
                summary = process_result(run, config, output_root=output)
                print(run, summary.get("overall", {}).get("coverage", summary.get("status")), flush=True)
            except Exception as exc:
                failures += 1
                print(f"{run}: FAILED {type(exc).__name__}: {exc}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
