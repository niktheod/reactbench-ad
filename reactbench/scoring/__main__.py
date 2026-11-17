"""Score saved predictions on ReactNAV or ReactPAI.

    python -m reactbench.scoring PREDICTIONS [PREDICTIONS ...] [--variant VARIANT_DIR] [--workers N] [--output-root DIR]

PREDICTIONS is one run of predictions, ``<model>/<reactnav|reactpai>/<variant>`` holding
``trajectories/<token>.npz`` (README "Evaluating your model"), or a folder of such runs, e.g.
``<outputs>/predictions/<model>/reactnav``. The original-scene predictions are in the sibling ``original/``. The
benchmark variant defaults to ``<reactnav|reactpai>/<variant>`` from configs/paths.yaml (``--variant`` names it for
a single run). Each run gets ``metrics/<navsim|paiav>_at_fault_{per_sample,summary,manifest}.json``, under
``--output-root`` when given. ReactNAV scoring needs the NAVSIM devkit (``scoring_python``), ReactPAI scoring
the PAI-AV download (``paiav_root``). Exits non-zero if any run fails.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .. import paths
from .results import DATASET_FOLDERS

PATH_KEYS = {"navsim": "reactnav", "pai-av": "reactpai"}


def is_run(path):
    return (path / "trajectories").is_dir() or (path / "metrics" / "summary.json").is_file()


def runs_under(path):
    """The prediction runs of PREDICTIONS (the original scenes are inputs, not runs)."""
    path = Path(path).resolve()
    if is_run(path):
        return [path]
    return sorted(p for p in path.iterdir() if p.is_dir() and p.name != "original" and is_run(p))


def benchmark_variant(run, dataset, explicit):
    """The benchmark variant of a predictions run: explicit, named by its bookkeeping, or <reactnav|reactpai>/<name>."""
    if explicit is not None:
        return Path(explicit).resolve()
    if (run / "metrics" / "summary.json").is_file():
        return None
    return paths.get(PATH_KEYS[dataset]) / run.name


def score(job):
    run, dataset, bench, output, workers = job
    try:
        if dataset == "navsim":
            from . import navsim
            summary = navsim.process_result(run, navsim.load_config({"workers": workers}), output_root=output,
                                            variant_dir=bench)
            coverage = summary.get("overall", {}).get("coverage", summary.get("status"))
        else:
            from . import paiav
            coverage = paiav.process_result(run, paiav.load_config({"workers": 1}), output_root=output,
                                            variant_dir=bench).get("coverage")
        return str(run), coverage, None
    except Exception as exc:
        return str(run), None, f"{type(exc).__name__}: {exc}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("predictions", nargs="+", type=Path, help="A predictions run, or a folder of runs")
    parser.add_argument("--variant", type=Path, metavar="VARIANT_DIR",
                        help="Benchmark variant (final_images/, metadata/) of a single predictions run")
    parser.add_argument("--dataset", choices=sorted(PATH_KEYS.values()),
                        help="Default: the parent folder of each run (reactnav or reactpai)")
    parser.add_argument("--workers", type=int, default=1, help="CPU processes (default 1)")
    parser.add_argument("--output-root", type=Path,
                        help="Write metrics/ under OUTPUT_ROOT/<run> instead of next to the predictions")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    jobs = []
    for base in args.predictions:
        runs = runs_under(base)
        if not runs:
            parser.error(f"no predictions run under {base}")
        if args.variant is not None and (len(runs) > 1 or len(args.predictions) > 1):
            parser.error("--variant names the benchmark variant of a single predictions run")
        for run in runs:
            dataset = ({v: k for k, v in PATH_KEYS.items()}[args.dataset] if args.dataset
                       else DATASET_FOLDERS.get(run.parent.name))
            if dataset is None:
                parser.error(f"cannot tell the dataset of {run}: use <model>/<reactnav|reactpai>/<variant> or --dataset")
            output = None
            if args.output_root is not None:
                output = args.output_root / (run.name if run == base.resolve() else run.relative_to(base.resolve()))
            jobs.append((run, dataset, benchmark_variant(run, dataset, args.variant), output, args.workers))
    # ReactNAV parallelizes within a run (its scoring backend); ReactPAI across runs.
    navsim_jobs = [job for job in jobs if job[1] == "navsim"]
    paiav_jobs = [job for job in jobs if job[1] == "pai-av"]
    results = [score(job) for job in navsim_jobs]
    if args.workers > 1 and len(paiav_jobs) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            results += list(pool.map(score, paiav_jobs))
    else:
        results += [score(job) for job in paiav_jobs]
    failures = 0
    for run, coverage, error in results:
        print(run, coverage if error is None else f"FAILED {error}", flush=True)
        failures += error is not None or (isinstance(coverage, dict) and
                                          any(coverage.get(k) for k in ("failed", "partial", "invalid")))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
