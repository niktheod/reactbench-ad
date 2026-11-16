"""Export the per-sample scores behind every table, without trajectories, CoTs or images.

The export mirrors the outputs layout, so ``REACTBENCH_OUTPUTS=<export> python -m reactbench.summarize``
reproduces the tables from it:

- ``predictions/<model>/<dataset>/<run>/metrics/*_at_fault_{per_sample,summary,manifest}.json`` of every run
  (insertion runs, standing animals, CoT swap, fixed CoTs): per sample and condition NC, NCins,
  DAC, P, first-contact speed, ADE between the original and edited plans, and the inserted object's motion. The
  manifest is marked ``exported`` (with the checksum of the exported rows), so that ``summarize`` reads the rows
  without the predictions they were computed from;
- ``real_scenes/``: per scene, model and headway the absolute P and contact speed (scores and summaries);
- ``knockout/summary.json`` and ``knockout/per_token.json`` (ADE/FDE of the knockout and reseeded rollouts);
- ``exploration_probe/*.json`` (per group the executed PDMS and ReactBench P of each candidate);
- ``navtest_pdms/<checkpoint>/**/*.csv`` (per-scene navtest PDMS) and ``summaries/feasibility``.

Absolute paths inside the files are reduced to their last two components.

    python -m reactbench.export OUTPUT_DIR [--runs MODEL:DATASET=DIR ...]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from . import paths
from .io import file_digest, relative, write_json
from .runs import parse_run_name
from .scoring.rows import read_rows
from .summarize.runs import DATASETS, Runs

MODELS = ("alpamayo1_5", "alpamayo2", "autovla", "autovla_l0r0", "recogdrive", "recogdrive_vlm", "recogdrive_il",
          "recogdrive_il_rt", "drivewam", "simwam", "simwam_il")


MACHINE_KEYS = ("source_documents", "config", "data_config", "backend_provenance", "result_dir", "run_dir",
                "original_context_result")


def copy_json(source, target):
    write_json(target, relative(json.loads(Path(source).read_text())))


def export_run(source, target, prefix):
    """Rows and summary with paths reduced, and a manifest that marks them as exported."""
    rows, manifest = read_rows(source.parent, prefix)  # checks the scores against their predictions first
    if rows is None:
        return 0
    for name in ("per_sample", "summary"):
        if (source / f"{prefix}_{name}.json").is_file():
            copy_json(source / f"{prefix}_{name}.json", target / f"{prefix}_{name}.json")
    # Keep what describes the scores; the run's configuration and paths describe the machine.
    manifest = relative({k: v for k, v in manifest.items() if k not in MACHINE_KEYS})
    manifest.update(exported=True, per_sample_sha256=file_digest(target / f"{prefix}_per_sample.json"))
    write_json(target / f"{prefix}_manifest.json", manifest)
    return 1


def paper_run(name, dataset):
    """Insertion runs and the CoT swap / fixed-CoT runs of the tables (tau = 2.0 s); no development runs."""
    base = name.removesuffix("_cot_swap").removesuffix("_fixed_cots")
    parsed = parse_run_name(base, DATASETS[dataset]["dataset"])
    return parsed is not None and (base == name or parsed[2] == 2.0)


def export_predictions(runs, output):
    count = 0
    found = paths.outputs("predictions")
    others = sorted(p.name for p in found.iterdir() if p.is_dir() and p.name not in MODELS) if found.is_dir() else []
    for model in [*MODELS, *others]:
        for dataset, spec in DATASETS.items():
            root = runs.directory(model, dataset)
            if not root.is_dir():
                continue
            for run in sorted(p for p in root.iterdir() if p.is_dir() and paper_run(p.name, dataset)):
                count += export_run(run / "metrics", output / "predictions" / model / dataset / run.name / "metrics",
                                    spec["prefix"])
    return count


def export_knockout(output):
    summary = paths.outputs("knockout", "summary.json")
    if not summary.is_file():
        return 0
    from .analysis.knockout import analyze  # imports torch
    copy_json(summary, output / "knockout" / "summary.json")
    tokens = analyze.read_tokens_file(analyze.DEFAULT_TOKENS_FILE)
    per_token = {model: {token: analyze.token_metrics(analyze.load_result(analyze.DEFAULT_OUTPUT_ROOT / model / f"{token}.npz")["poses"])
                         for token in tokens} for model in analyze.MODEL_ORDER}
    write_json(output / "knockout" / "per_token.json", per_token)
    return 2


def export_tree(source, target, patterns):
    count = 0
    for pattern in patterns:
        for path in sorted(Path(source).glob(pattern)):
            destination = target / path.relative_to(source)
            if path.suffix == ".json":
                copy_json(path, destination)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
            count += 1
    return count


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", type=Path)
    parser.add_argument("--runs", nargs="*", default=[], metavar="MODEL:DATASET=DIR")
    args = parser.parse_args(argv)
    runs = Runs.from_arguments(args.runs)
    counts = {"predictions": export_predictions(runs, args.output), "knockout": export_knockout(args.output)}
    for name, patterns in (("real_scenes", ["runs/*/*/scores.json", "runs/*/*/summary.json", "summary/*/summary.json"]),
                           ("exploration_probe", ["*.json"]), ("navtest_pdms", ["**/*.csv"]),
                           ("summaries/feasibility", ["*.json"])):
        counts[name] = export_tree(paths.outputs(name), args.output / name, patterns)
    print(json.dumps(counts, indent=1))


if __name__ == "__main__":
    main()
