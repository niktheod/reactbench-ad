"""Export the per-sample scores behind every table, without trajectories, CoTs or images.

The export mirrors the outputs layout, so ``REACTBENCH_OUTPUTS=<export> python -m reactbench.summarize``
reproduces the tables from it:

- ``predictions/<model>/<dataset>/<run>/metrics/*_at_fault_per_sample.json`` of every run (insertion runs,
  original scenes, standing animals, CoT swap, fixed CoTs): per sample and condition NC, NCins, DAC,
  P, first-contact speed, ADE between the original and edited plans, and the inserted object's motion;
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
from .io import write_json
from .summarize.runs import DATASETS, Runs

MODELS = ("alpamayo1_5", "alpamayo2", "autovla", "recogdrive", "recogdrive_vlm", "recogdrive_il", "recogdrive_il_rt",
          "drivewam", "simwam", "simwam_il")


def relative(value):
    """Absolute paths reduced to their last two components, recursively."""
    if isinstance(value, dict):
        return {k: relative(v) for k, v in value.items()}
    if isinstance(value, list):
        return [relative(v) for v in value]
    if isinstance(value, str) and value.startswith("/") and "\n" not in value:
        return "/".join(Path(value).parts[-2:])
    return value


def copy_json(source, target):
    write_json(target, relative(json.loads(Path(source).read_text())))


def export_predictions(runs, output):
    count = 0
    for model in MODELS:
        for dataset, spec in DATASETS.items():
            root = runs.directory(model, dataset)
            if not root.is_dir():
                continue
            for run in sorted(p for p in root.iterdir() if p.is_dir()):
                for path in sorted((run / "metrics").glob(f"{spec['prefix']}_*.json")):
                    copy_json(path, output / "predictions" / model / dataset / run.name / "metrics" / path.name)
                    count += 1
    return count


def export_knockout(output):
    from .analysis.knockout import analyze
    summary = paths.outputs("knockout", "summary.json")
    if not summary.is_file():
        return 0
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
