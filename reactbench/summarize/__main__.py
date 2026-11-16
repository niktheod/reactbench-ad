"""Print the paper's tables from scored runs and write them as JSON/CSV.

    python -m reactbench.summarize                      # every table whose inputs exist
    python -m reactbench.summarize --tables main tir    # selected tables
    python -m reactbench.summarize --tir-curves --by-speed
    python -m reactbench.summarize --models alpamayo1_5 mymodel --tables main tir   # your model, scored runs

Tables: main (main and detailed results), tir (trajectory invariance rate), side_cameras (AutoVLA with
CAM_L1/R1 or CAM_L0/R0), training_stages, fine_tuning (ReactNAV and navtest), generalization (unseen objects,
real scenes), real_scene_control (released single-frame models on the real scenes), speed_distribution,
feasibility (tau = 0.5 s, unavoidable samples), cot (CoT swap, fixed CoTs), knockout (attention knockout) and
exploration_probe.

Scored runs are found under ``<outputs>/predictions/<model>/<dataset>/`` (see ``runs.py``);
``--runs MODEL:DATASET=DIR`` overrides one location. ``--models`` selects the models of the benchmark tables
(default: the six of the paper); any folder name under ``predictions/`` is a model. The other inputs are the outputs of their
own commands (README): ``<outputs>/summaries/feasibility``, ``<outputs>/real_scenes``,
``<outputs>/knockout/summary.json``, ``<outputs>/exploration_probe`` and ``<outputs>/navtest_pdms``.
Outputs go to ``<outputs>/summaries``.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from .. import paths
from ..io import read_json, write_json
from . import benchmark, cot, feasibility, markdown, training
from .runs import DATASETS, MODELS, Runs

TABLES = ("main", "tir", "side_cameras", "training_stages", "fine_tuning", "generalization", "real_scene_control",
          "speed_distribution", "feasibility", "cot", "knockout", "exploration_probe")


def write_csv(path, records):
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows({k: (v if not isinstance(v, (list, dict)) else " ".join(map(str, v))) for k, v in r.items()}
                         for r in records)


def _skip(table, reason):
    print(f"({table}: skipped, {reason})", file=sys.stderr)


def benchmark_tables(runs, args, output):
    """Main, detailed and TIR tables of each dataset, and the optional CSVs."""
    text = []
    for dataset in args.datasets:
        models = [m for m in (args.models or MODELS[dataset]) if runs.insertion_runs(m, dataset)]
        if not models:
            _skip(f"{dataset} benchmark tables", f"no scored {dataset} runs")
            continue
        if "main" in args.tables:
            records = benchmark.main_table(runs, dataset, models)
            write_json(output / f"main_{dataset}.json", records)
            write_csv(output / f"main_{dataset}.csv", records)
            text += [markdown.main_table(records, dataset), markdown.detailed_table(records, dataset)]
        if "tir" in args.tables:
            records, curves = benchmark.tir(runs, dataset, models, curves=args.tir_curves)
            write_json(output / f"tir_{dataset}.json", records)
            text.append(markdown.tir_table(records))
            if args.tir_curves:
                write_csv(output / f"tir_curves_{dataset}.csv", curves)
        if "side_cameras" in args.tables and dataset == "reactnav":
            data = benchmark.side_cameras(runs)
            if data:
                write_json(output / "side_cameras.json", data)
                text.append(markdown.side_camera_table(data))
            else:
                _skip("side_cameras", "needs the scored autovla and autovla_l0r0 runs")
        if args.by_speed:
            write_csv(output / f"delta_p_by_speed_{dataset}.csv", benchmark.by_speed(runs, dataset, models))
        if "feasibility" in args.tables:
            try:
                records = feasibility.table(runs, dataset, models)
            except FileNotFoundError as error:
                _skip(f"{dataset} feasibility", str(error))
            else:
                write_json(output / f"feasibility_{dataset}.json", records)
                text.append(markdown.feasibility_table(records))
    return text


def other_tables(runs, args, output):
    text = []
    if "training_stages" in args.tables:
        records = training.training_stages(runs)
        write_json(output / "training_stages.json", records)
        text.append(markdown.training_stages_table(records))
    if "fine_tuning" in args.tables:
        try:
            records = training.fine_tuning(runs)
        except FileNotFoundError as error:
            _skip("fine_tuning", str(error))
        else:
            shortening = training.original_plan_shortening(runs)
            write_json(output / "fine_tuning.json", records)
            if shortening:
                write_json(output / "fine_tuning_original_plans.json", shortening)
            text.append(markdown.fine_tuning_table(records, shortening))
    if "generalization" in args.tables:
        try:
            unseen = training.unseen_objects(runs)
        except FileNotFoundError as error:
            _skip("unseen objects", str(error))
            unseen = []
        real = training.real_scenes()
        write_json(output / "generalization.json", {"unseen_objects": unseen, "real_scenes": real})
        text.append(markdown.generalization_table(unseen, real))
    if "real_scene_control" in args.tables:
        records = training.real_scene_control()
        write_json(output / "real_scene_control.json", records)
        text.append(markdown.real_scene_control_table(records))
    if "speed_distribution" in args.tables:
        counts = {}
        for dataset in args.datasets:
            model = next((m for m in (args.models or MODELS[dataset]) if runs.insertion_runs(m, dataset)), None)
            counts[dataset] = training.speed_distribution(runs, dataset, model) if model else {}
        write_json(output / "speed_distribution.json", counts)
        text.append(markdown.speed_distribution_table(counts))
    if "cot" in args.tables:
        swap, fixed = cot.swap_table(runs), cot.fixed_table(runs)
        write_json(output / "cot.json", {"swap": swap, "fixed": fixed})
        text += [markdown.cot_swap_table(swap), markdown.fixed_cot_table(fixed)]
    if "knockout" in args.tables:
        summary = read_json(paths.outputs("knockout", "summary.json"))
        if summary is None:
            _skip("knockout", "run python -m reactbench.analysis.knockout.analyze first")
        text.append(markdown.knockout_table(summary))
    if "exploration_probe" in args.tables:
        records = training.exploration_probe()
        write_json(output / "exploration_probe.json", records)
        text.append(markdown.exploration_probe_table(records))
    return text


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tables", nargs="+", choices=TABLES, default=list(TABLES))
    parser.add_argument("--runs", nargs="*", default=[], metavar="MODEL:DATASET=DIR")
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=sorted(DATASETS))
    parser.add_argument("--models", nargs="+", help="Models of the benchmark tables (default: the paper's six)")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--tir-curves", action="store_true", help="Also write the TIR curves (CSV)")
    parser.add_argument("--by-speed", action="store_true", help="Also write dP by ego speed (CSV)")
    args = parser.parse_args(argv)
    runs = Runs.from_arguments(args.runs)
    output = args.output_dir or paths.outputs("summaries")
    text = benchmark_tables(runs, args, output) + other_tables(runs, args, output)
    text = "\n\n".join(t for t in text if t) + "\n"
    output.mkdir(parents=True, exist_ok=True)
    (output / "tables.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
