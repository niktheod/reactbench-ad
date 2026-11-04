"""Unavoidable collisions: dP at tau = 0.5 s on all samples and on the avoidable ones only.

A sample is unavoidable when no constant deceleration up to 8 m/s^2 along the ground-truth path
avoids the inserted object (verdicts from ``python -m reactbench.scoring.feasibility``). Both
columns use each model's own valid pairs; gray = the 95% bootstrap CI includes zero.
"""
from __future__ import annotations

from pathlib import Path

from .. import paths
from ..io import read_json
from ..runs import OBJECTS
from .benchmark import own_pairs
from .runs import MODELS
from .stats import bootstrap_mean

TAU_S = 0.5


def verdicts(dataset, directory=None):
    """{(object, tau): {token: feasible}} and the unavoidable share per (object, tau)."""
    path = Path(directory or paths.outputs("summaries", "feasibility")) / f"{dataset}_per_sample.json"
    document = read_json(path)
    if document is None:
        raise FileNotFoundError(f"Missing {path}; run python -m reactbench.scoring.feasibility first")
    result = {}
    for row in document["samples"]:
        result.setdefault((row["object"], row["tau_s"]), {})[row["token"]] = row["feasible"]
    return result


def table(runs, dataset, models=None, directory=None):
    found = verdicts(dataset, directory)
    records = []
    for obj in OBJECTS:
        cell = found.get((obj, TAU_S))
        if cell is None:
            continue
        unavoidable_pct = 100. * sum(not v for v in cell.values()) / len(cell)
        feasible = {token for token, ok in cell.items() if ok}
        for model in models or MODELS[dataset]:
            run = runs.insertion_runs(model, dataset).get((obj, TAU_S))
            if run is None:
                continue
            rows = runs.rows(run, dataset)
            missing = {r["token"] for r in rows if r.get("world_fingerprint")} - set(cell)
            if missing:
                raise ValueError(f"No feasibility verdict for {len(missing)} {dataset} {obj} tokens")
            everything, avoidable = own_pairs(rows, dataset), own_pairs(rows, dataset, tokens=feasible)
            records.append({"dataset": dataset, "model": model, "object": obj, "tau_s": TAU_S,
                            "unavoidable_pct": unavoidable_pct,
                            "all_n": len(everything["keys"]),
                            "all_delta_p_pp": sum(everything["delta_pp"]) / len(everything["keys"]),
                            "all_ci95_pp": bootstrap_mean(everything["delta_pp"]),
                            "avoidable_n": len(avoidable["keys"]),
                            "avoidable_delta_p_pp": sum(avoidable["delta_pp"]) / len(avoidable["keys"]),
                            "avoidable_ci95_pp": bootstrap_mean(avoidable["delta_pp"])})
    return records
