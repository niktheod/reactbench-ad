"""ReactBench tables: main/detailed results, trajectory invariance rate (TIR) and dP by ego speed.

Values of the main table use, per (object, tau), the scenes for which every evaluated model has a
valid original/edited pair (common cohort). Its intervals, which decide the gray "CI includes zero"
marks, resample each model's own valid pairs. All intervals: 95% percentile bootstrap over scenes,
10,000 resamples.
"""
from __future__ import annotations

import math

import numpy as np

from ..runs import OBJECTS
from ..scoring.ade import THRESHOLDS_M
from ..scoring.collisions import paired_contact_speed
from ..scoring.motion_bins import speed_bin_sort_key
from ..scoring.reporting import comparison_summary, pair_records
from .runs import DATASETS, MODELS
from .stats import aggregate_bin, bootstrap_mean


def _indexed(rows, dataset):
    """Augmented-world rows keyed by (token, insertion world)."""
    indexed = {}
    for row in rows:
        if not row.get("world_fingerprint") or (dataset == "reactnav" and row.get("world") != "augmented"):
            continue
        key = (row["token"], row["world_fingerprint"])
        if key in indexed:
            raise ValueError(f"Duplicate scene/insertion {key}")
        indexed[key] = row
    return indexed


def _valid(row, name, dataset):
    score = row.get("scores", {}).get(name)
    if dataset == "reactnav":
        return isinstance(score, dict) and score.get("nc_dac") is not None
    return isinstance(score, dict) and isinstance(score.get("all"), bool)


def own_pairs(rows, dataset, before="original", after="edited", tokens=None):
    """Per-scene dP (percentage points) and contact-speed changes (km/h) over the model's own valid pairs.

    ``tokens`` optionally restricts the pairs to a set of scene tokens.
    """
    indexed = _indexed(rows, dataset)
    if tokens is not None:
        indexed = {k: r for k, r in indexed.items() if k[0] in tokens}
    keys = sorted(k for k, r in indexed.items() if _valid(r, before, dataset) and _valid(r, after, dataset))
    field = DATASETS[dataset]["primary"]
    if dataset == "reactnav":
        delta = [100. * (indexed[k]["scores"][after][field] - indexed[k]["scores"][before][field]) for k in keys]
        contact_rows = sorted(k for k, r in indexed.items()
                              if all(isinstance(r["scores"].get(n), dict) and isinstance(r["scores"][n].get("all"), bool)
                                     for n in (before, after)))
    else:
        delta = [100. * (int(indexed[k]["scores"][before]["all"]) - int(indexed[k]["scores"][after]["all"])) for k in keys]
        contact_rows = keys
    contacts = {k: paired_contact_speed(indexed[k]["scores"][before], indexed[k]["scores"][after]) for k in contact_rows}
    speed = [3.6 * (c["to_speed_mps"] - c["from_speed_mps"]) for c in contacts.values() if c["status"] == "scored"]
    return {"keys": keys, "delta_pp": delta, "contact_delta_kmh": speed,
            "groups": {k: indexed[k].get("groups", {}) for k in keys}}


def main_table(runs, dataset, models=None):
    """One record per (object, tau, model), as in the paper's main and detailed tables."""
    metric = DATASETS[dataset]["primary"]
    settings = {}
    for model in models or MODELS[dataset]:
        for (obj, tau), run in runs.insertion_runs(model, dataset).items():
            if obj not in OBJECTS:
                continue
            rows = runs.rows(run, dataset)
            valid = [p["row"] for p in pair_records(rows, metric) if p["row"].get("primary_pair") == "edited"]
            indexed = {(r["token"], r["world_fingerprint"]): r for r in valid}
            if len(indexed) != len(valid):
                raise ValueError(f"Duplicate scene/insertion: {run}")
            settings.setdefault((obj, tau), {})[model] = (rows, indexed)
    records = []
    for (obj, tau), by_model in sorted(settings.items(), key=lambda item: (OBJECTS.index(item[0][0]), item[0][1])):
        available = [set(indexed) for _, indexed in by_model.values() if indexed]
        common = set.intersection(*available) if available else set()
        for model, (rows, indexed) in by_model.items():
            selected = [indexed[k] for k in sorted(common) if k in indexed]
            stat = comparison_summary(selected, metric)
            primary, inserted, contact = stat["primary"], stat["nc_inserted"], stat["contact_speed"]
            own = own_pairs(rows, dataset)

            def scale(value, factor):
                return value * factor if value is not None else None
            records.append({
                "dataset": dataset, "model": model, "object": obj, "tau_s": tau, "primary_metric": metric,
                "p_original_pct": scale(primary["from_rate"], 100), "p_edited_pct": scale(primary["to_rate"], 100),
                "delta_p_pp": primary["improvement_pp"],
                "nc_ins_original_pct": scale(inserted["from_rate"], 100), "nc_ins_edited_pct": scale(inserted["to_rate"], 100),
                "delta_nc_ins_pp": inserted["improvement_pp"],
                "cs_original_kmh": scale(contact["mean_from_speed_mps"], 3.6),
                "cs_edited_kmh": scale(contact["mean_to_speed_mps"], 3.6),
                "delta_cs_kmh": scale(contact["mean_delta_mps"], 3.6),
                "delta_p_ci95_pp": bootstrap_mean(own["delta_pp"]),
                "delta_cs_ci95_kmh": bootstrap_mean(own["contact_delta_kmh"]),
                "paired_n": stat["paired_count"], "contact_n": contact["count"],
                "own_paired_n": len(own["keys"]), "own_contact_n": len(own["contact_delta_kmh"]),
                "invalid_pair_n": len(rows) - len(indexed), "unmatched_pair_n": len(indexed) - len(selected)})
    return records


def tir(runs, dataset, models=None, *, curves=False):
    """Percentage of pairs with original/edited ADE below each threshold, pooled over tau."""
    records, curve_rows = [], []
    for model in models or MODELS[dataset]:
        found = runs.insertion_runs(model, dataset)
        for obj in OBJECTS:
            values = []
            for (o, tau), run in sorted(found.items()):
                if o != obj:
                    continue
                for row in runs.rows(run, dataset):
                    value = row.get("ade_m")
                    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
                        values.append(float(value))
            if not values:
                continue
            values = np.sort(np.asarray(values))
            record = {"dataset": dataset, "model": model, "object": obj, "n": len(values)}
            for threshold in THRESHOLDS_M:
                record[f"tir_{threshold:.2f}_pct"] = 100. * np.searchsorted(values, threshold, side="left") / len(values)
            records.append(record)
            if curves:
                x = np.unique(np.concatenate(([0.0], THRESHOLDS_M, values[values <= .5])))
                for xi, pct in zip(x, 100 * np.searchsorted(values, x, side="left") / len(values)):
                    curve_rows.append({"dataset": dataset, "model": model, "object": obj, "epsilon_m": float(xi),
                                       "tir_pct": float(pct)})
    return records, curve_rows


def by_speed(runs, dataset, models=None, *, min_bin_count=10):
    """dP per ego-speed bin (5 km/h) at the current frame, equal-weight mean over the four tau."""
    records = []
    for model in models or MODELS[dataset]:
        found = runs.insertion_runs(model, dataset)
        for obj in OBJECTS:
            taus = sorted(tau for o, tau in found if o == obj)
            if len(taus) != 4:
                continue
            cells = {}
            for tau in taus:
                own = own_pairs(runs.rows(found[obj, tau], dataset), dataset)
                groups = {}
                for key, value in zip(own["keys"], own["delta_pp"]):
                    label = own["groups"][key].get("speed_bin_kmh")
                    if label is not None:
                        groups.setdefault(label, {})[key[0]] = value
                cells[tau] = {label: {"n": len(v), "scene_deltas": v} for label, v in groups.items()}
            for label in sorted({b for c in cells.values() for b in c}, key=speed_bin_sort_key):
                value = aggregate_bin([cells[tau].get(label, {}) for tau in taus], min_bin_count=min_bin_count)
                records.append({"dataset": dataset, "model": model, "object": obj, "speed_bin_kmh": label,
                                "visible": value["visible"], "n": value.get("n"), "delta_p_pp": value["delta"],
                                "ci95_low_pp": value["ci95"][0], "ci95_high_pp": value["ci95"][1]})
    return records
