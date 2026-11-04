"""Paired statistics of scored runs: P, NCins and contact speed (protocol V2, at-fault collisions)."""
from __future__ import annotations

from collections import Counter, defaultdict
from .motion_bins import speed_bin_sort_key
import numpy as np

CONTRACT = {
    "version": 2, "trajectory": "raw predicted rear-axle poses; no controller or vehicle rollout",
    "nc": "1 - any at-fault contact (reactbench/scoring/at_fault.py: stock PDMS classes + pedestrian/cyclist "
          "lateral and reversing rules)",
    "primary": {"pai-av": "nc", "navsim": "nc * dac, per prediction before averaging"},
    "improvement": "100 * mean(to - from), percentage points; positive is better",
    "contact_delta": "to first at-fault-contact ego speed - from first at-fault-contact ego speed, m/s; negative is slower",
    "contact_percentage": "100 * (to - from) / from, per pair; exclude zero reference contact speeds only",
    "contact_cohort": "both predictions have an at-fault contact and both incoming native-segment speeds exist; actors may differ",
    "initial_contact": "collision at t=0 retained; preceding predicted speed unavailable",
}


def mean(values):
    return float(np.mean(values)) if len(values) else None


def contact_summary(pairs):
    contacts = [p.get("contact_speed") or {} for p in pairs]
    good = [c for c in contacts if c.get("status") == "scored"]
    deltas = [c["to_speed_mps"] - c["from_speed_mps"] for c in good]
    percentages = [100 * (c["to_speed_mps"] - c["from_speed_mps"]) / c["from_speed_mps"]
                   for c in good if c["from_speed_mps"] > 0]
    return {"count": len(good), "valid_pair_count": len(pairs),
            "both_collide_count": sum(bool(p["all"]["original_collision"] and p["all"]["edited_collision"])
                                      if "all" in p else c.get("status") in ("scored", "initial_contact", "missing_contact_speed")
                                      for p, c in zip(pairs, contacts)),
            "status_counts": dict(Counter(c.get("status", "missing_contact_speed") for c in contacts)),
            "mean_from_speed_mps": mean([c["from_speed_mps"] for c in good]),
            "mean_to_speed_mps": mean([c["to_speed_mps"] for c in good]),
            "mean_delta_mps": mean(deltas), "median_delta_mps": float(np.median(deltas)) if deltas else None,
            "percentage_count": len(percentages), "zero_reference_speed_count": len(good) - len(percentages),
            "mean_percentage_delta": mean(percentages),
            "median_percentage_delta": float(np.median(percentages)) if percentages else None,
            "different_first_actor_count": sum(c.get("shared_first_contact_actor") is False for c in good)}


def score_value(score, metric):
    if not isinstance(score, dict) or not isinstance(score.get("all"), bool):
        return None
    if metric == "nc":
        return int(not score["all"])
    if metric == "nc_inserted":
        return int(not score["inserted"]) if isinstance(score.get("inserted"), bool) else None
    if metric == "dac":
        return score.get("dac")
    if metric == "nc_dac":
        # A missing map is unavailable even when collision already makes the product fail.
        return int(not score["all"]) * score["dac"] if score.get("dac") is not None else None
    raise ValueError(f"Unknown raw metric: {metric}")


def pair_records(rows, metric="nc", pair_name=None, *, require_primary=True):
    """Same complete cohort for primary and inserted panels; contacts have separate eligibility."""
    result = []
    for row in rows:
        pair = row.get("pairs", {}).get(pair_name or row.get("primary_pair"))
        if not pair:
            continue
        a, b = (row.get("scores", {}).get(pair.get(k)) for k in ("from_condition", "to_condition"))
        primary = [score_value(s, metric) for s in (a, b)]
        inserted = [score_value(s, "nc_inserted") for s in (a, b)]
        if require_primary and any(v is None for v in primary + inserted):
            continue
        if any(s is None for s in (a, b)):
            continue
        result.append({"row": row, "pair": pair, "primary": primary, "inserted": inserted})
    return result


def binary_summary(values):
    transitions = Counter("improved" if a < b else "worsened" if a > b else "both_pass" if a else "both_fail"
                          for a, b in values)
    return {"count": len(values), "from_rate": mean([a for a, _ in values]),
            "to_rate": mean([b for _, b in values]),
            "improvement_pp": mean([100 * (b - a) for a, b in values]),
            "transitions": {k: transitions[k] for k in ("improved", "worsened", "both_fail", "both_pass")}}


def comparison_summary(rows, metric="nc", pair_name=None):
    pairs = pair_records(rows, metric, pair_name)
    contacts = pair_records(rows, metric, pair_name, require_primary=False)
    return {"sample_count": len(rows), "paired_count": len(pairs), "excluded_pair_count": len(rows) - len(pairs),
            "primary": binary_summary([p["primary"] for p in pairs]),
            "nc_inserted": binary_summary([p["inserted"] for p in pairs]),
            "contact_speed": contact_summary([p["pair"] for p in contacts]),
            "coverage": dict(Counter(r.get("status", "unknown") for r in rows)),
            "errors": dict(Counter(str(v) for r in rows for v in r.get("errors", {}).values()))}


def transition_tokens(rows, metric):
    """Tokens whose primary-pair outcome flips (improved: fail -> pass), for the metric and for NCins."""
    pairs = pair_records(rows, metric)

    def tokens(key, change):
        return sorted(str(p["row"]["token"]) for p in pairs if p[key][1] - p[key][0] == change)
    return {name: {"improved": tokens(key, 1), "worsened": tokens(key, -1)}
            for name, key in ((metric, "primary"), ("nc_inserted", "inserted"))}


def summarize(rows, metric="nc"):
    groups = {}
    for field in ("actor_type", "placement_timestep", "speed_bin_kmh", "acceleration_bin_mps2"):
        buckets = defaultdict(list)
        for row in rows:
            if row.get("groups", {}).get(field) is not None:
                buckets[str(row["groups"][field])].append(row)
        labels = sorted(buckets, key=speed_bin_sort_key if field in ("speed_bin_kmh", "acceleration_bin_mps2") else None)
        groups[field] = {k: comparison_summary(buckets[k], metric) for k in labels}
    return {"version": CONTRACT["version"], "contract": CONTRACT, "family": "at_fault",
            "transition_tokens": transition_tokens(rows, metric), "primary_metric": metric, "overall": comparison_summary(rows, metric), "groups": groups,
            "by_pair": {name: comparison_summary(rows, metric, name)
                        for name in sorted({n for r in rows for n in r.get("pairs", {})})}}


def summarize_original(rows, metric="nc_dac"):
    """Absolute clean-scene rates, using one valid cohort for the requested metric."""
    nc = "nc"
    if metric not in (nc, "nc_dac"):
        raise ValueError(f"Unsupported original-scene metric: {metric}")
    metrics = (nc, "dac", "nc_dac") if metric == "nc_dac" else (nc,)
    scores = [score for row in rows for score in row.get("scores", {}).values()
              if score_value(score, metric) is not None]
    expected = sum(max(1, len(row.get("prediction_sources", row.get("scores", {})))) for row in rows)
    return {"version": CONTRACT["version"], "mode": "original", "world": "clean",
            "contract": CONTRACT, "family": "at_fault", "primary_metric": metric,
            "aggregation": ("Mean of per-prediction products; exclude unavailable scores/DAC" if metric == "nc_dac"
                            else f"Mean of per-prediction {nc}; exclude unavailable scores"),
            "overall": {"sample_count": len(rows), "prediction_count": expected,
                        "valid_count": len(scores), "excluded_count": expected - len(scores),
                        **{f"mean_{metric}": mean([score_value(s, metric) for s in scores])
                           for metric in metrics},
                        "coverage": dict(Counter(r.get("status", "unknown") for r in rows)),
                        "errors": dict(Counter(str(v) for r in rows for v in r.get("errors", {}).values()))}}
