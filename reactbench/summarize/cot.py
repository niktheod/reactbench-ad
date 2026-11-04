"""CoT analysis on ReactPAI at tau = 2.0 s: CoT swap (Fig. CoT swap) and fixed CoTs (appendix).

CoT swap: NC of the four (image, CoT) combinations OO, OE, EO, EE. Fixed CoTs: NC change when the
generated CoT of the edited scene is replaced by each fixed CoT. 95% scene-bootstrap intervals.
"""
from __future__ import annotations

from ..analysis.cot.cot_swap import CROSSED_EDITED_IMAGE, CROSSED_ORIGINAL_IMAGE, NATURAL_EDITED, NATURAL_ORIGINAL
from ..analysis.cot.fixed_cot import CONDITIONS as FIXED_CONDITIONS
from ..runs import OBJECTS, run_name
from .benchmark import _indexed, own_pairs
from .runs import MODELS
from .stats import bootstrap_mean

TAU_S = 2.0
SWAP_CONDITIONS = {"OO": NATURAL_ORIGINAL, "OE": CROSSED_ORIGINAL_IMAGE,
                   "EO": CROSSED_EDITED_IMAGE, "EE": NATURAL_EDITED}


def _run(runs, model, obj, suffix):
    root = runs.directory(model, "reactpai")
    for name in (run_name(obj, TAU_S, "pai-av"),):
        path = root / f"{name}_{suffix}"
        if path.is_dir():
            return path
    return None


def swap_table(runs, models=None):
    records = []
    for model in models or MODELS["reactpai"]:
        for obj in OBJECTS:
            run = _run(runs, model, obj, "cot_swap")
            if run is None:
                continue
            indexed = _indexed(runs.rows(run, "reactpai"), "reactpai")
            for code, condition in SWAP_CONDITIONS.items():
                keys = sorted(k for k, r in indexed.items()
                              if isinstance((r["scores"].get(condition) or {}).get("all"), bool))
                values = [100. * (1 - int(indexed[k]["scores"][condition]["all"])) for k in keys]
                records.append({"model": model, "object": obj, "tau_s": TAU_S, "condition": code, "n": len(keys),
                                "nc_pct": sum(values) / len(values) if values else None,
                                "ci95_pct": bootstrap_mean(values)})
    return records


def fixed_table(runs, models=None):
    records = []
    for model in models or MODELS["reactpai"]:
        for obj in OBJECTS:
            run = _run(runs, model, obj, "fixed_cots")
            if run is None:
                continue
            rows = runs.rows(run, "reactpai")
            for condition in FIXED_CONDITIONS:
                pairs = own_pairs(rows, "reactpai", before="natural_edited", after=condition)
                n = len(pairs["keys"])
                records.append({"model": model, "object": obj, "tau_s": TAU_S, "condition": condition, "n": n,
                                "delta_nc_pp": sum(pairs["delta_pp"]) / n if n else None,
                                "ci95_pp": bootstrap_mean(pairs["delta_pp"])})
    return records
