"""Numbers in the text of the paper's ReactPAI analysis (section 4.2), from the Alpamayo CoTs and plans.

- Inserted cars: the share of edited-scene CoTs at tau >= 1.5 s that call the car stopped while the
  original-scene CoT of the clip does not (keyword classes below; "stopped traffic" excluded).
- Standing pedestrians at tau = 2.0 s: the share of edited-scene CoTs that describe the pedestrian as static,
  and, over those clips, the mean of the original minus the edited plan's speed over the final 0.5 s of the
  4 s horizon (3.5-4.0 s).

Pedestrian labels are assigned in priority order crossing, standing, walking, blocking; standing (standing,
waiting) and blocking (blocking, in the lane) count as static. These are keyword matches on the CoT text.

    python -m reactbench.analysis.cot.text_numbers
"""
from __future__ import annotations

import argparse
import re

import numpy as np

from ...io import write_json
from ...runs import run_name
from ...summarize.runs import Runs
from .summary import MODELS, PEDESTRIAN, read_cot

STOPPED = re.compile(r"\bstopped\b(?! traffic)|stationary|\bparked\b|standstill|not moving|standing still|immobile", re.I)
LABELS = (("crossing", re.compile(r"\bcross\w*", re.I)), ("standing", re.compile(r"\bstand\w*|\bwait\w*", re.I)),
          ("walking", re.compile(r"\bwalk\w*", re.I)),
          ("blocking", re.compile(r"\bblock\w*|\bin (?:the |our |my )?(?:ego )?lane\b", re.I)))
STATIC = ("standing", "blocking")
WINDOW_S = (3.5, 4.0)


def pedestrian_label(text):
    if not PEDESTRIAN.search(text):
        return "no pedestrian"
    return next((label for label, pattern in LABELS if pattern.search(text)), "other")


def window_speed(path, window=WINDOW_S):
    """Mean speed (m/s) of a saved plan over the window, from its poses at the window's ends."""
    with np.load(path, allow_pickle=False) as archive:
        poses, dt = np.asarray(archive["trajectory"], dtype=float), float(archive["interval_length_s"])
    start, end = (int(round(t / dt)) - 1 for t in window)  # pose i is at (i + 1) * dt
    return float(np.linalg.norm(poses[end, :2] - poses[start, :2]) / (window[1] - window[0]))


def common_tokens(runs, model, names):
    root = runs.directory(model, "reactpai")
    return sorted(set.intersection(*({str(r["token"]) for r in runs.rows(root / name, "reactpai")} for name in names)))


def car_stopped(runs, model, taus=(1.5, 2.0)):
    root = runs.directory(model, "reactpai")
    names = [run_name("car", tau, "pai-av") for tau in taus]
    tokens = common_tokens(runs, model, names)
    original = {t: read_cot(root / "original" / "reasoning" / f"{t}.json") for t in tokens}
    gained = n = 0
    for name in names:
        for t in tokens:
            edited = read_cot(root / name / "reasoning" / f"{t}.json")
            gained += bool(STOPPED.search(edited)) and not STOPPED.search(original[t])
            n += 1
    return {"model": model, "taus_s": list(taus), "cots": n, "gained_stopped_pct": 100. * gained / n}


def standing_static(runs, model, tau=2.0):
    root = runs.directory(model, "reactpai")
    name = run_name("standing_pedestrian", tau, "pai-av")
    tokens = common_tokens(runs, model, [name])
    labels = {t: pedestrian_label(read_cot(root / name / "reasoning" / f"{t}.json")) for t in tokens}
    static = [t for t in tokens if labels[t] in STATIC]
    reduction = [window_speed(root / "original" / "trajectories" / f"{t}.npz")
                 - window_speed(root / name / "trajectories" / f"{t}.npz") for t in static]
    return {"model": model, "tau_s": tau, "clips": len(tokens), "static_pct": 100. * len(static) / len(tokens),
            "static_speed_reduction_mps": float(np.mean(reduction)),
            "static_speed_reduction_kmh": 3.6 * float(np.mean(reduction))}


def main(argv=None):
    from ... import paths
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", nargs="*", default=[], metavar="MODEL:DATASET=DIR")
    args = parser.parse_args(argv)
    runs = Runs.from_arguments(args.runs)
    records = {"car_stopped": [car_stopped(runs, m) for m in MODELS],
               "standing_static": [standing_static(runs, m) for m in MODELS]}
    write_json(paths.outputs("summaries", "cot_text_numbers.json"), records)
    for r in records["car_stopped"]:
        print(f"{r['model']}: inserted car newly called stopped in {r['gained_stopped_pct']:.1f}% of CoTs (tau >= 1.5 s)")
    for r in records["standing_static"]:
        print(f"{r['model']}: standing pedestrian called static in {r['static_pct']:.1f}% of CoTs at tau = 2 s; "
              f"those plans are {r['static_speed_reduction_kmh']:.1f} km/h slower over 3.5-4 s")


if __name__ == "__main__":
    main()
