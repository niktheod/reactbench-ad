"""Effect of the inserted objects on the Alpamayo CoTs on ReactPAI (paper figure "CoT summary").

Per model and object, pooled over the four look-ahead times, on the clips scored in all 12 runs of the model:
the distribution of the CoT's leading action (slow/stop = stop, yield or slow down; keep distance; other) for
the edited scenes and for the original scenes, the share of edited-scene CoTs that differ from the
original-scene CoT of the same clip, and the share that names the object (a pedestrian or, for cars, a
vehicle), next to the same share for the original-scene CoTs.

    python -m reactbench.analysis.cot.summary
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from ...io import write_json
from ...runs import OBJECTS, run_name
from ...summarize.runs import Runs

MODELS = ("alpamayo1_5", "alpamayo2")
TAUS_S = (0.5, 1.0, 1.5, 2.0)

# Leading action of the CoT: the first pattern that matches its start (order matters).
ADVERB = r"(?:(?:gently|slightly|smoothly|firmly|gradually|cautiously|carefully) )?"
LEADING = [
    (r"(?:stop|halt|wait|come to (?:a )?(?:full |complete )?stop|"
     r"(?:brake|slow(?: down)?|decelerate) (?:to|and come to) (?:a )?(?:full |complete )?stop)\b", "Stop"),
    (r"yield\b", "Yield"),
    (r"(?:slow|decelerat\w*|brake|braking|reduce (?:the )?speed|ease off)\b", "Slow down"),
    (r"(?:(?:keep|maintain) (?:a |the )?(?:safe )?(?:following )?distance|follow)\b", "Keep distance"),
    (r"(?:adapt|adjust) speed\b.*\bdistance\b", "Keep distance"),
    (r"(?:nudge|bypass|narrow\w*|lane change|change (?:to |one )?(?:the )?(?:\w+ ){0,2}lanes?|merge|split|pass|overtake|"
     r"steer|swerve|go around|move (?:left|right|around)|shift)\b", "Steer around"),
    (r"(?:turn|accelerate|keep|maintain|proceed|continue)\b.*\bto (?:pass|bypass|go around|get around|overtake|avoid)\b",
     "Steer around"),
    # Lead-relative speed regulation / explicit following without an explicit slow or stop.
    (r"(?:adapt|adjust|maintain|keep|match) (?:the |a |our )?(?:\w+ )?speed\b.*\b(?:lead|leading|vehicle ahead|car ahead)\b",
     "Keep distance"),
    (r".*\b(?:follow(?:ing)? the (?:lead|leading)|following distance)\b", "Keep distance"),
    (r"(?:keep|maintain|stay|accelerate|resume|proceed|continue|go|drive|turn|adapt|adjust|"
     r"creep|start|follow the|enter|exit|make|take|cruise|travel|move)\b", "Proceed"),
]
PEDESTRIAN = re.compile(r"\b(?:pedestrians?|persons?|people|jaywalk\w*|child(?:ren)?|kids?|man|woman|walkers?|vrus?|human)\b")
VEHICLE = re.compile(r"\b(?:vehicles?|cars?|trucks?|vans?|suvs?|sedans?|bus|buses|pickups?|trailers?|hatchbacks?|lead)\b")
GROUPS = (("slow/stop", {"Stop", "Yield", "Slow down"}), ("keep distance", {"Keep distance"}))


def read_cot(path):
    value = json.loads(Path(path).read_text()).get("cot", "")
    while isinstance(value, list):
        value = value[0] if value else ""
    return value.strip()


def norm(text):
    return " ".join(re.findall(r"[a-z0-9']+", text.lower()))


def action(text):
    text = re.sub("^" + ADVERB, "", norm(text))
    for pattern, label in LEADING:
        if re.match(pattern, text):
            return label
    return "Other"


def names(text, obj):
    return bool((VEHICLE if obj == "car" else PEDESTRIAN).search(text.lower()))


def shares(actions, n):
    """Percent of CoTs per leading-action group; 'other' takes the rest."""
    values = {label: 100. * sum(actions.get(k, 0) for k in keys) / n for label, keys in GROUPS}
    return {**values, "other": 100. - sum(values.values())}


def model_texts(runs, model):
    """{run: [first CoT per clip]} for the original scenes and the 12 runs, on the clips common to all runs."""
    root = runs.directory(model, "reactpai")
    names_ = [run_name(obj, tau, "pai-av") for obj in OBJECTS for tau in TAUS_S]
    cohorts = [{str(r["token"]) for r in runs.rows(root / name, "reactpai")} for name in names_]
    common = sorted(set.intersection(*cohorts))
    return {name: [read_cot(root / name / "reasoning" / f"{token}.json") for token in common]
            for name in ["original", *names_]}, len(common)


def summary(runs, models=MODELS):
    records = []
    for model in models:
        texts, n_clips = model_texts(runs, model)
        original = texts["original"]
        for obj in OBJECTS:
            edited = [t for tau in TAUS_S for t in texts[run_name(obj, tau, "pai-av")]]
            originals = original * len(TAUS_S)
            records.append({
                "model": model, "object": obj, "clips": n_clips, "edited_cots": len(edited),
                "edited_leading_action_pct": shares(Counter(map(action, edited)), len(edited)),
                "original_leading_action_pct": shares(Counter(map(action, original)), len(original)),
                "changed_pct": 100. * sum(norm(o) != norm(e) for o, e in zip(originals, edited)) / len(edited),
                "names_object_pct": 100. * sum(names(e, obj) for e in edited) / len(edited),
                "original_names_object_pct": 100. * sum(names(o, obj) for o in original) / len(original)})
    return records


def main(argv=None):
    from ... import paths
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", nargs="*", default=[], metavar="MODEL:DATASET=DIR")
    args = parser.parse_args(argv)
    records = summary(Runs.from_arguments(args.runs))
    write_json(paths.outputs("summaries", "cot_summary.json"), records)
    for r in records:
        e, o = r["edited_leading_action_pct"], r["original_leading_action_pct"]
        print(f"{r['model']:12s} {r['object']:20s} slow/stop {e['slow/stop']:5.1f} (orig {o['slow/stop']:5.1f})  "
              f"keep distance {e['keep distance']:5.1f} (orig {o['keep distance']:5.1f})  changed {r['changed_pct']:5.1f}  "
              f"names {r['names_object_pct']:5.1f} ({r['original_names_object_pct']:.1f})")


if __name__ == "__main__":
    main()
