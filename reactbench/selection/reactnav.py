"""ReactNAV selection (paper section 3.1): the 3,228 navtest scenes in ``data/reactnav/reactnav.yaml``.

1. ``reactbench/compositing/navsim/classify_samples.py --split navtest`` checks, for every navtest scene,
   that a crossing pedestrian (from the left or the right), a standing pedestrian and a car can all be placed
   at every look-ahead time, and writes ``navtest_direction_classifications.json`` (token -> left / right /
   failure; the side a crossing pedestrian comes from).
2. ``candidates`` keeps the placeable scenes with an ego speed of 5-40 km/h at the current frame (no condition
   on the acceleration): 3,253 scenes, in navtest order, plus the crossing side of each.
3. The 12 ReactNAV runs are composited on the candidates (``scripts/composite_navsim.sh --filter ...``).
4. ``finalize`` drops the scenes that failed in any of the runs (25: depth alignment, grounding or
   visibility), leaving the 3,228 scenes of ReactNAV.

    python -m reactbench.selection.reactnav candidates outputs/classification/navsim/navtest_direction_classifications.json
    python -m reactbench.selection.reactnav finalize datasets/candidates

The released ``data/reactnav/{reactnav.yaml,directions.json}`` are the result; compositing them directly
gives the same scenes, since every per-scene choice (asset, crossing side, seeds) depends on the scene only.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path

import yaml

from .. import paths
from ..runs import OBJECTS, parse_run_name

NAVTEST_FILTER = (paths.REPO_ROOT / "third_party/recogdrive/navsim/planning/script/config/common/train_test_split"
                  / "scene_filter/navtest.yaml")
SPEED_RANGE_KMH = (5.0, 40.0)
PLACEABLE = ("left", "right")
DEFAULT_OUTPUT = paths.outputs("selection", "reactnav")


def scan_logs(log_names, logs_root):
    """({token: ego speed in m/s at that frame}, {token: log name}) over every frame of the logs."""
    speeds, logs = {}, {}
    for log_name in log_names:
        with (Path(logs_root) / f"{log_name}.pkl").open("rb") as handle:
            for frame in pickle.load(handle):
                vx, vy = (float(v) for v in frame["ego_dynamic_state"][:2])
                speeds[frame["token"]] = math.hypot(vx, vy)
                logs[frame["token"]] = log_name
    return speeds, logs


def restrict(scene_filter, tokens, token_logs):
    """The scene filter with only ``tokens`` (in its own order) and the logs that keep one."""
    kept = set(tokens)
    tokens = [t for t in scene_filter["tokens"] if t in kept]
    used = {token_logs[t] for t in tokens}
    return {**scene_filter, "log_names": [log for log in scene_filter["log_names"] if log in used], "tokens": tokens}


def candidates(classifications, navtest, speeds, token_logs, speed_range_kmh=SPEED_RANGE_KMH):
    low, high = (v / 3.6 for v in speed_range_kmh)
    tokens = [t for t in navtest["tokens"] if classifications.get(t) in PLACEABLE and low <= speeds[t] < high]
    return restrict(navtest, tokens, token_logs)


def run_failures(runs_root):
    """{run name: failed tokens} of the ReactNAV runs under ``runs_root`` (their failures.jsonl)."""
    failures = {}
    for run in sorted(Path(runs_root).iterdir()):
        parsed = parse_run_name(run.name, "navsim")
        if not parsed or parsed[0] not in OBJECTS or not (run / "failures.jsonl").is_file():
            continue
        failures[run.name] = sorted({json.loads(line)["token"]
                                     for line in (run / "failures.jsonl").read_text().splitlines() if line.strip()})
    return failures


def write(output_dir, scene_filter, directions, name):
    """``<name>.yaml`` and the crossing sides ``<name>_directions.json`` (ReactNAV: ``directions.json``)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"{name}.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(scene_filter, handle, sort_keys=False)
    directions_name = "directions.json" if name == "reactnav" else f"{name}_directions.json"
    (output_dir / directions_name).write_text(json.dumps(directions, indent=2) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    step = commands.add_parser("candidates", help="Placeable navtest scenes at 5-40 km/h")
    step.add_argument("classifications", type=Path, help="navtest_direction_classifications.json")
    step.add_argument("--navtest-filter", type=Path, default=NAVTEST_FILTER)
    step.add_argument("--logs", type=Path, default=None, help="navsim_logs/test (default: configs/paths.yaml)")
    step.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    step = commands.add_parser("finalize", help="Drop the scenes that failed in any of the 12 runs")
    step.add_argument("runs_root", type=Path, help="Directory with the 12 runs composited on the candidates")
    step.add_argument("--candidates", type=Path, default=DEFAULT_OUTPUT / "candidates.yaml")
    step.add_argument("--logs", type=Path, default=None, help="navsim_logs/test (default: configs/paths.yaml)")
    step.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    logs_root = args.logs or paths.navsim_logs("test")

    if args.command == "candidates":
        classifications = json.loads(args.classifications.read_text())
        navtest = yaml.safe_load(args.navtest_filter.read_text())
        speeds, token_logs = scan_logs(navtest["log_names"], logs_root)
        scene_filter = candidates(classifications, navtest, speeds, token_logs)
        write(args.output_dir, scene_filter, {t: classifications[t] for t in scene_filter["tokens"]}, "candidates")
        placeable = sum(v in PLACEABLE for v in classifications.values())
        print(f"{placeable} placeable navtest scenes, {len(scene_filter['tokens'])} at "
              f"{SPEED_RANGE_KMH[0]:g}-{SPEED_RANGE_KMH[1]:g} km/h -> {args.output_dir / 'candidates.yaml'}")
        return
    scene_filter = yaml.safe_load(args.candidates.read_text())
    directions = json.loads(args.candidates.with_name(f"{args.candidates.stem}_directions.json").read_text())
    failures = run_failures(args.runs_root)
    if len(failures) != 12:
        raise ValueError(f"Expected the 12 ReactNAV runs under {args.runs_root}, found {sorted(failures)}")
    failed = set().union(*map(set, failures.values()))
    _, token_logs = scan_logs(scene_filter["log_names"], logs_root)
    final = restrict(scene_filter, [t for t in scene_filter["tokens"] if t not in failed], token_logs)
    write(args.output_dir, final, {t: directions[t] for t in final["tokens"]}, "reactnav")
    (args.output_dir / "failures_union.json").write_text(json.dumps(sorted(failed), indent=2) + "\n")
    print(f"{len(scene_filter['tokens'])} candidates, {len(failed)} failed in some run, "
          f"{len(final['tokens'])} in ReactNAV -> {args.output_dir / 'reactnav.yaml'}")


if __name__ == "__main__":
    main()
