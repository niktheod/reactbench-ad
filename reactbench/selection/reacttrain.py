"""ReactTrain source scenes (paper appendix, ReactTrain).

``data/reacttrain/reacttrain.yaml`` holds the 8,734 navtrain training-log scenes with an ego speed of at least
5 km/h whose hazardous and benign variants of all three objects pass the placement, front-visibility and
benign-clearance checks at every look-ahead time (``classify_samples.py --split navtrain --benign``);
``benign_sides.json`` holds the side of each benign object. The 24 runs draw the same 2,000 of them
(``--max-samples 2000``, seed 42). ``intersect`` keeps the scenes rendered in all 24 runs (1,957, listed in
``data/reacttrain/sources.txt``): it moves the other scenes' images and metadata to ``<root>/excluded/`` and
writes ``<root>/intersection_tokens.txt``, which the ReactTrain caches read.

    python -m reactbench.selection.reacttrain intersect datasets/reacttrain
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

from .. import paths

RUN = re.compile(r"^(benign|hazardous)_(car|standing|walking)_timestep_([1-4])_n[0-9]+$")
SOURCES = paths.REPO_ROOT / "data" / "reacttrain" / "sources.txt"


def successful_tokens(run):
    final = {p.name for p in (run / "final_images").iterdir() if p.is_dir()}
    metadata = {p.stem for p in (run / "metadata").glob("*.json")}
    return final & metadata


def intersect(root):
    runs = sorted(p for p in Path(root).iterdir() if p.is_dir() and RUN.fullmatch(p.name))
    if len(runs) != 24:
        raise ValueError(f"Expected the 24 ReactTrain runs under {root}, found {len(runs)}")
    per_run = {run: successful_tokens(run) for run in runs}
    common = set.intersection(*per_run.values())
    moved = 0
    for run, tokens in per_run.items():
        for token in sorted(tokens - common):
            target = Path(root) / "excluded" / run.name
            for source in (run / "final_images" / token, run / "metadata" / f"{token}.json"):
                if source.exists():
                    (target / source.parent.name).mkdir(parents=True, exist_ok=True)
                    shutil.move(str(source), str(target / source.parent.name / source.name))
            moved += 1
    (Path(root) / "intersection_tokens.txt").write_text("".join(f"{t}\n" for t in sorted(common)))
    return sorted(common), moved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    step = commands.add_parser("intersect", help="Keep the scenes rendered in all 24 runs")
    step.add_argument("root", type=Path, help="Directory with the 24 ReactTrain runs")
    args = parser.parse_args(argv)
    common, moved = intersect(args.root)
    released = SOURCES.read_text().split()
    print(f"{len(common)} scenes rendered in all 24 runs ({moved} run/scene entries moved to excluded/); "
          f"{'identical to' if common == sorted(released) else 'DIFFERENT from'} data/reacttrain/sources.txt")


if __name__ == "__main__":
    main()
