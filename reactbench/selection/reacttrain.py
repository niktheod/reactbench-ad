"""ReactTrain source scenes (paper appendix, ReactTrain).

``data/reacttrain/reacttrain.yaml`` holds the 8,734 navtrain training-log scenes with an ego speed of at least
5 km/h at the current frame whose hazardous and benign insertions of all three objects pass the placement,
front-visibility and benign-clearance checks at every look-ahead time; ``directions.json`` holds the crossing
side and ``benign_sides.json`` the side of each benign object. The 24 variants draw the same 2,000 scenes of them
(``--max-samples 2000``, seed 42). ``intersect`` keeps the scenes rendered in all 24 variants (1,957, listed in
``data/reacttrain/sources.txt``): it moves the other scenes' images and metadata to ``<root>/excluded/`` and writes
``<root>/intersection_tokens.txt``, which the ReactTrain caches read.

    python -m reactbench.selection.reacttrain intersect datasets/reacttrain
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

from .. import paths

VARIANT = re.compile(r"^(benign|hazardous)_(car|standing|walking)_timestep_([1-4])_n[0-9]+$")
SOURCES = paths.REPO_ROOT / "data" / "reacttrain" / "sources.txt"
def successful_tokens(variant):
    final = {p.name for p in (variant / "final_images").iterdir() if p.is_dir()}
    metadata = {p.stem for p in (variant / "metadata").glob("*.json")}
    return final & metadata


def intersect(root):
    variants = sorted(p for p in Path(root).iterdir() if p.is_dir() and VARIANT.fullmatch(p.name))
    if len(variants) != 24:
        raise ValueError(f"Expected the 24 ReactTrain variants under {root}, found {len(variants)}")
    per_variant = {variant: successful_tokens(variant) for variant in variants}
    common = set.intersection(*per_variant.values())
    moved = 0
    for variant, tokens in per_variant.items():
        for token in sorted(tokens - common):
            target = Path(root) / "excluded" / variant.name
            for source in (variant / "final_images" / token, variant / "metadata" / f"{token}.json"):
                if source.exists():
                    (target / source.parent.name).mkdir(parents=True, exist_ok=True)
                    shutil.move(str(source), str(target / source.parent.name / source.name))
            moved += 1
    (Path(root) / "intersection_tokens.txt").write_text("".join(f"{t}\n" for t in sorted(common)))
    return sorted(common), moved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    step = commands.add_parser("intersect", help="Keep the scenes rendered in all 24 variants")
    step.add_argument("root", type=Path, help="Directory with the 24 ReactTrain variants")
    args = parser.parse_args(argv)
    common, moved = intersect(args.root)
    released = SOURCES.read_text().split()
    print(f"{len(common)} scenes rendered in all 24 variants ({moved} variant/scene entries moved to excluded/); "
          f"{'identical to' if common == sorted(released) else 'DIFFERENT from'} data/reacttrain/sources.txt")


if __name__ == "__main__":
    main()
