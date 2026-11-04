"""Look up the classifier's chosen benign side for each token and actor mode."""

import argparse
import json
from pathlib import Path

DEFAULT_SIDES = (Path(__file__).resolve().parent.parent / "classification_results"
                 / "navtrain_benign_front25" / "navtrain_benign_sides.json")


def add_arguments(parser):
    parser.add_argument("--benign-sides", type=Path, default=DEFAULT_SIDES,
                        help="Classifier's token-to-side JSON; defaults to the full navtrain result.")


def configure(args, fast):
    if not getattr(args, "benign", False) or args.benign_side != "auto":
        return
    path = args.benign_sides.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing classified benign sides: {path}; supply --benign-sides FILE")
    args.benign_sides = path
    args.benign_side_lookup = json.loads(path.read_text())
    args.benign_selection = {"source": "classified_sides", "path": str(path),
                             "sha256": fast.canonical_hash(args.benign_side_lookup)}


def token_arguments(args, fast, token):
    if not getattr(args, "benign", False) or getattr(args, "benign_side", "auto") != "auto":
        return args
    mode = "car" if args.car else "standing" if args.standing else "crossing"
    side = getattr(args, "benign_side_lookup", {}).get(token, {}).get(mode)
    if side not in ("left", "right"):
        raise fast.TokenFailure("missing_classified_side",
                                f"Missing or invalid classified benign {mode} side for {token}")
    result = argparse.Namespace(**vars(args))
    result.benign_side = side
    result.benign_selection_source = "classified_sides"
    return result
