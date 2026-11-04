from pathlib import Path
import json
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "reactbench" / "compositing" / "navsim"))
import batch_composite_navsim_blender as fast
import benign_sides


def arguments(tmp_path, *extra):
    return fast.parse_args([
        "--scene-filter", str(tmp_path / "filter.yaml"), "--asset-dir", str(tmp_path),
        "--split", "trainval", "--timestep", "1", "--run-name", "test",
        "--contact-only", "--benign-sides", str(tmp_path / "sides.json"), *extra,
    ])


def test_saved_sides_and_cache_identity(tmp_path):
    path = tmp_path / "sides.json"
    path.write_text(json.dumps({"a": {"car": "right", "standing": "left", "crossing": "left"}}))
    for flags, side in ((["--car"], "right"), (["--standing"], "left"), ([], "left")):
        args = arguments(tmp_path, "--benign", *flags)
        benign_sides.configure(args, fast)
        selected = benign_sides.token_arguments(args, fast, "a")
        assert selected.benign_side == side
        assert selected.seed == args.seed == 42
        assert args.benign_side == "auto"
        assert args.min_front_visible_fraction == .25
    previous = fast.placement_request(args)
    path.write_text(json.dumps({"a": {"crossing": "right"}}))
    benign_sides.configure(args, fast)
    assert fast.placement_request(args) != previous


def test_missing_classification_fails_without_search(tmp_path):
    args = arguments(tmp_path, "--benign")
    with pytest.raises(FileNotFoundError, match="--benign-sides"):
        benign_sides.configure(args, fast)
    args.benign_sides.write_text('{"a": {"crossing": "invalid"}}')
    benign_sides.configure(args, fast)
    for token in ("a", "missing"):
        with pytest.raises(fast.TokenFailure, match="classified benign crossing side"):
            benign_sides.token_arguments(args, fast, token)


def test_explicit_side_and_hazardous_do_not_need_side_file(tmp_path):
    for flags in ([], ["--benign", "--benign-side", "left"]):
        args = arguments(tmp_path, *flags, "--min-front-visible-fraction", "0")
        benign_sides.configure(args, fast)
        assert benign_sides.token_arguments(args, fast, "missing") is args
        assert args.min_front_visible_fraction == 0
