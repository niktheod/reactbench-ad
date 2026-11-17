"""The released ReactNAV runs: packing them (authors) and completing them with the user's navtest images.

A ReactNAV run ``<object>_timestep_<k>`` holds, per navtest token, the twelve model inputs
``final_images/<token>/{CAM_L0,CAM_F0,CAM_R0}/{0,1,2,3}.{png,jpg}`` (frame 3 is the current frame, 0.5 s apart),
the placement of the inserted object in ``metadata/<token>.json``, and ``run_config.json``. Only the frames that
the object changes are released, as JPEGs with navtest's own settings (quality 80, 4:2:0 chroma subsampling); the
compositor writes them as lossless PNGs, which the paper evaluated. The unchanged frames are the original navtest
JPEGs, which ``link`` adds as links into the user's NAVSIM ``sensor_blobs`` (paths from the NAVSIM annotations):

    python -m reactbench.datasets.navsim_annotations data/reactnav/reactnav.yaml   # <outputs>/annotations/reactnav.json
    python -m reactbench.datasets.reactnav link [RUNS_ROOT]                         # default: reactnav in configs/paths.yaml

Authors: ``python -m reactbench.datasets.reactnav pack SOURCE_ROOT OUTPUT_ROOT`` saves the edited frames of
compacted compositor runs as JPEGs, copies their metadata and writes a ``run_config.json`` without machine paths.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from .. import paths
from ..io import read_json, relative, write_json
from ..runs import parse_run_name

CAMERAS = {"CAM_L0": "front_left_camera", "CAM_F0": "front_camera", "CAM_R0": "front_right_camera"}
FRAMES = 4
SCENE_FILTER = "data/reactnav/reactnav.yaml"
SUFFIXES = (".png", ".jpg", ".jpeg")
# The released edited frames: navtest's own JPEG settings (its images use the quality-80 tables and 4:2:0).
JPEG_OPTIONS = {"quality": 80, "subsampling": "4:2:0", "optimize": True}


def released_runs(root):
    """{released run name: run directory} of the insertion runs under ``root`` (old ``_contact_only`` names too)."""
    found = {}
    for run in sorted(Path(root).iterdir()):
        parsed = parse_run_name(run.name, "navsim") if run.is_dir() else None
        if parsed:
            name = f"{parsed[0]}_timestep_{parsed[1]}"
            if name in found:
                raise ValueError(f"Two runs for {name} under {root}")
            found[name] = run
    return found


def pack(source_root, output_root, *, runs=None, max_tokens=None):
    """Save the edited frames of each run as JPEGs and copy its metadata; unchanged frames (links to navtest) are
    left out."""
    output_root = Path(output_root)
    for name, run in released_runs(source_root).items():
        if runs and name not in runs:
            continue
        target = output_root / name
        config = relative(read_json(run / "run_config.json"))
        config.pop("data_root", None)
        config["scene_filter"] = SCENE_FILTER
        tokens = sorted(p.name for p in (run / "final_images").iterdir() if p.is_dir())[:max_tokens]
        if max_tokens is not None:
            config["selected_tokens"] = [t for t in config.get("selected_tokens", tokens) if t in set(tokens)]
        write_json(target / "run_config.json", config)
        edited = 0
        for token in tokens:
            for image in sorted((run / "final_images" / token).rglob("*")):
                if image.is_file() and not image.is_symlink():
                    if image.suffix != ".png":
                        raise ValueError(f"Edited frames of a compacted run are PNGs: {image}")
                    destination = target / "final_images" / image.relative_to(run / "final_images")
                    destination = destination.with_suffix(".jpg")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with Image.open(image) as frame:
                        frame.convert("RGB").save(destination, "JPEG", **JPEG_OPTIONS)
                    edited += 1
            write_json(target / "metadata" / f"{token}.json", relative(read_json(run / "metadata" / f"{token}.json")))
        print(f"{name}: {len(tokens)} tokens, {edited} edited frames", flush=True)


def link(runs_root, annotations_file):
    """Add the unchanged frames of every token as links to the user's navtest images."""
    annotations = {r["token"]: r for r in json.loads(Path(annotations_file).read_text())}
    for name, run in released_runs(runs_root).items():
        added = 0
        for token_dir in sorted(p for p in (run / "final_images").iterdir() if p.is_dir()):
            annotation = annotations.get(token_dir.name)
            if annotation is None:
                raise KeyError(f"{token_dir.name} is not in {annotations_file}")
            for camera, key in CAMERAS.items():
                sources = annotation[key]
                if len(sources) != FRAMES:
                    raise ValueError(f"{token_dir.name}: expected {FRAMES} {key} frames, got {len(sources)}")
                for frame, source in enumerate(sources):
                    stem = token_dir / camera / str(frame)
                    if any(stem.with_suffix(s).exists() for s in SUFFIXES):
                        continue
                    if not Path(source).is_file():
                        raise FileNotFoundError(f"navtest image missing: {source}")
                    stem.parent.mkdir(parents=True, exist_ok=True)
                    stem.with_suffix(Path(source).suffix).symlink_to(Path(source).resolve())
                    added += 1
        print(f"{name}: {added} unchanged frames linked", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("link", help="Add the unchanged navtest frames (users)")
    command.add_argument("runs_root", type=Path, nargs="?", help="Default: reactnav in configs/paths.yaml")
    command.add_argument("--annotations", type=Path, help="Default: <outputs>/annotations/reactnav.json")
    command = commands.add_parser("pack", help="Save the edited frames (JPEG) and metadata for release (authors)")
    command.add_argument("source_root", type=Path)
    command.add_argument("output_root", type=Path)
    command.add_argument("--runs", nargs="+", help="Released run names (default: all insertion runs)")
    command.add_argument("--max-tokens", type=int, help="First N tokens of each run (a check)")
    args = parser.parse_args(argv)
    if args.command == "link":
        annotations = args.annotations or paths.get("annotations", paths.outputs("annotations")) / "reactnav.json"
        link(args.runs_root or paths.get("reactnav"), annotations)
    else:
        pack(args.source_root, args.output_root, runs=args.runs, max_tokens=args.max_tokens)


if __name__ == "__main__":
    main()
