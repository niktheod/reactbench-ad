"""Shared inventory and resumable I/O for the ReactTrain edited samples (no model imports at startup).

Locations come from the release's configs/paths.yaml: ``reacttrain`` (the compositor runs on navtrain
scenes) and ``reacttrain_work`` (caches, targets and training views; default outputs/reacttrain).
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from functools import lru_cache
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import re
import sys
import tempfile

TRAIN_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = TRAIN_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(TRAIN_ROOT))
from reactbench import paths as release_paths  # noqa: E402

# Caches, targets and views of the ReactTrain fine-tuning (the NAVSIM experiment root of the original runs).
EXP = release_paths.get("reacttrain_work", release_paths.outputs("reacttrain"))
EDITED_ROOT = release_paths.get("reacttrain", "datasets/reacttrain")
OUTPUT_ROOT = EXP / "recogdrive_edited_training/navtrain_selected_t34_v1"


def navsim_default(function):
    """A NAVSIM data location from configs/paths.yaml, or None when it is not configured."""
    try:
        return function()
    except KeyError:
        return None
SPLIT = TRAIN_ROOT / "navsim/planning/script/config/training/default_train_val_test_log_split.yaml"
VARIANT = re.compile(r"^(benign|hazardous)_(car|standing|walking)_timestep_([1-4])_n[0-9]+$")
# Cross-node arithmetic may differ in the last bits. File identities, input
# geometry metadata, configuration and discrete fields remain exact.
DERIVED_FLOAT_ATOL = 1e-12
DERIVED_MOTION_FIELDS = {
    "heading_rad", "velocity_xy_mps", "length_m", "width_m",
    "front_length_m", "rear_length_m", "left_width_m", "right_width_m",
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def file_identity(path, *, content=False):
    path = Path(path).absolute()
    stat = path.stat()
    if not path.is_file() or not stat.st_size:
        raise ValueError(f"Missing/empty input: {path}")
    result = {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if content:
        result["sha256"] = sha256(path)
        if path.stat().st_mtime_ns != stat.st_mtime_ns:
            raise ValueError(f"Input changed while hashing: {path}")
    return result


def verify_file(reference):
    if file_identity(reference["path"], content="sha256" in reference) != reference:
        raise ValueError(f"Frozen input changed: {reference['path']}")
    return Path(reference["path"])


@contextmanager
def locked(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, staging = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            writer(handle)
        os.replace(staging, path)
    finally:
        if os.path.exists(staging):
            os.unlink(staging)


def write_json(path, value):
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    atomic_write(path, lambda handle: handle.write(data))


def read_json(path):
    return json.loads(Path(path).read_text())


def save_features(path, features):
    def write(handle):
        with gzip.GzipFile(fileobj=handle, mode="wb", compresslevel=1, mtime=0) as compressed:
            pickle.dump(features, compressed, protocol=pickle.HIGHEST_PROTOCOL)
    atomic_write(path, write)


def load_features(path):
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)


def argument_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--edited-root", type=Path, default=EDITED_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--timesteps", type=int, nargs="+", choices=(1, 2, 3, 4), default=[3, 4])
    parser.add_argument("--max-scenes", type=int, help="Smoke cohort: first N source tokens, all selected variants")
    parser.add_argument("--log-path", type=Path, default=navsim_default(lambda: release_paths.navsim_logs("trainval")))
    parser.add_argument("--feature-cache", type=Path, default=EXP / "recogdrive_agent_cache_dir_train")
    parser.add_argument("--train-log-split", type=Path, default=SPLIT)
    parser.add_argument("--dry-run", action="store_true", help="Validate/discover inputs and print counts without writing or loading models/maps")
    return parser


def validate_arguments(args):
    args.timesteps = sorted(set(args.timesteps))
    if args.log_path is None:
        raise ValueError("Set navsim_data (or navsim_logs_trainval) in configs/paths.yaml, or pass --log-path")
    for name in ("edited_root", "output", "log_path", "feature_cache", "train_log_split"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.max_scenes is not None and args.max_scenes < 1:
        raise ValueError("--max-scenes must be positive")
    if (args.max_scenes is not None or args.timesteps != [3, 4]) and args.output == OUTPUT_ROOT:
        raise ValueError("A smoke cohort or different timesteps require a separate --output")
    for path in (args.edited_root, args.log_path, args.feature_cache):
        if args.output == path or args.output.is_relative_to(path) or path.is_relative_to(args.output):
            raise ValueError(f"Output must be separate from source data/caches: {path}")


@lru_cache(maxsize=1)
def source_frames(path, mtime_ns=None):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def discover(args):
    """Freeze selected variants and audit membership in the actual training log split."""
    import yaml
    from pyquaternion import Quaternion
    from reactbench.scoring.actor_motion import actor_motion_from_metadata, current_front_record as _current_front_record

    validate_arguments(args)
    token_file = args.edited_root / "intersection_tokens.txt"
    tokens = token_file.read_text().splitlines()
    if not tokens or len(set(tokens)) != len(tokens) or any(not re.fullmatch(r"[0-9a-f]{16}", t) for t in tokens):
        raise ValueError("Intersection must contain unique 16-character source tokens")
    all_tokens = set(tokens)
    tokens = sorted(tokens)[:args.max_scenes]
    split = yaml.safe_load(args.train_log_split.read_text())
    train_logs, val_logs = set(split["train_logs"]), set(split["val_logs"])
    if train_logs & val_logs:
        raise ValueError("Training and validation logs overlap")
    variants = {}
    for directory in sorted(args.edited_root.iterdir()):
        match = VARIANT.fullmatch(directory.name)
        if not directory.is_dir() or match is None or int(match[3]) not in args.timesteps:
            continue
        kind, actor, timestep = match[1], match[2], int(match[3])
        name = f"{kind}_{actor}_t{timestep}"
        if name in variants:
            raise ValueError(f"Duplicate variant: {name}")
        run_path = directory / "run_config.json"
        run = read_json(run_path)
        if run.get("split") != "trainval" or run.get("future_timestep") != timestep or bool(run.get("benign", False)) != (kind == "benign"):
            raise ValueError(f"Run configuration disagrees with variant: {directory}")
        final_tokens = {p.name for p in (directory / "final_images").iterdir() if p.is_dir()}
        metadata_tokens = {p.stem for p in (directory / "metadata").glob("*.json")}
        if final_tokens != all_tokens or metadata_tokens != all_tokens:
            raise ValueError(f"Variant does not match the full intersection: {directory}")
        variants[name] = {"directory": str(directory), "run_config": file_identity(run_path, content=True),
                          "kind": kind, "actor": actor, "timestep": timestep}
    expected = {f"{k}_{a}_t{t}" for k in ("benign", "hazardous")
                for a in ("car", "standing", "walking") for t in args.timesteps}
    if set(variants) != expected:
        raise ValueError(f"Missing selected variants: {sorted(expected-set(variants))}")
    print(f"Checking {len(tokens)} source scenes and {len(variants)} variants...", flush=True)

    # Metadata provides the log hint; verify it against the source log and split.
    first = Path(next(iter(variants.values()))["directory"])
    log_tokens = {}
    for token in tokens:
        metadata = read_json(first / "metadata" / f"{token}.json")
        log = Path(_current_front_record(metadata)["source_path"]).parents[1].name
        if log not in train_logs or log in val_logs:
            raise ValueError(f"Source {token} is outside the training partition: {log}")
        log_tokens.setdefault(log, []).append(token)

    sources, samples = {}, []
    for log_index, (log, selected) in enumerate(sorted(log_tokens.items()), 1):
        log_path = args.log_path / f"{log}.pkl"
        reference = file_identity(log_path)
        frames = source_frames(str(log_path), reference["mtime_ns"])
        indices = {}
        selected_set = set(selected)
        for index, frame in enumerate(frames):
            if frame["token"] in selected_set:
                if frame["token"] in indices:
                    raise ValueError(f"Duplicate source token in log: {frame['token']}")
                indices[frame["token"]] = index
        if set(indices) != selected_set:
            raise ValueError(f"Source tokens missing from {log}")
        for token in sorted(selected, key=indices.get):
            index = indices[token]
            frame = frames[index]
            if index < 3 or index + 10 >= len(frames) or frame["log_name"] != log or not frame["roadblock_ids"]:
                raise ValueError(f"Incomplete history/five-second future/route for {token}")
            original = args.feature_cache / log / token
            heading = float(Quaternion(*frame["ego2global_rotation"]).yaw_pitch_roll[0])
            sources[token] = {"source": reference, "log_name": log, "frame_index": index,
                              "scene_token": frame["scene_token"], "ego_heading_rad": heading,
                              "features": file_identity(original / "internvl_feature.gz"),
                              "targets": file_identity(original / "trajectory_target.gz", content=True)}
            for variant, details in sorted(variants.items()):
                directory = Path(details["directory"])
                metadata_path = directory / "metadata" / f"{token}.json"
                metadata_reference = file_identity(metadata_path, content=True)
                metadata = read_json(metadata_path)
                if (metadata.get("status") != "success" or metadata.get("token") != token
                        or metadata.get("future_timestep") != details["timestep"]
                        or bool(metadata.get("placement", {}).get("benign", False)) != (details["kind"] == "benign")
                        or metadata.get("actor_type", "pedestrian") != ("car" if details["actor"] == "car" else "pedestrian")):
                    raise ValueError(f"Metadata disagrees with variant: {metadata_path}")
                current = _current_front_record(metadata)
                if current["frame_index"] != 3 or current["timestamp_us"] != frame["timestamp"]:
                    raise ValueError(f"Edited image/source timestamp mismatch: {metadata_path}")
                images = [p for p in (directory / "final_images" / token / "CAM_F0").glob("3.*")
                          if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
                if len(images) != 1:
                    raise ValueError(f"Expected one current front image: {variant}/{token}")
                motion = actor_motion_from_metadata(metadata, ego_heading_rad=heading)
                moving = math.hypot(*motion["velocity_xy_mps"]) > 0
                if moving != (details["actor"] == "walking"):
                    raise ValueError(f"Actor motion disagrees with variant category: {metadata_path}")
                sample_id = f"{token}__{variant}"
                samples.append({"sample_id": sample_id, "source_token": token, "variant": variant,
                    # A read-only inventory need not stream tens of GB of pixels.
                    # Real preparation freezes content hashes before either job writes caches.
                    "image": file_identity(images[0], content=not args.dry_run), "metadata": metadata_reference,
                    "actor_motion": motion, "features": f"features/{log}/{sample_id}/internvl_feature.gz",
                    "targets": f"features/{log}/{sample_id}/trajectory_target.gz",
                    "metric_cache": f"metrics/{log}/unknown/{sample_id}/metric_cache.pkl"})
        verify_file(reference)
        if log_index % 25 == 0:
            print(f"Inventory: {len(sources)}/{len(tokens)} source scenes checked", flush=True)
    value = {"version": 1, "output": str(args.output), "edited_root": str(args.edited_root),
             "timesteps": args.timesteps, "max_scenes": args.max_scenes,
             "intersection": file_identity(token_file, content=True),
             "train_log_split": file_identity(args.train_log_split, content=True),
             "variants": variants, "sources": sources, "samples": samples,
             "original_target_role": "Recorded original trajectory for the stage-3 loader; not an edited expert label"}
    return {**value, "identity": digest(value)}


def manifest_difference(saved, current, path=()):
    """Return the first incompatible field; tolerate only derived roundoff."""
    if isinstance(saved, dict) and isinstance(current, dict):
        for key in sorted(saved.keys() | current.keys()):
            # The two payload hashes are checked separately before comparison.
            if not path and key == "identity":
                continue
            if key not in saved or key not in current:
                return "/".join(map(str, (*path, key)))
            difference = manifest_difference(saved[key], current[key], (*path, key))
            if difference is not None:
                return difference
        return None
    if isinstance(saved, list) and isinstance(current, list) and len(saved) == len(current):
        for index, (old, new) in enumerate(zip(saved, current)):
            difference = manifest_difference(old, new, (*path, index))
            if difference is not None:
                return difference
        return None
    derived = ((len(path) == 3 and path[0] == "sources" and path[2] == "ego_heading_rad")
               or (len(path) in (4, 5) and path[0] == "samples"
                   and path[2] == "actor_motion" and path[3] in DERIVED_MOTION_FIELDS))
    if derived and isinstance(saved, float) and isinstance(current, float):
        if (math.isfinite(saved) and math.isfinite(current)
                and math.isclose(saved, current, rel_tol=0, abs_tol=DERIVED_FLOAT_ATOL)):
            return None
    elif type(saved) is type(current) and saved == current:
        return None
    return "/".join(map(str, path)) or "<root>"


def freeze_manifest(args, manifest):
    """Freeze or reuse the authoritative manifest, including its exact identity."""
    print(f"Selected {len(manifest['sources'])} source scenes x {len(manifest['variants'])} variants = "
          f"{len(manifest['samples'])} samples; output: {args.output}", flush=True)
    if args.dry_run:
        return manifest
    if manifest.get("identity") != digest({k: v for k, v in manifest.items() if k != "identity"}):
        raise ValueError("Current manifest identity does not match its contents")
    with locked(args.output / ".manifest.lock"):
        path = args.output / "manifest.json"
        if path.exists():
            saved = read_json(path)
            if saved.get("identity") != digest({k: v for k, v in saved.items() if k != "identity"}):
                raise ValueError("Frozen manifest identity does not match its contents")
            difference = manifest_difference(saved, manifest)
            if difference is not None:
                raise ValueError(f"Frozen manifest differs from inputs at {difference}; use a new --output")
            # Both jobs must use the frozen values/hash, not their local roundoff.
            return saved
        else:
            if any(p.name != ".manifest.lock" for p in args.output.iterdir()):
                raise FileExistsError("Output is nonempty without a manifest; use a dedicated --output")
            write_json(path, manifest)
            return manifest


def code_identity(paths):
    return {str(p.relative_to(REPO_ROOT)): sha256(p) for p in paths}


def ensure_contract(output, kind, contract):
    path = output / f"{kind}_contract.json"
    if path.exists() and read_json(path) != contract:
        raise ValueError(f"{kind} cache contract changed; use a new --output")
    if not path.exists():
        write_json(path, contract)
    return digest(contract)


def record_path(output, kind, sample):
    return output / "preparation" / kind / f"{sample['sample_id']}.json"


def completed_record(output, kind, sample, identity):
    path = record_path(output, kind, sample)
    if not path.is_file():
        return None
    try:
        record = read_json(path)
        if record.get("identity") != identity or record.get("error") or not record.get("files"):
            return None
        for reference in record["files"]:
            verify_file(reference)
        return record
    except (OSError, ValueError):
        return None


def record_result(output, kind, sample, identity, paths=(), error=None, **extra):
    record = {"sample_id": sample["sample_id"], "identity": identity,
              "files": [file_identity(p) for p in paths], **extra}
    if error is not None:
        record["error"] = f"{type(error).__name__}: {error}"
    write_json(record_path(output, kind, sample), record)
    return record


def scene_from_source(source, *, future_frames):
    from navsim.common.dataclasses import Scene, SensorConfig
    frames = source_frames(str(verify_file(source["source"])), source["source"]["mtime_ns"])
    current = source["frame_index"]
    return Scene.from_scene_dict_list(frames[current-3:current+future_frames+1], None, 4, future_frames,
                                     SensorConfig.build_no_sensors())


def verify_sample(sample, source):
    for name in ("image", "metadata"):
        verify_file(sample[name])
    verify_file(source["source"])


def finish(output, kind, manifest, contract, records):
    failed = sum(bool(r.get("error")) for r in records)
    expected = len(manifest["samples"])
    summary = {"manifest_identity": manifest["identity"], "contract": contract,
               "samples": len(records), "expected_samples": expected, "failures": failed,
               "complete": failed == 0 and len(records) == expected}
    write_json(output / f"{kind}_status.json", summary)
    print(f"{kind}: {len(records)-failed}/{len(records)} complete; {failed} failures", flush=True)
    if failed:
        raise RuntimeError(f"{failed} {kind} samples failed; inspect preparation/{kind}/ and rerun to retry")
    if len(records) != expected:
        raise RuntimeError(f"Incomplete {kind} preparation: {len(records)}/{expected} records")
