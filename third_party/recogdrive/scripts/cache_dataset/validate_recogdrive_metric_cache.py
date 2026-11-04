#!/usr/bin/env python3

import argparse
import csv
import hashlib
import json
import lzma
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from hydra.utils import instantiate
from omegaconf import OmegaConf

from navsim.common.dataloader import MetricCacheLoader, SceneLoader
from navsim.common.dataclasses import SensorConfig
from navsim.planning.metric_caching.metric_cache import MetricCache


RECOGDRIVE_ROOT = Path(__file__).resolve().parents[2]
NAVTRAIN_FILTER = (
    RECOGDRIVE_ROOT
    / "navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a ReCogDrive navtrain metric cache.")
    parser.add_argument("--cache-path", required=True, type=Path)
    parser.add_argument("--navsim-log-path", required=True, type=Path)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--manifest-path", type=Path, default=None)
    return parser.parse_args()


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_navtrain_tokens(navsim_log_path: Path, max_scenes: Optional[int]) -> List[str]:
    filter_cfg = OmegaConf.load(NAVTRAIN_FILTER)
    filter_cfg.max_scenes = max_scenes
    scene_loader = SceneLoader(
        sensor_blobs_path=None,
        data_path=navsim_log_path,
        scene_filter=instantiate(filter_cfg),
        sensor_config=SensorConfig.build_no_sensors(),
    )
    tokens = list(scene_loader.tokens)
    if len(tokens) != len(set(tokens)):
        fail("The navtrain SceneLoader produced duplicate tokens.")
    return tokens


def load_metadata_paths(metadata_file: Path) -> List[Path]:
    with metadata_file.open(newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != ["file_name"]:
            fail(f"Unexpected metadata columns in {metadata_file}: {reader.fieldnames}")
        paths = [Path(row["file_name"]) for row in reader]
    if not paths:
        fail(f"Metadata file contains no cache paths: {metadata_file}")
    return paths


def validate_cache_file(path_string: str) -> Tuple[str, str]:
    path = Path(path_string)
    try:
        with lzma.open(path, "rb") as file:
            cache = pickle.load(file)
    except Exception as error:
        raise RuntimeError(f"Could not load metric cache {path}: {error}") from error

    if not isinstance(cache, MetricCache):
        raise RuntimeError(f"Unexpected object type in {path}: {type(cache).__name__}")

    embedded_path = Path(cache.file_path)
    if not embedded_path.is_absolute():
        raise RuntimeError(f"MetricCache.file_path is not absolute in {path}: {embedded_path}")
    if embedded_path.resolve() != path.resolve():
        raise RuntimeError(
            f"MetricCache.file_path does not match its metadata path: {embedded_path} != {path}"
        )
    return path.parent.name, type(cache).__name__


def validate_structure(cache_path: Path, metadata_paths: List[Path]) -> Dict[str, Path]:
    cache_root = cache_path.resolve()
    paths_by_token: Dict[str, Path] = {}
    seen_paths = set()

    for path in metadata_paths:
        if not path.is_absolute():
            fail(f"Metadata path is not absolute: {path}")
        resolved_path = path.resolve()
        try:
            resolved_path.relative_to(cache_root)
        except ValueError:
            fail(f"Metadata path escapes the cache root {cache_root}: {path}")
        if path.name != "metric_cache.pkl":
            fail(f"Unexpected metric-cache filename: {path}")
        if not path.is_file():
            fail(f"Metadata path does not exist: {path}")
        if resolved_path in seen_paths:
            fail(f"Duplicate metadata path: {path}")

        token = path.parent.name
        if token in paths_by_token:
            fail(f"Duplicate token in metric-cache metadata: {token}")
        seen_paths.add(resolved_path)
        paths_by_token[token] = path

    return paths_by_token


def validate_coverage(expected_tokens: List[str], paths_by_token: Dict[str, Path]) -> None:
    expected = set(expected_tokens)
    actual = set(paths_by_token)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        fail(
            "Metric-cache token coverage mismatch: "
            f"expected={len(expected)}, actual={len(actual)}, "
            f"missing={len(missing)} {missing[:10]}, extra={len(extra)} {extra[:10]}"
        )


def write_manifest(
    manifest_path: Path,
    cache_path: Path,
    metadata_file: Path,
    expected_count: int,
    workers: int,
    max_scenes: Optional[int],
) -> None:
    manifest = {
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "cache_path": str(cache_path.resolve()),
        "metadata_file": str(metadata_file.resolve()),
        "metadata_sha256": sha256_file(metadata_file),
        "expected_token_count": expected_count,
        "metadata_row_count": expected_count,
        "loadable_cache_count": expected_count,
        "max_scenes": max_scenes,
        "workers": workers,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_path, manifest_path)


def main() -> None:
    args = parse_args()
    if not args.cache_path.is_absolute():
        fail(f"--cache-path must be absolute: {args.cache_path}")
    if not args.navsim_log_path.is_absolute():
        fail(f"--navsim-log-path must be absolute: {args.navsim_log_path}")
    if args.max_scenes is not None and args.max_scenes <= 0:
        fail("--max-scenes must be a positive integer.")
    if args.workers <= 0:
        fail("--workers must be a positive integer.")

    cache_path = args.cache_path.resolve()
    metadata_dir = cache_path / "metadata"
    metadata_files = sorted(metadata_dir.glob("*.csv")) if metadata_dir.is_dir() else []
    if len(metadata_files) != 1:
        fail(f"Expected exactly one metadata CSV in {metadata_dir}, found {len(metadata_files)}.")
    metadata_file = metadata_files[0]

    print("Loading the expected navtrain token set...", flush=True)
    expected_tokens = expected_navtrain_tokens(args.navsim_log_path, args.max_scenes)
    print(f"Expected tokens: {len(expected_tokens)}", flush=True)

    metadata_paths = load_metadata_paths(metadata_file)
    paths_by_token = validate_structure(cache_path, metadata_paths)
    validate_coverage(expected_tokens, paths_by_token)

    ordered_paths = [str(paths_by_token[token]) for token in expected_tokens]
    worker_count = min(args.workers, len(ordered_paths))
    print(f"Loading {len(ordered_paths)} metric-cache files with {worker_count} workers...", flush=True)
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        for index, _ in enumerate(
            executor.map(validate_cache_file, ordered_paths, chunksize=32), start=1
        ):
            if index % 1000 == 0 or index == len(ordered_paths):
                print(f"Validated {index} / {len(ordered_paths)} cache files", flush=True)

    metric_cache_loader = MetricCacheLoader(cache_path)
    if set(metric_cache_loader.tokens) != set(expected_tokens):
        fail("MetricCacheLoader token coverage differs from the navtrain SceneLoader token set.")

    manifest_path = args.manifest_path or metadata_dir / "validation_manifest.json"
    write_manifest(
        manifest_path=manifest_path,
        cache_path=cache_path,
        metadata_file=metadata_file,
        expected_count=len(expected_tokens),
        workers=worker_count,
        max_scenes=args.max_scenes,
    )
    print(f"Metric cache validation passed. Manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
