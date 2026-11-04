#!/usr/bin/env python3

import argparse
import gzip
import hashlib
import io
import json
import multiprocessing
import os
import pickle
import sys
from collections import deque
from concurrent.futures import (
    Future,
    ProcessPoolExecutor,
    TimeoutError as FutureTimeoutError,
)
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Dict, Iterator, List, Optional, Set, Tuple

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

RECOGDRIVE_ROOT = Path(__file__).resolve().parents[2]
NAVTRAIN_FILTER = (
    RECOGDRIVE_ROOT
    / "navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml"
)
TRAIN_VAL_LOG_SPLIT = (
    RECOGDRIVE_ROOT
    / "navsim/planning/script/config/training/default_train_val_test_log_split.yaml"
)
sys.path.insert(0, str(RECOGDRIVE_ROOT.parents[1]))
from reactbench import paths as release_paths  # noqa: E402

# Read by prepare_mixed_training_dataset.py --original-validation.
DEFAULT_MANIFEST = (release_paths.get("reacttrain_work", release_paths.outputs("reacttrain"))
                    / "recogdrive_agent_cache_dir_train_validation.json")

EXPECTED_FULL_NAVTRAIN_TOKEN_COUNT = 103_288
FEATURE_FILE_NAME = "internvl_feature.gz"
TARGET_FILE_NAME = "trajectory_target.gz"
EXPECTED_FILE_NAMES = {FEATURE_FILE_NAME, TARGET_FILE_NAME}
VALIDATION_CHUNK_SIZE = 16
PENDING_CHUNKS_PER_WORKER = 2
WORKER_STARTUP_TIMEOUT_SECONDS = 300

CacheEntry = Tuple[str, Path, int, int]
ValidationArgument = Tuple[str, str]
ValidationResult = Tuple[str, int, int, str, str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the ReCogDrive navtrain hidden-state feature cache."
    )
    parser.add_argument("--cache-path", required=True, type=Path)
    parser.add_argument("--navsim-log-path", required=True, type=Path)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_navtrain_scenes(
    navsim_log_path: Path, max_scenes: Optional[int]
) -> Dict[str, str]:
    filter_cfg = OmegaConf.load(NAVTRAIN_FILTER)
    filter_cfg.max_scenes = max_scenes
    scene_filter = instantiate(filter_cfg)

    log_files = list(navsim_log_path.iterdir())
    if scene_filter.log_names is not None:
        configured_logs = set(scene_filter.log_names)
        log_files = [
            log_file
            for log_file in log_files
            if log_file.name.replace(".pkl", "") in configured_logs
        ]
    filtered_tokens = (
        set(scene_filter.tokens) if scene_filter.tokens is not None else None
    )

    expected_log_by_token: Dict[str, str] = {}
    for log_pickle_path in log_files:
        with log_pickle_path.open("rb") as file:
            scene_dict_list = pickle.load(file)

        for start_index in range(0, len(scene_dict_list), scene_filter.frame_interval):
            frame_list = scene_dict_list[
                start_index : start_index + scene_filter.num_frames
            ]
            if len(frame_list) < scene_filter.num_frames:
                continue

            anchor_frame = frame_list[scene_filter.num_history_frames - 1]
            if scene_filter.has_route and not anchor_frame["roadblock_ids"]:
                continue

            token = anchor_frame["token"]
            if filtered_tokens is not None and token not in filtered_tokens:
                continue
            if token in expected_log_by_token:
                fail(f"The navtrain scene filter produced duplicate token {token}.")

            expected_log_by_token[token] = str(frame_list[0]["log_name"])
            if (
                scene_filter.max_scenes is not None
                and len(expected_log_by_token) >= scene_filter.max_scenes
            ):
                break

        if (
            scene_filter.max_scenes is not None
            and len(expected_log_by_token) >= scene_filter.max_scenes
        ):
            break

    if max_scenes is None and len(expected_log_by_token) != EXPECTED_FULL_NAVTRAIN_TOKEN_COUNT:
        fail(
            "Unexpected full navtrain cardinality: "
            f"expected={EXPECTED_FULL_NAVTRAIN_TOKEN_COUNT}, "
            f"actual={len(expected_log_by_token)}"
        )
    return expected_log_by_token


def load_train_val_log_sets() -> Tuple[Set[str], Set[str]]:
    split_cfg = OmegaConf.load(TRAIN_VAL_LOG_SPLIT)
    train_logs = [str(log_name) for log_name in split_cfg.train_logs]
    val_logs = [str(log_name) for log_name in split_cfg.val_logs]

    if len(train_logs) != len(set(train_logs)):
        fail(f"Duplicate log names found in {TRAIN_VAL_LOG_SPLIT}'s train_logs.")
    if len(val_logs) != len(set(val_logs)):
        fail(f"Duplicate log names found in {TRAIN_VAL_LOG_SPLIT}'s val_logs.")

    train_log_set = set(train_logs)
    val_log_set = set(val_logs)
    overlap = sorted(train_log_set & val_log_set)
    if overlap:
        fail(f"Train/validation log split overlaps: {overlap[:10]}")
    return train_log_set, val_log_set


def discover_cache(cache_path: Path) -> Tuple[Dict[str, CacheEntry], Set[str]]:
    entries_by_token: Dict[str, CacheEntry] = {}
    actual_log_names: Set[str] = set()

    for log_path in cache_path.iterdir():
        if log_path.is_symlink() or not log_path.is_dir():
            fail(f"Unexpected non-directory entry in cache root: {log_path}")
        log_name = log_path.name
        actual_log_names.add(log_name)

        for token_path in log_path.iterdir():
            if token_path.is_symlink() or not token_path.is_dir():
                fail(f"Unexpected non-directory entry in cache log directory: {token_path}")
            token = token_path.name
            if token in entries_by_token:
                previous_log = entries_by_token[token][0]
                fail(
                    f"Duplicate cache token {token} in logs {previous_log} and {log_name}."
                )

            files_by_name: Dict[str, Path] = {}
            for entry in token_path.iterdir():
                if entry.is_symlink() or not entry.is_file():
                    fail(f"Unexpected non-file entry in cache token directory: {entry}")
                files_by_name[entry.name] = entry

            actual_file_names = set(files_by_name)
            missing_files = sorted(EXPECTED_FILE_NAMES - actual_file_names)
            extra_files = sorted(actual_file_names - EXPECTED_FILE_NAMES)
            if missing_files or extra_files:
                fail(
                    f"Unexpected files for cache token {token}: "
                    f"missing={missing_files}, extra={extra_files}"
                )

            feature_path = files_by_name[FEATURE_FILE_NAME]
            target_path = files_by_name[TARGET_FILE_NAME]
            entries_by_token[token] = (
                log_name,
                token_path,
                feature_path.stat().st_size,
                target_path.stat().st_size,
            )

    return entries_by_token, actual_log_names


def validate_coverage(
    expected_log_by_token: Dict[str, str],
    entries_by_token: Dict[str, CacheEntry],
    actual_log_names: Set[str],
) -> None:
    expected_tokens = set(expected_log_by_token)
    actual_tokens = set(entries_by_token)
    missing_tokens = sorted(expected_tokens - actual_tokens)
    extra_tokens = sorted(actual_tokens - expected_tokens)
    if missing_tokens or extra_tokens:
        fail(
            "Hidden-state cache token coverage mismatch: "
            f"expected={len(expected_tokens)}, actual={len(actual_tokens)}, "
            f"missing={len(missing_tokens)} {missing_tokens[:10]}, "
            f"extra={len(extra_tokens)} {extra_tokens[:10]}"
        )

    misplaced_tokens = sorted(
        token
        for token, expected_log in expected_log_by_token.items()
        if entries_by_token[token][0] != expected_log
    )
    if misplaced_tokens:
        examples = [
            {
                "token": token,
                "expected_log": expected_log_by_token[token],
                "actual_log": entries_by_token[token][0],
            }
            for token in misplaced_tokens[:10]
        ]
        fail(f"Cache tokens stored under the wrong log directory: {examples}")

    expected_log_names = set(expected_log_by_token.values())
    missing_logs = sorted(expected_log_names - actual_log_names)
    extra_logs = sorted(actual_log_names - expected_log_names)
    if missing_logs or extra_logs:
        fail(
            "Hidden-state cache log coverage mismatch: "
            f"expected={len(expected_log_names)}, actual={len(actual_log_names)}, "
            f"missing={len(missing_logs)} {missing_logs[:10]}, "
            f"extra={len(extra_logs)} {extra_logs[:10]}"
        )


def load_gzip_pickle(path: Path) -> Tuple[object, str, int]:
    try:
        compressed_data = path.read_bytes()
        compressed_sha256 = hashlib.sha256(compressed_data).hexdigest()
        with gzip.GzipFile(fileobj=io.BytesIO(compressed_data), mode="rb") as file:
            value = pickle.load(file)
            if file.read(1) != b"":
                fail(f"Unexpected trailing data after the pickle payload in {path}.")
    except Exception as error:
        raise RuntimeError(f"Could not load hidden-state cache file {path}: {error}") from error
    return value, compressed_sha256, len(compressed_data)


def require_tensor(
    data: Dict[str, object],
    key: str,
    path: Path,
    expected_dtype: torch.dtype,
    expected_shape: Optional[Tuple[int, ...]] = None,
) -> torch.Tensor:
    value = data[key]
    if not isinstance(value, torch.Tensor):
        fail(f"Expected tensor {key!r} in {path}, found {type(value).__name__}.")
    if value.device.type != "cpu":
        fail(f"Expected CPU tensor {key!r} in {path}, found device {value.device}.")
    if value.dtype != expected_dtype:
        fail(
            f"Unexpected dtype for {key!r} in {path}: "
            f"expected={expected_dtype}, actual={value.dtype}"
        )
    if expected_shape is not None and tuple(value.shape) != expected_shape:
        fail(
            f"Unexpected shape for {key!r} in {path}: "
            f"expected={expected_shape}, actual={tuple(value.shape)}"
        )
    if not bool(torch.isfinite(value).all().item()):
        fail(f"Non-finite values found in tensor {key!r} in {path}.")
    return value


def validate_cache_pair(argument: ValidationArgument) -> ValidationResult:
    token, token_path_string = argument
    token_path = Path(token_path_string)
    feature_path = token_path / FEATURE_FILE_NAME
    target_path = token_path / TARGET_FILE_NAME

    feature_object, feature_sha256, feature_size = load_gzip_pickle(feature_path)
    target_object, target_sha256, target_size = load_gzip_pickle(target_path)

    if not isinstance(feature_object, dict):
        fail(
            f"Expected a feature dictionary in {feature_path}, "
            f"found {type(feature_object).__name__}."
        )
    expected_feature_keys = {
        "history_trajectory",
        "high_command_one_hot",
        "last_hidden_state",
        "status_feature",
    }
    if set(feature_object) != expected_feature_keys:
        fail(
            f"Unexpected feature keys in {feature_path}: "
            f"expected={sorted(expected_feature_keys)}, actual={sorted(feature_object)}"
        )

    require_tensor(
        feature_object,
        "history_trajectory",
        feature_path,
        torch.float32,
        (4, 3),
    )
    require_tensor(
        feature_object,
        "high_command_one_hot",
        feature_path,
        torch.float32,
        (4,),
    )
    last_hidden_state = require_tensor(
        feature_object,
        "last_hidden_state",
        feature_path,
        torch.float32,
    )
    if last_hidden_state.ndim != 2 or last_hidden_state.shape[0] <= 0:
        fail(
            f"Expected non-empty rank-2 last_hidden_state in {feature_path}, "
            f"found shape {tuple(last_hidden_state.shape)}."
        )
    if last_hidden_state.shape[1] != 1536:
        fail(
            f"Unexpected last_hidden_state width in {feature_path}: "
            f"expected=1536, actual={last_hidden_state.shape[1]}"
        )
    require_tensor(
        feature_object,
        "status_feature",
        feature_path,
        torch.float32,
        (8,),
    )

    if not isinstance(target_object, dict):
        fail(
            f"Expected a target dictionary in {target_path}, "
            f"found {type(target_object).__name__}."
        )
    if set(target_object) != {"trajectory"}:
        fail(
            f"Unexpected target keys in {target_path}: "
            f"expected=['trajectory'], actual={sorted(target_object)}"
        )
    require_tensor(
        target_object,
        "trajectory",
        target_path,
        torch.float64,
        (8, 3),
    )

    return (
        token,
        feature_size,
        target_size,
        feature_sha256,
        target_sha256,
        int(last_hidden_state.shape[0]),
    )


def validate_cache_chunk(
    arguments: List[ValidationArgument],
) -> List[ValidationResult]:
    return [validate_cache_pair(argument) for argument in arguments]


def terminate_processes(executor: ProcessPoolExecutor) -> None:
    processes = getattr(executor, "_processes", None)
    if processes is not None:
        for process in processes.values():
            try:
                process.terminate()
            except ProcessLookupError:
                pass


def validate_cache_pairs(
    arguments: List[ValidationArgument], worker_count: int
) -> Iterator[ValidationResult]:
    spawn_context = multiprocessing.get_context("spawn")
    executor = ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=spawn_context,
    )
    pending_chunks: Deque[Future[List[ValidationResult]]] = deque()
    next_offset = 0
    max_pending_chunks = worker_count * PENDING_CHUNKS_PER_WORKER

    def submit_next_chunk() -> bool:
        nonlocal next_offset
        if next_offset >= len(arguments):
            return False
        chunk = arguments[next_offset : next_offset + VALIDATION_CHUNK_SIZE]
        next_offset += len(chunk)
        pending_chunks.append(executor.submit(validate_cache_chunk, chunk))
        return True

    try:
        for _ in range(max_pending_chunks):
            if not submit_next_chunk():
                break

        first_chunk = True
        while pending_chunks:
            future = pending_chunks.popleft()
            try:
                chunk_results = future.result(
                    timeout=WORKER_STARTUP_TIMEOUT_SECONDS if first_chunk else None
                )
            except FutureTimeoutError as error:
                terminate_processes(executor)
                raise RuntimeError(
                    "Spawned cache-validation workers did not return the first "
                    f"{min(VALIDATION_CHUNK_SIZE, len(arguments))} pairs within "
                    f"{WORKER_STARTUP_TIMEOUT_SECONDS} seconds."
                ) from error

            first_chunk = False
            submit_next_chunk()
            yield from chunk_results
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def split_token_counts(
    expected_log_by_token: Dict[str, str], train_logs: Set[str], val_logs: Set[str]
) -> Tuple[int, int]:
    expected_logs = set(expected_log_by_token.values())
    unassigned_logs = sorted(expected_logs - train_logs - val_logs)
    if unassigned_logs:
        fail(f"Navtrain logs absent from the train/validation split: {unassigned_logs[:10]}")

    train_count = sum(
        expected_log in train_logs for expected_log in expected_log_by_token.values()
    )
    val_count = sum(expected_log in val_logs for expected_log in expected_log_by_token.values())
    if train_count + val_count != len(expected_log_by_token):
        fail(
            "Train/validation token counts do not cover navtrain: "
            f"train={train_count}, validation={val_count}, "
            f"total={len(expected_log_by_token)}"
        )
    return train_count, val_count


def write_manifest(manifest_path: Path, manifest: Dict[str, object]) -> None:
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
    navsim_log_path = args.navsim_log_path.resolve()
    manifest_path = args.manifest_path.resolve()
    if not cache_path.is_dir():
        fail(f"Hidden-state cache path does not exist: {cache_path}")
    if not navsim_log_path.is_dir():
        fail(f"NAVSIM log path does not exist: {navsim_log_path}")
    if manifest_path.suffix != ".json":
        fail(f"--manifest-path must name a JSON file: {manifest_path}")
    try:
        manifest_path.relative_to(cache_path)
    except ValueError:
        pass
    else:
        fail(
            "The validation manifest must be outside the cache root so "
            f"CacheOnlyDataset cannot mistake it for a log directory: {manifest_path}"
        )

    print("Loading the expected navtrain token-to-log mapping...", flush=True)
    expected_log_by_token = expected_navtrain_scenes(navsim_log_path, args.max_scenes)
    print(f"Expected tokens: {len(expected_log_by_token)}", flush=True)

    train_logs, val_logs = load_train_val_log_sets()
    train_token_count, val_token_count = split_token_counts(
        expected_log_by_token, train_logs, val_logs
    )
    print(
        f"Expected split: train={train_token_count}, validation={val_token_count}",
        flush=True,
    )

    print(f"Scanning cache structure under {cache_path}...", flush=True)
    entries_by_token, actual_log_names = discover_cache(cache_path)
    validate_coverage(expected_log_by_token, entries_by_token, actual_log_names)

    ordered_tokens = sorted(expected_log_by_token)
    validation_arguments = [
        (token, str(entries_by_token[token][1])) for token in ordered_tokens
    ]
    worker_count = min(args.workers, len(validation_arguments))
    print(
        f"Loading and validating {len(validation_arguments)} feature/target pairs "
        f"with {worker_count} workers...",
        flush=True,
    )

    total_feature_bytes = 0
    total_target_bytes = 0
    minimum_sequence_length: Optional[int] = None
    maximum_sequence_length: Optional[int] = None
    content_digest = hashlib.sha256()

    results = validate_cache_pairs(validation_arguments, worker_count)
    for index, result in enumerate(results, start=1):
        (
            token,
            feature_size,
            target_size,
            feature_sha256,
            target_sha256,
            sequence_length,
        ) = result
        total_feature_bytes += feature_size
        total_target_bytes += target_size
        minimum_sequence_length = (
            sequence_length
            if minimum_sequence_length is None
            else min(minimum_sequence_length, sequence_length)
        )
        maximum_sequence_length = (
            sequence_length
            if maximum_sequence_length is None
            else max(maximum_sequence_length, sequence_length)
        )
        content_digest.update(
            f"{token}/{FEATURE_FILE_NAME}\0{feature_size}\0{feature_sha256}\n".encode()
        )
        content_digest.update(
            f"{token}/{TARGET_FILE_NAME}\0{target_size}\0{target_sha256}\n".encode()
        )
        if index == min(VALIDATION_CHUNK_SIZE, len(validation_arguments)):
            print("Spawned worker startup check passed.", flush=True)
        if index % 1000 == 0 or index == len(validation_arguments):
            print(
                f"Validated {index} / {len(validation_arguments)} cache pairs",
                flush=True,
            )

    manifest: Dict[str, object] = {
        "schema_version": 1,
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "validator": str(Path(__file__).resolve()),
        "validator_sha256": sha256_file(Path(__file__)),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cache_path": str(cache_path),
        "navsim_log_path": str(navsim_log_path),
        "navtrain_filter": str(NAVTRAIN_FILTER.resolve()),
        "navtrain_filter_sha256": sha256_file(NAVTRAIN_FILTER),
        "train_val_log_split": str(TRAIN_VAL_LOG_SPLIT.resolve()),
        "train_val_log_split_sha256": sha256_file(TRAIN_VAL_LOG_SPLIT),
        "max_scenes": args.max_scenes,
        "workers": worker_count,
        "expected_token_count": len(expected_log_by_token),
        "actual_token_count": len(entries_by_token),
        "train_token_count": train_token_count,
        "validation_token_count": val_token_count,
        "log_directory_count": len(actual_log_names),
        "validated_feature_file_count": len(validation_arguments),
        "validated_target_file_count": len(validation_arguments),
        "feature_file_name": FEATURE_FILE_NAME,
        "target_file_name": TARGET_FILE_NAME,
        "feature_bytes": total_feature_bytes,
        "target_bytes": total_target_bytes,
        "total_bytes": total_feature_bytes + total_target_bytes,
        "combined_compressed_content_sha256": content_digest.hexdigest(),
        "last_hidden_state_sequence_length_min": minimum_sequence_length,
        "last_hidden_state_sequence_length_max": maximum_sequence_length,
        "tensor_schema": {
            FEATURE_FILE_NAME: {
                "history_trajectory": {"dtype": "torch.float32", "shape": [4, 3]},
                "high_command_one_hot": {"dtype": "torch.float32", "shape": [4]},
                "last_hidden_state": {
                    "dtype": "torch.float32",
                    "shape": ["variable_positive", 1536],
                },
                "status_feature": {"dtype": "torch.float32", "shape": [8]},
            },
            TARGET_FILE_NAME: {
                "trajectory": {"dtype": "torch.float64", "shape": [8, 3]}
            },
        },
    }
    write_manifest(manifest_path, manifest)
    print(f"Hidden-state cache validation passed. Manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
