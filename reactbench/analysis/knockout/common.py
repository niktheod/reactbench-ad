"""Token selection, file helpers and the per-token loop of the CoT knockout."""

from __future__ import annotations

import json
import random
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import yaml

from ...io import write_json


def checked_token(token: Any) -> str:
    if not isinstance(token, str) or not token or token in {".", ".."} or Path(token).name != token:
        raise ValueError(f"Invalid token: {token!r}")
    return token


def unique_tokens(values: Any, source: str) -> list[str]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"Expected a non-empty token list in {source}")
    tokens = [checked_token(token) for token in values]
    if len(tokens) != len(set(tokens)):
        raise ValueError(f"Duplicate tokens in {source}")
    return tokens


def read_tokens_file(path: Path) -> list[str]:
    with Path(path).open(encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return unique_tokens(document.get("tokens"), str(path))


def resolve_directory(path: Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise NotADirectoryError(f"{label} does not exist or is not a directory: {resolved}")
    return resolved


def resolve_file(path: Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    resolved = resolve_file(path, "configuration")
    with resolved.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {resolved}")
    return value


def select_tokens(available: Sequence[str], tokens_file: Path) -> list[str]:
    """The ordered tokens of ``tokens_file``; each must be in the input variant."""
    if len(available) != len(set(available)):
        raise ValueError("The input variant has duplicate tokens")
    requested = read_tokens_file(resolve_file(tokens_file, "token list"))
    missing = sorted(set(requested).difference(available))
    if missing:
        raise ValueError(f"Requested tokens are absent from the input variant: {missing}")
    return requested


def successful_original_tokens(prediction_dir: Path) -> list[str]:
    """Tokens of an original-scene evaluation that produced a prediction."""
    manifest_path = Path(prediction_dir) / "evaluation_manifest.json"
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict) or manifest.get("mode") != "original":
        raise ValueError(f"{manifest_path} is not an original-scene evaluation manifest")
    return unique_tokens(manifest.get("successful_tokens"), str(manifest_path))


def variant_tokens(variant_dir: Path) -> list[str]:
    final_images = resolve_directory(Path(variant_dir) / "final_images", "final images")
    return sorted(checked_token(path.name) for path in final_images.iterdir() if path.is_dir())


def choose_tokens(pools: Sequence[Sequence[str]], *, count: int, seed: int) -> tuple[list[str], int]:
    """A seeded random subset of the tokens common to every pool, sorted."""
    if not pools:
        raise ValueError("No token pools were given")
    common = sorted(set.intersection(*(set(pool) for pool in pools)))
    if not 0 < count <= len(common):
        raise ValueError(f"Cannot choose {count} tokens from {len(common)} common tokens")
    return sorted(random.Random(seed).sample(common, count)), len(common)


def write_token_selection(
    output: Path,
    *,
    variant_dir: Path,
    prediction_dirs: Mapping[str, Path],
    count: int,
    seed: int,
    overwrite: bool = False,
) -> dict[str, Any]:
    output = Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} exists; pass --overwrite to replace the selection")
    pools = [successful_original_tokens(path) for path in prediction_dirs.values()]
    pools.append(variant_tokens(variant_dir))
    tokens, pool_size = choose_tokens(pools, count=count, seed=seed)
    document = {"tokens": tokens, "count": count, "seed": seed, "pool_size": pool_size}
    write_json(output, document)
    return document


def single_text(extra: Mapping[str, Any], key: str) -> str:
    """The one string a batch-one prediction stores under ``key``."""
    if key not in extra:
        raise KeyError(f"The prediction has no {key!r} output")
    values = np.asarray(extra[key], dtype=object).reshape(-1)
    if values.size != 1 or not isinstance(values[0], str):
        raise ValueError(f"Expected exactly one {key!r} string, got {values.size} values")
    return str(values[0])


def gpu_info() -> list[dict[str, Any]]:
    import torch

    if not torch.cuda.is_available():
        return []
    return [
        {
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "capability": list(torch.cuda.get_device_capability(index)),
        }
        for index in range(torch.cuda.device_count())
    ]


@dataclass
class SampleContext:
    index: int
    token: str
    stage: str = "prepare_input"
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LoopSummary:
    selected: int
    skipped: int
    successful: int
    failed: int


class FailureStore:
    """``failed_samples.json``: the latest failure of every token that has not succeeded since."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._records: dict[str, dict[str, Any]] = {}
        if self.path.is_file():
            existing = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(existing, list):
                raise ValueError(f"Expected a JSON list in {self.path}")
            for record in existing:
                token = record.get("token") if isinstance(record, dict) else None
                if not isinstance(token, str) or not token:
                    raise ValueError(f"Failure without a valid token in {self.path}")
                self._records[token] = record
        self.write()

    def record(self, record: Mapping[str, Any]) -> None:
        self._records[str(record["token"])] = dict(record)
        self.write()

    def resolve(self, token: str) -> None:
        if self._records.pop(token, None) is not None:
            self.write()

    def write(self) -> None:
        write_json(self.path, list(self._records.values()))


def run_loop(
    *,
    tokens: Sequence[str],
    manager: Any,
    overwrite: bool,
    run_one: Callable[[str, SampleContext], Mapping[str, Any]],
) -> LoopSummary:
    """Run ``run_one`` per token, save through ``manager`` and record failures without stopping."""
    failures = FailureStore(manager.failures_path)
    skipped = successful = failed = 0
    print(f"Running {len(tokens)} token(s) into {manager.output_dir}", flush=True)
    for index, token in enumerate(tokens, start=1):
        if not overwrite and manager.is_complete(token):
            failures.resolve(token)
            skipped += 1
            print(f"Skipped {index}/{len(tokens)} token={token} (complete)", flush=True)
            continue
        context = SampleContext(index=index, token=token)
        try:
            result = run_one(token, context)
            context.stage = "save"
            manager.save(token, result)
            failures.resolve(token)
            successful += 1
            print(f"Completed {index}/{len(tokens)} token={token}", flush=True)
        except Exception as exc:
            failed += 1
            failures.record({
                "index": index, "token": token, "stage": context.stage,
                "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(),
                **context.details,
            })
            print(f"Failed {index}/{len(tokens)} token={token} stage={context.stage}: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    failures.write()
    summary = LoopSummary(selected=len(tokens), skipped=skipped, successful=successful, failed=failed)
    print(f"Finished: selected={summary.selected}, skipped={summary.skipped}, successful={summary.successful}, "
          f"failed={summary.failed}; failures={manager.failures_path}", flush=True)
    return summary
