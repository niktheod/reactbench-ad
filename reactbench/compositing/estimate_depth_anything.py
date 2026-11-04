#!/usr/bin/env python3
"""Batch relative-depth inference (Depth Anything V2) for the NAVSIM and PAI-AV compositors.

This script runs in a separate, modern PyTorch environment; the compositor's own
environment handles calibration and metric LiDAR alignment. Input is JSONL with
``input_path`` and ``output_path`` fields.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as functional
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
DEFAULT_CACHE_VERSION = "fullres_float32_v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def npz_scalar(data, key: str):
    if key not in data.files:
        return None
    value = np.asarray(data[key])
    if value.size != 1:
        return None
    return str(value.reshape(()).item())


def cache_matches(
    output: Path,
    source: Path,
    source_sha256: str,
    cache_version: str,
    model_id: str,
) -> bool:
    if not output.is_file():
        return False
    try:
        with np.load(output, allow_pickle=False) as data:
            if "relative_depth" not in data.files:
                return False
            if npz_scalar(data, "model_id") != model_id:
                return False
            cached_sha256 = npz_scalar(data, "source_sha256")
            if cached_sha256 is not None:
                return (
                    cached_sha256 == source_sha256
                    and npz_scalar(data, "cache_version") == cache_version
                )
            # Preserve compatibility with run-local predictions made before
            # persistent cache metadata was introduced.
            cached_source = npz_scalar(data, "source_path")
            return (
                cached_source is not None
                and Path(cached_source).resolve() == source
                and output.stat().st_mtime_ns >= source.stat().st_mtime_ns
            )
    except (OSError, TypeError, ValueError):
        return False


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    jobs = [json.loads(line) for line in args.jobs.read_text().splitlines() if line.strip()]
    if not jobs:
        print("DEPTH_BATCH_SUMMARY=" + json.dumps({"completed": 0, "cached": 0}))
        return

    processor = AutoImageProcessor.from_pretrained(
        args.model, local_files_only=args.local_files_only, use_fast=False
    )
    model = AutoModelForDepthEstimation.from_pretrained(
        args.model, local_files_only=args.local_files_only
    )
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    completed = 0
    cached = 0
    with torch.inference_mode():
        for job in jobs:
            source = Path(job["input_path"]).resolve()
            output = Path(job["output_path"]).resolve()
            source_sha256 = job.get("source_sha256") or file_sha256(source)
            cache_version = job.get("cache_version", DEFAULT_CACHE_VERSION)
            output.parent.mkdir(parents=True, exist_ok=True)
            lock_path = output.with_suffix(output.suffix + ".lock")
            with lock_path.open("a+b") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                if cache_matches(
                    output,
                    source,
                    source_sha256,
                    cache_version,
                    args.model,
                ):
                    cached += 1
                    continue
                with Image.open(source) as source_image:
                    image = source_image.convert("RGB")
                width, height = image.size
                inputs = processor(images=image, return_tensors="pt")
                inputs = {key: value.to(device) for key, value in inputs.items()}
                prediction = model(**inputs).predicted_depth.unsqueeze(1)
                prediction = functional.interpolate(
                    prediction,
                    size=(height, width),
                    mode="bicubic",
                    align_corners=False,
                )[0, 0]
                relative_depth = prediction.float().cpu().numpy().astype(np.float32)
                temporary = output.with_name(
                    f"{output.name}.tmp.{os.getpid()}.npz"
                )
                try:
                    np.savez_compressed(
                        temporary,
                        relative_depth=relative_depth,
                        model_id=np.asarray(args.model),
                        source_path=np.asarray(str(source)),
                        source_sha256=np.asarray(source_sha256),
                        cache_version=np.asarray(cache_version),
                    )
                    os.replace(temporary, output)
                finally:
                    if temporary.exists():
                        temporary.unlink()
            completed += 1
            print(f"DEPTH_JOB_DONE={source.name}", flush=True)

    print(
        "DEPTH_BATCH_SUMMARY="
        + json.dumps({"completed": completed, "cached": cached, "device": str(device)})
    )


if __name__ == "__main__":
    main()
