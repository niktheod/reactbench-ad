"""Versioned, resumable CPU-to-GPU handoffs for the editing pipelines.

Only small prepared state is serialized; sensor inputs and render manifests
remain referenced files. No pickle or executable deserialization is used.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import zlib

import numpy as np

SCHEMA = "editing_preparation_v1"
EXECUTION_ARGUMENTS = {
    "stage", "prepare_only", "render_workers", "render_batch_size",
    "finalize_workers", "blender", "depth_python", "depth_model_cache",
}


class PreparationError(RuntimeError):
    pass


def encode(value):
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            if value.dtype != np.dtype(object):
                raise PreparationError("Structured object arrays cannot be checkpointed")
            return {"type": "object_array", "shape": list(value.shape),
                    "items": [encode(item) for item in value.flat]}
        dtype = value.dtype.descr if value.dtype.fields else value.dtype.str
        return {"type": "array", "dtype": encode(dtype),
                "shape": list(value.shape),
                "data": base64.b64encode(value.tobytes()).decode("ascii")}
    if isinstance(value, np.generic):
        return {"type": "scalar", "value": encode(np.asarray(value))}
    if isinstance(value, (bytes, bytearray)):
        return {"type": "bytes", "data": base64.b64encode(value).decode("ascii")}
    if isinstance(value, Path):
        return {"type": "path", "value": str(value)}
    if isinstance(value, dict):
        return {"type": "dict", "items": [[encode(k), encode(v)] for k, v in value.items()]}
    if isinstance(value, tuple):
        return {"type": "tuple", "items": [encode(v) for v in value]}
    if isinstance(value, list):
        return [encode(v) for v in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise PreparationError(f"Unsupported prepared-state type: {type(value).__name__}")


def decode(value):
    if isinstance(value, list):
        return [decode(v) for v in value]
    if not isinstance(value, dict):
        return value
    kind = value["type"]
    if kind == "dict":
        return {decode(k): decode(v) for k, v in value["items"]}
    if kind == "tuple":
        return tuple(decode(v) for v in value["items"])
    if kind == "path":
        return Path(value["value"])
    if kind == "bytes":
        return base64.b64decode(value["data"], validate=True)
    if kind == "scalar":
        return decode(value["value"])[()]
    if kind == "object_array":
        result = np.empty(len(value["items"]), dtype=object)
        for index, item in enumerate(value["items"]):
            result[index] = decode(item)
        return result.reshape(value["shape"])
    if kind == "array":
        dtype = np.dtype(decode(value["dtype"]))
        if dtype.hasobject:
            raise PreparationError("Object arrays cannot be restored")
        return np.frombuffer(base64.b64decode(value["data"], validate=True),
                             dtype=dtype).reshape(value["shape"]).copy()
    raise PreparationError(f"Unknown prepared-state type: {kind}")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_identity(path, *, content=False):
    path = Path(path).resolve()
    stat = path.stat()
    result = {"path": str(path), "size": stat.st_size}
    if content:
        with path.open("rb") as stream:
            hasher = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(block)
        result["sha256"] = hasher.hexdigest()
    else:
        result["mtime_ns"] = stat.st_mtime_ns
    return result


def verify_files(records):
    for record in records:
        try:
            actual = file_identity(record["path"], content="sha256" in record)
        except OSError as exc:
            raise PreparationError(f"Prepared input is unavailable: {record['path']}") from exc
        if actual != record:
            raise PreparationError(f"Prepared input changed: {record['path']}; rerun CPU preparation")


def request_identity(args, code_files):
    """Capture CLI identity before adapters mutate arguments or load scenes."""
    values, inputs = {}, []
    for key, value in sorted(vars(args).items()):
        if key in EXECUTION_ARGUMENTS:
            continue
        if isinstance(value, Path):
            value = value.expanduser().resolve()
            if value.is_file():
                inputs.append(file_identity(value, content=True))
            value = str(value)
        values[key] = value
    asset_dir = getattr(args, "asset_dir", None)
    if asset_dir is not None:
        paths = (Path(asset_dir).rglob("*.blend") if getattr(args, "car", False)
                 else Path(asset_dir).glob("*.blend"))
        for path in sorted(paths):
            inputs.append(file_identity(path))
            for suffix in (".json", ".xoma"):
                sidecar = path.with_suffix(suffix)
                if sidecar.is_file():
                    inputs.append(file_identity(sidecar, content=True))
    return {"arguments": values, "inputs": inputs,
            "code": [file_identity(path, content=True) for path in sorted(set(map(str, code_files)))]}


def write_document(path, document):
    """Publish a complete compressed document without exposing partial writes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".checkpoint-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            with gzip.GzipFile(fileobj=stream, mode="wb", mtime=0) as compressed:
                compressed.write(json.dumps(encode(document), separators=(",", ":")).encode())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_document(path):
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return decode(json.load(stream))
    except (OSError, EOFError, ValueError, KeyError, TypeError, zlib.error) as exc:
        raise PreparationError(f"Invalid/missing preparation checkpoint: {path}; rerun CPU preparation") from exc


class PreparationStore:
    def __init__(self, run_dir, dataset, request):
        self.run_dir = Path(run_dir)
        self.root = self.run_dir / "prepared"
        self.dataset = dataset
        self.request = request
        self.request_hash = digest(request)
        self.started = time.monotonic()

    def token_path(self, token):
        if not isinstance(token, str) or not token or Path(token).name != token or token in {".", ".."}:
            raise PreparationError(f"Invalid token: {token!r}")
        return self.root / f"{token}.json.gz"

    def begin(self, tokens, args, *, extra=None, inputs=()):
        if len(set(tokens)) != len(tokens):
            raise PreparationError("Duplicate selected tokens")
        # Invalidate a previous complete index before any new preparation work.
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "index.json.gz").unlink(missing_ok=True)
        self.header = {
            "schema": SCHEMA, "dataset": self.dataset,
            "request_hash": self.request_hash, "tokens": list(tokens),
            "args": dict(vars(args)), "extra": extra or {},
            "run_config": file_identity(self.run_dir / "run_config.json", content=True),
            "inputs": [file_identity(path) for path in sorted(set(map(str, inputs)))],
        }
        self.signature = digest({"request": self.request_hash,
                                 "run_config": self.header["run_config"],
                                 "inputs": self.header["inputs"]})
        self.header["signature"] = self.signature
        self.outcomes = {}

    def restore(self, token):
        path = self.token_path(token)
        if not path.is_file():
            return None
        document = read_document(path)
        if document.get("schema") != SCHEMA or document.get("signature") != self.signature:
            return None
        verify_files(document["inputs"])
        if document["prepared"]["token"] != token:
            raise PreparationError(f"Checkpoint token mismatch: {path}")
        prepared = document["prepared"]
        if "job_manifests" in document:
            prepared["jobs"] = [json.loads(Path(path).read_text()) for path in document["job_manifests"]]
        return prepared

    def save(self, prepared, inputs):
        token = prepared["token"]
        path = self.token_path(token)
        document = {"schema": SCHEMA, "signature": self.signature,
                    "prepared": dict(prepared),
                    "inputs": [file_identity(p) for p in sorted(set(map(str, inputs)))]}
        if prepared.get("jobs"):
            manifests = {}
            for record in prepared["view_records"]:
                for key in ("manifest_path", "labeled_manifest_path"):
                    if record.get(key):
                        manifest_path = Path(record[key])
                        manifest = json.loads(manifest_path.read_text())
                        manifests[manifest["job_id"]] = str(manifest_path)
            document["job_manifests"] = [manifests[job["job_id"]] for job in prepared["jobs"]]
            document["prepared"]["jobs"] = []
        write_document(path, document)
        self.outcomes[token] = {"status": "prepared", "checkpoint": file_identity(path, content=True)}

    def skip(self, token, status, paths):
        self.outcomes[token] = {"status": status,
                                "inputs": [file_identity(p, content=True) for p in paths]}

    def finish(self):
        if set(self.outcomes) != set(self.header["tokens"]):
            raise PreparationError("Preparation has unaccounted selected tokens")
        verify_files(self.header["inputs"])
        verify_files([self.header["run_config"]])
        # Do not advertise readiness if code, labels, or assets changed during
        # the long CPU stage. Bulk inputs use size/mtime; small inputs use SHA.
        verify_files(self.request.get("inputs", []))
        verify_files(self.request.get("code", []))
        for token, outcome in self.outcomes.items():
            if outcome["status"] == "prepared":
                verify_files([outcome["checkpoint"]])
                if self.restore(token) is None:
                    raise PreparationError(f"Incomplete checkpoint: {token}")
            else:
                verify_files(outcome["inputs"])
        checkpoint_bytes = sum(row.get("checkpoint", {}).get("size", 0) for row in self.outcomes.values())
        document = {**self.header, "outcomes": self.outcomes,
                    "checkpoint_bytes": checkpoint_bytes,
                    "elapsed_seconds": time.monotonic() - self.started}
        write_document(self.root / "index.json.gz", document)
        print(f"CPU_PREPARATION_COMPLETE clips={len(self.outcomes)} checkpoint_bytes={checkpoint_bytes} "
              f"elapsed_seconds={document['elapsed_seconds']:.3f}", flush=True)

    def load(self, args):
        index = read_document(self.root / "index.json.gz")
        if (not isinstance(index, dict) or index.get("schema") != SCHEMA or index.get("dataset") != self.dataset
                or index.get("request_hash") != self.request_hash):
            raise PreparationError("Preparation configuration/code changed; rerun CPU preparation")
        required = {"tokens", "outcomes", "inputs", "run_config", "signature", "args", "checkpoint_bytes"}
        if not required <= index.keys():
            raise PreparationError("Incomplete preparation index")
        if set(index["tokens"]) != set(index["outcomes"]):
            raise PreparationError("Incomplete preparation index")
        verify_files(index["inputs"])
        verify_files([index["run_config"]])
        self.signature = index["signature"]
        prepared = []
        for token in index["tokens"]:
            outcome = index["outcomes"][token]
            if outcome["status"] == "prepared":
                verify_files([outcome["checkpoint"]])
                item = self.restore(token)
                if item is None:
                    raise PreparationError(f"Incompatible checkpoint: {token}")
                prepared.append(item)
            elif outcome["status"] in {"cached_success", "failed"}:
                verify_files(outcome["inputs"])
            else:
                raise PreparationError(f"Unknown preparation outcome for {token}")
        runtime = dict(index["args"])
        runtime.update({key: value for key, value in vars(args).items() if key in EXECUTION_ARGUMENTS})
        print(f"PREPARATION_LOADED clips={len(prepared)} "
              f"checkpoint_bytes={index['checkpoint_bytes']} "
              f"elapsed_seconds={time.monotonic() - self.started:.3f}", flush=True)
        return argparse.Namespace(**runtime), prepared, index


def prepared_input_paths(prepared, data_root, split):
    paths = []
    for record in prepared["view_records"]:
        for key in ("source_path", "manifest_path", "labeled_manifest_path"):
            if record.get(key):
                paths.append(Path(record[key]))
    for frame in prepared["frames"]:
        if frame.get("_paiav_lidar_xyz_path"):
            paths.append(Path(frame["_paiav_lidar_xyz_path"]))
        elif frame.get("lidar_path"):
            paths.append(Path(data_root) / "sensor_blobs" / split / frame["lidar_path"])
    asset = prepared["asset"]
    for key in ("path", "metadata_path"):
        if asset.get(key):
            paths.append(Path(asset[key]))
    if asset.get("preparation", {}).get("path"):
        paths.append(Path(asset["preparation"]["path"]))
    return paths


def refresh_render_jobs(args, prepared_tokens, engine):
    """Recheck every manifest, including passes cached when CPU prep ran."""
    jobs = []
    for prepared in prepared_tokens:
        pending = []
        for record in prepared["view_records"]:
            for labeled in (False, True):
                key = "labeled_manifest_path" if labeled else "manifest_path"
                if not record.get(key):
                    continue
                manifest = json.loads(Path(record[key]).read_text())
                if labeled:
                    output = record["labeled_actor_sun_path"]
                    info = record["labeled_render_info_path"]
                    passes = {"directional_shadow": Path(record["labeled_directional_shadow_path"])}
                else:
                    output, info = record["raw_path"], record["render_info_path"]
                    passes = {name: Path(record[f"{name}_path"])
                              for name in ("actor_depth", "actor_sun", "actor_sky", "ambient_occlusion")}
                if args.overwrite or not engine.cache_matches(
                    Path(output), Path(info), manifest["cache_key"],
                    required_paths=engine.required_render_paths(manifest, passes),
                ):
                    pending.append(manifest)
        prepared["jobs"] = pending
        jobs.extend(pending)
    return jobs
