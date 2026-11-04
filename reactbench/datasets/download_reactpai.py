"""Download the PAI-AV data of the 1,045 ReactPAI clips.

The PAI-AV licence does not allow redistributing edited clips, so ReactPAI is regenerated from the original
data. PAI-AV stores each sensor in per-chunk archives of about 100 clips, but ReactPAI uses only a few clips of
each of its 243 chunks. For every chunk file listed in ``data/reactpai/selection_manifest.json``, this command
therefore copies only the selected clips' members out of the remote ZIP archive (read by HTTP byte range) into
a local ZIP at the same relative path, and downloads small per-chunk files (calibration parquets) whole. These
are the cameras, LiDAR, egomotion and calibration that the compositor and the evaluation read, and the obstacle
labels that the ReactPAI scorer reads, at the dataset revision of the paper: about 470 GB instead of the
~10 TB of the whole archives.

Every copied member is checked against the remote CRC and size, whole files against the remote size and
SHA-256. Files are written atomically, so an interrupted download resumes where it stopped. The dataset is
gated: accept its licence on Hugging Face and log in (``hf auth login``) first.

    python -m reactbench.datasets.download_reactpai [--output-dir DIR] [--workers N] [--dry-run]

The output directory defaults to ``paiav_root`` in configs/paths.yaml.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile, ZipInfo

from .. import paths

MANIFEST = paths.REPO_ROOT / "data" / "reactpai" / "selection_manifest.json"
# Hugging Face rate-limits requests, so each remote ZIP is read with one request for its central
# directory and one per span of selected members.
TAIL_BYTES = (1 << 20, 32 << 20)         # central-directory read sizes to try
LOCAL_HEADER_MARGIN = 30 + 65_535        # upper bound of a local header beyond its name
SPAN_GAP, MAX_SPAN = 1 << 20, 64 << 20  # merge nearby members, up to this span size
_local = threading.local()


@dataclass(frozen=True)
class Task:
    path: str                        # repository path, also the local path below the output directory
    members: tuple[str, ...] | None  # ZIP members to copy; None = download the whole file


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def hf():
    fs = getattr(_local, "fs", None)
    if fs is None:
        from huggingface_hub import HfFileSystem
        fs = _local.fs = HfFileSystem(skip_instance_cache=True)
    return fs


def retry(function, attempts=6):
    for attempt in range(1, attempts + 1):
        try:
            return function()
        except (KeyError, FileNotFoundError):
            raise
        except Exception:  # network or integrity failure: retry with backoff
            if attempt == attempts:
                raise
            time.sleep(min(120, 2 ** attempt) + random.uniform(0, 1))


class SpanReader(io.RawIOBase):
    """Seekable view of a remote file that fetches one planned byte span per request."""

    def __init__(self, handle, size, spans):
        self.handle, self.size, self.spans = handle, size, spans
        self.position, self.cached = 0, (0, 0, b"")

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}[whence]
        self.position = base + offset
        return self.position

    def _load(self):
        for start, end in self.spans:
            if start <= self.position < end:
                self.handle.seek(start)
                data = self.handle.read(end - start)
                if len(data) != end - start:
                    raise OSError(f"Short read of bytes {start}-{end}")
                self.cached = (start, end, data)
                return
        raise LookupError(f"Read outside the planned spans at byte {self.position}")

    def read(self, size=-1):
        size = self.size - self.position if size is None or size < 0 else size
        parts = []
        while size > 0 and self.position < self.size:
            start, end, data = self.cached
            if not start <= self.position < end:
                self._load()
                start, end, data = self.cached
            count = min(size, end - self.position)
            parts.append(data[self.position - start:self.position - start + count])
            self.position += count
            size -= count
        return b"".join(parts)

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)


def member_spans(infos, size):
    """Byte spans covering each member's local header and data, merged when close."""
    spans = []
    for info in sorted(infos, key=lambda info: info.header_offset):
        start = info.header_offset
        end = min(size, start + LOCAL_HEADER_MARGIN + len(info.filename.encode()) + info.compress_size)
        if spans and start <= spans[-1][1] + SPAN_GAP and end - spans[-1][0] <= MAX_SPAN:
            spans[-1] = (spans[-1][0], max(spans[-1][1], end))
        else:
            spans.append((start, end))
    return spans


def open_remote_zip(handle):
    for tail in TAIL_BYTES:
        reader = SpanReader(handle, handle.size, [(max(0, handle.size - tail), handle.size)])
        try:
            return ZipFile(reader), reader
        except LookupError:
            continue  # the central directory is larger than the tail read
    raise OSError(f"Central directory of {handle.path} exceeds {TAIL_BYTES[-1]} bytes")


def copy_members(remote, task, partial):
    """Copy the selected members into a local stored ZIP, keeping names, order and timestamps."""
    with hf().open(remote, "rb", cache_type="none") as handle:
        source, reader = open_remote_zip(handle)
        with source:
            infos = [source.getinfo(name) for name in task.members]
            reader.spans = reader.spans + member_spans(infos, handle.size)
            with ZipFile(partial, "w", compression=ZIP_STORED, allowZip64=True) as target:
                for info in infos:
                    entry = ZipInfo(info.filename, date_time=info.date_time)
                    entry.compress_type = ZIP_STORED
                    entry.external_attr = info.external_attr
                    with source.open(info) as fin, target.open(entry, "w", force_zip64=True) as fout:
                        shutil.copyfileobj(fin, fout, 8 << 20)
                    written = target.getinfo(info.filename)
                    if (written.CRC, written.file_size) != (info.CRC, info.file_size):
                        raise OSError(f"CRC/size mismatch for {info.filename} of {task.path}")


def copy_file(remote, partial):
    info = hf().info(remote)
    expected = (info.get("lfs") or {}).get("sha256")
    with hf().open(remote, "rb", cache_type="none") as reader:
        data = reader.read()
    if len(data) != info["size"] or (expected and hashlib.sha256(data).hexdigest() != expected):
        raise OSError(f"Size/SHA-256 mismatch for {remote}")
    partial.write_bytes(data)


def complete(local, task):
    """A previous run already wrote this file (files are only ever written atomically)."""
    if not local.is_file():
        return False
    if task.members is None:
        return True
    try:
        with ZipFile(local) as archive:
            return tuple(archive.namelist()) == task.members
    except OSError:
        return False


def fetch(task, root, repo):
    local = root / task.path
    if complete(local, task):
        return "present"
    local.parent.mkdir(parents=True, exist_ok=True)
    partial = local.with_name(local.name + ".partial")
    try:
        if task.members is None:
            retry(lambda: copy_file(f"{repo}/{task.path}", partial))
        else:
            retry(lambda: copy_members(f"{repo}/{task.path}", task, partial))
        os.replace(partial, local)
    finally:
        partial.unlink(missing_ok=True)
    return "downloaded"


def download_catalogs(manifest, root):
    """The catalog files (whole), checked against the manifest's SHA-256."""
    from huggingface_hub import hf_hub_download
    dataset = manifest["dataset"]
    for name, expected in dataset["catalog_inputs"].items():
        path = root / name
        if not (path.is_file() and file_sha256(path) == expected["sha256"]):
            hf_hub_download(dataset["repo_id"], name, repo_type="dataset", revision=dataset["revision"],
                            local_dir=str(root))
        if file_sha256(path) != expected["sha256"]:
            raise OSError(f"Catalog {name} differs from the manifest")


def tasks(manifest, root):
    """One task per chunk file of the manifest: ZIP archives with the selected clips' members, or whole files."""
    import pandas as pd
    patterns = pd.read_csv(root / "features.csv").set_index("feature")["clip_files_in_zip"]
    chunk_of = pd.read_parquet(root / "clip_index.parquet")["chunk"]
    clips = {}
    for clip in manifest["samples"]["clip_ids"]:
        clips.setdefault(int(chunk_of.loc[clip]), []).append(clip)
    result = []
    for chunk in manifest["samples"]["chunks"]:
        selected = sorted(clips[int(chunk["chunk_id"])])
        for feature, path in chunk["feature_files"].items():
            members = None
            if path.endswith(".zip"):
                templates = json.loads(patterns.loc[feature]).values()
                members = tuple(t.format(clip_id=clip) for clip in selected for t in templates)
            result.append(Task(path, members))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true", help="List the files to download and stop")
    args = parser.parse_args(argv)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    manifest = json.loads(args.manifest.read_text())
    dataset = manifest["dataset"]
    root = Path(args.output_dir or paths.get("paiav_root")).expanduser().resolve()
    download_catalogs(manifest, root)
    work = tasks(manifest, root)
    print(f"{len(work)} files of {dataset['repo_id']}@{dataset['revision'][:7]} for "
          f"{manifest['samples']['count']} clips in {manifest['samples']['chunk_count']} chunks -> {root}", flush=True)
    if args.dry_run:
        print("\n".join(task.path for task in work))
        return 0
    repo = f"datasets/{dataset['repo_id']}@{dataset['revision']}"
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch, task, root, repo): task for task in work}
        for done, future in enumerate(as_completed(futures), 1):
            task = futures[future]
            try:
                status = future.result()
            except Exception as exc:
                failures.append(task.path)
                print(f"[{done}/{len(work)}] FAILED {task.path}: {type(exc).__name__}: {exc}", flush=True)
                continue
            if status == "downloaded" or done % 100 == 0:
                print(f"[{done}/{len(work)}] {status} {task.path}", flush=True)
    print(f"{len(work) - len(failures)} of {len(work)} files complete"
          + (f"; rerun to retry {len(failures)} failures" if failures else ""))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
