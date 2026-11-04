#!/usr/bin/env python3
"""Preview or acquire only the selected, revision-pinned obstacle label archives."""
from __future__ import annotations

import argparse
import concurrent.futures
import os
from pathlib import Path
import sys
import zipfile

from .. import paths
from ..io import file_digest, read_json, write_json
from ..scoring.paiav_world import resource_plan


def acquire(plan, resource, root):
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import get_hf_file_metadata, get_token, hf_hub_download, hf_hub_url
    url = hf_hub_url(plan["repo_id"], resource["relative_path"], repo_type="dataset", revision=plan["revision"])
    metadata = get_hf_file_metadata(url, token=get_token())
    if metadata.commit_hash != plan["revision"]:
        raise ValueError("Hub resolved a different dataset revision")
    path = Path(hf_hub_download(plan["repo_id"], resource["relative_path"], repo_type="dataset",
                              revision=plan["revision"], local_dir=str(root)))
    checksum = file_digest(path)
    if path.stat().st_size != metadata.size or checksum != metadata.etag:
        raise ValueError(f"Downloaded LFS archive checksum/size mismatch: {path}")
    with zipfile.ZipFile(path) as archive:
        for member in resource["members"].values():
            archive.read(member)  # Checks CRC as well as presence; never extracts files.
    receipt = {"repo_id": plan["repo_id"], "revision": plan["revision"],
               "relative_path": resource["relative_path"], "sha256": checksum,
               "size_bytes": metadata.size, "verified_members": list(resource["members"].values())}
    write_json(path.with_suffix(path.suffix + ".provenance.json"), receipt)
    print(f"Verified obstacle chunk {resource['chunk_id']}: {metadata.size:,} bytes", flush=True)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=paths.get("reactpai_manifest", "data/reactpai/selection_manifest.json"))
    parser.add_argument("--data-root", type=Path, default=paths.get("paiav_root", required=False))
    parser.add_argument("--obstacle-root", type=Path,
                        default=paths.get("paiav_obstacles", paths.outputs("work", "paiav", "resources")))
    parser.add_argument("--tokens-file", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--download", action="store_true", help="Without this flag, print a read-only resource preview")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    tokens = read_json(args.tokens_file) if args.tokens_file else None
    if isinstance(tokens, dict):
        tokens = tokens["tokens"]
    plan = resource_plan(args.manifest, args.data_root, args.obstacle_root, tokens)
    print(f"{plan['token_count']} clips; {len(plan['resources'])} chunks; revision {plan['revision']}", flush=True)
    for resource in plan["resources"]:
        print(f"{resource['relative_path']}: {len(resource['members'])} selected members; local={Path(resource['path']).is_file()}", flush=True)
    if args.download:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(acquire, plan, resource, args.obstacle_root) for resource in plan["resources"]]
            failures = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    failures.append(str(exc))
                    print(f"Download failed: {exc}", file=sys.stderr, flush=True)
        return int(bool(failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
