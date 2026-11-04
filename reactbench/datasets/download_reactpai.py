"""Download the PAI-AV files that the 1,045 ReactPAI clips are read from.

The PAI-AV licence does not allow redistributing edited clips, so ReactPAI is regenerated from the original
data: this fetches the catalog files and, for each of the 243 chunks that hold a ReactPAI clip, the chunk
archives of the 11 features the compositor and the evaluation read, at the dataset revision of the paper.
The dataset is gated: accept its licence on Hugging Face and log in (``hf auth login``) first.

    python -m reactbench.datasets.download_reactpai [--output-dir DIR] [--dry-run]

The output directory defaults to ``paiav_root`` in configs/paths.yaml.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .. import paths

MANIFEST = paths.REPO_ROOT / "data" / "reactpai" / "selection_manifest.json"


def repository_files(manifest):
    """The catalog files and every feature archive of the selected chunks, as repository paths."""
    files = set(manifest["dataset"]["catalog_inputs"])
    for chunk in manifest["samples"]["chunks"]:
        files.update(chunk["feature_files"].values())
    return sorted(files)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true", help="List the files without downloading")
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    files = repository_files(manifest)
    dataset = manifest["dataset"]
    print(f"{len(files)} files of {dataset['repo_id']}@{dataset['revision'][:7]} for "
          f"{manifest['samples']['count']} clips in {manifest['samples']['chunk_count']} chunks")
    if args.dry_run:
        print("\n".join(files))
        return
    from huggingface_hub import snapshot_download
    output = args.output_dir or paths.get("paiav_root")
    snapshot_download(repo_id=dataset["repo_id"], repo_type="dataset", revision=dataset["revision"],
                      allow_patterns=files, local_dir=str(output), max_workers=args.workers)
    print(f"Downloaded to {output}")


if __name__ == "__main__":
    main()
