#!/usr/bin/env python3
"""Prepare/reuse frozen GLS masks on CPUs; --check exits 3 if preparation is needed."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import tempfile

from . import ego_masks
from .. import locations
from .paiav_io import PAIAVSceneLoader, load_selection_manifest

DEFAULT_DATA_ROOT = locations.PAIAV_ROOT
_WORKER_LOADER = None


def initialize(args, tokens):
    global _WORKER_LOADER
    _WORKER_LOADER = PAIAVSceneLoader(args.data_root, load_selection_manifest(args.manifest),
                                     tokens, Path(tempfile.gettempdir()),
                                     anchor_time_s=args.anchor_time_s, future_timestep=1,
                                     ego_mask_mode='dataset')


def prepare_one(token, cache, model, check_only=False):
    loader = _WORKER_LOADER
    try:
        if not ego_masks.supported(loader, token):
            return dict(clip_id=token, status='dataset')
        path, metadata, reused = ego_masks.ensure_entry(loader, token, cache, model, check_only=check_only)
        return dict(clip_id=token, status='missing' if metadata is None else 'reused' if reused else 'prepared',
                    path=str(path), input_hash=metadata['input_hash'] if metadata else None,
                    mask_sha256=metadata['files']['refined.png'] if metadata else None)
    except Exception as exc:
        return dict(clip_id=token, status='failed', error=f'{type(exc).__name__}: {exc}')


def prepare(args):
    from .edit import select_tokens
    selection = load_selection_manifest(args.manifest)
    tokens = select_tokens(selection.clip_ids, args.max_clips, args.seed)
    if not tokens:
        return []
    if args.ego_mask_mode == 'dataset':
        return [dict(clip_id=token, status='dataset') for token in tokens]
    records = []
    if args.check or args.workers == 1:
        initialize(args, tokens)
        for token in tokens:
            records.append(prepare_one(token, args.ego_mask_cache, args.ego_mask_model, args.check))
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(tokens)),
                                 initializer=initialize, initargs=(args, tokens)) as pool:
            pending = [pool.submit(prepare_one, token, args.ego_mask_cache, args.ego_mask_model) for token in tokens]
            for future in as_completed(pending):
                record = future.result()
                records.append(record)
                print(json.dumps(record), flush=True)
    return sorted(records, key=lambda record: record['clip_id'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_DATA_ROOT / 'pai_av_5500_plus5_filtered_manifest.json')
    parser.add_argument('--anchor-time-s', type=float, default=5.1)
    parser.add_argument('--max-clips', type=int)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=min(4, int(os.environ.get('SLURM_CPUS_PER_TASK', '1'))))
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--status', type=Path, help='Optional atomic preparation report; never written by --check')
    ego_masks.add_arguments(parser)
    args = parser.parse_args(argv)
    if args.workers < 1 or (args.max_clips is not None and args.max_clips < 1):
        parser.error('workers and max-clips must be positive')
    records = prepare(args)
    counts = {status: sum(r['status'] == status for r in records)
              for status in ('dataset', 'reused', 'prepared', 'missing', 'failed')}
    print(json.dumps(dict(method=ego_masks.METHOD, counts=counts)), flush=True)
    for record in records:
        if record['status'] == 'failed':
            print(json.dumps(record), flush=True)
    if args.status and not args.check:
        args.status.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix='.ego-mask-status-', dir=args.status.parent)
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(ego_masks.canonical(dict(counts=counts, records=records)))
            os.replace(temporary, args.status)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return 1 if counts['failed'] else 3 if counts['missing'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
