#!/usr/bin/env python3
"""Select GLS clips at 5–40 km/h for fresh classification; never promote old failures."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import copy
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from .ego_masks import canonical, file_sha256
from .. import locations
from .paiav_io import load_selection_manifest, resolve_chunk_feature_paths


def previous_category(record):
    codes = {failure['code'] for failure in record.get('final_failure_reasons', [])}
    codes.update(failure['code'] for candidate in record.get('candidate_evaluations', [])
                 for failure in candidate.get('failures', []))
    codes.update(failure['code'] for failure in (record.get('shared_validation') or {}).get('failures', []))
    category = ('passed' if record['classification'] != 'failure' else
                'mask_only' if codes == {'ego_mask_overlap'} else 'other_failure')
    return dict(category=category, classification=record['classification'],
                failure_codes=sorted(codes),
                incomplete_checks=any(candidate['status'] != 'evaluated'
                                      for candidate in record.get('candidate_evaluations', [])))


def subset_document(selection, tokens):
    if selection.manifest_role != 'pai_av_dataset_manifest':
        raise ValueError('Candidate selection requires a multi-chunk dataset manifest')
    document = copy.deepcopy(selection.manifest)
    counts = Counter(selection.chunk_id_for(token) for token in tokens)
    samples = document['samples']
    samples.update(clip_ids=list(tokens), count=len(tokens), chunk_count=len(counts),
                   chunks=[{**chunk, 'clip_count':counts[chunk['chunk_id']]}
                           for chunk in samples['chunks'] if chunk['chunk_id'] in counts])
    document.pop('metadata_summary', None)
    document['outputs'] = {}
    return document


def build(args):
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite manifest: {args.output}')
    selection = load_selection_manifest(args.manifest)
    records = {}
    for line in args.classification_details.read_text().splitlines():
        record = json.loads(line)
        token = record['clip_id']
        if token in records:
            raise ValueError(f'Duplicate classification record: {token}')
        records[token] = previous_category(record)
    wanted = set(selection.clip_ids)
    paths = resolve_chunk_feature_paths(args.data_root, selection, selection.chunk_ids)

    def measure(chunk_id):
        features = paths[chunk_id]
        dimensions = pd.read_parquet(features['vehicle_dimensions'])
        pool = {}
        with zipfile.ZipFile(features['egomotion']) as archive:
            for token, row in dimensions.iterrows():
                if token not in wanted or records.get(token, {}).get('category') not in ('passed', 'mask_only'):
                    continue
                # Match the established pool's exact rounded dimension-group rule.
                if tuple(round(float(row[k]), 3) for k in ('length','width','height')) != (5.207,2.157,1.823):
                    continue
                table = pd.read_parquet(io.BytesIO(archive.read(f'{token}.egomotion.parquet')),
                                        columns=['timestamp','vx','vy','vz'])
                a = table[['timestamp','vx','vy','vz']].to_numpy(float)
                a = a[np.isfinite(a).all(axis=1)]
                if not len(a):
                    raise ValueError(f'No finite velocity samples: {token}')
                nearest = a[np.argmin(abs(a[:,0]-5100000))]
                delta = abs(nearest[0]-5100000)/1e6
                if delta > .05:
                    raise ValueError(f'No velocity measurement within 50 ms of 5.1 s: {token}')
                speed = float(np.linalg.norm(nearest[1:])*3.6)
                if 5 <= speed <= 40:
                    pool[token] = dict(speed_kmh=speed, measured_timestamp_us=float(nearest[0]),
                                       timestamp_delta_s=float(delta),
                                       dimensions={k:float(row[k]) for k in ('length','width','height','wheelbase')},
                                       **records[token])
        return pool

    pool = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for result in executor.map(measure, selection.chunk_ids):
            pool.update(result)
    if not pool:
        raise ValueError('No eligible GLS candidates')
    document = subset_document(selection, sorted(pool))
    document['candidate_selection'] = dict(
        role='requires_fresh_classification',
        source_manifest=str(selection.manifest_path), source_manifest_sha256=file_sha256(selection.manifest_path),
        classification_details=str(args.classification_details.resolve()),
        classification_sha256=file_sha256(args.classification_details),
        speed_definition='3.6*norm(vx,vy,vz), nearest raw egomotion timestamp to 5.1 s within 50 ms',
        speed_bounds_kmh_inclusive=[5,40],
        previous_categories=dict(Counter(record['category'] for record in pool.values())),
        source_clips_without_classification=len(wanted-set(records)),
        records=pool,
        caution='Mask-only failures are candidates, not passes; rerun every remaining placement and visibility check.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects existing candidate manifests and legacy results.
    with args.output.open('xb') as stream:
        stream.write(canonical(document))
    load_selection_manifest(args.output)
    return dict(output=str(args.output), candidate_count=len(pool),
                previous_categories=document['candidate_selection']['previous_categories'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=locations.PAIAV_ROOT)
    parser.add_argument('--manifest', type=Path, required=True, help='Candidate-pool manifest (reactbench.selection.reactpai)')
    parser.add_argument('--classification-details', type=Path, required=True,
                        help='classify_samples.py per-clip details (JSONL)')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('workers must be positive')
    print(json.dumps(build(args)), flush=True)


if __name__ == '__main__':
    main()
