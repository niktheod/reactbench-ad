"""Cache publication, source identity, scoped policy and default launcher regression checks."""
from concurrent.futures import ThreadPoolExecutor
import copy
import io
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
import zipfile

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from reactbench.compositing.paiav import ego_masks as masks
from reactbench.compositing.paiav import paiav_io, edit, classify_samples


def png(mask):
    output = io.BytesIO()
    Image.fromarray(np.asarray(mask, dtype=np.uint8)*255).save(output, format='PNG')
    return output.getvalue()


@pytest.fixture
def cache_setup(tmp_path, monkeypatch):
    protected = np.zeros((1080, 1920), bool)
    protected[1000:] = True
    payload = png(protected)
    identity = dict(clip_id='clip', method=masks.METHOD, dataset_mask_sha256=masks.sha256(payload))
    calls = []

    def build(loader, token, inputs, config, destination):
        calls.append(token)
        time.sleep(.03)  # Expose publication races between competing preparers.
        for name in masks.MASK_NAMES:
            (destination / f'{name}.png').write_bytes(payload)
        np.savez_compressed(destination / 'boundaries.npz', wide_x=[1], wide_curve=[1000])
        (destination / 'diagnostics.json').write_text('{}')
        metadata = dict(identity=inputs, input_hash=masks.sha256(masks.canonical(inputs)),
                        protected_pixel_count=int(protected.sum()), attachment_presence='absent', resolved_rim=True,
                        files={p.name: masks.file_sha256(p) for p in destination.iterdir()})
        (destination / 'metadata.json').write_bytes(masks.canonical(metadata))

    monkeypatch.setattr(masks, 'describe_inputs', lambda *args: copy.deepcopy(identity))
    monkeypatch.setattr(masks, 'read_model', lambda *args: {})
    monkeypatch.setattr(masks, 'build_entry', build)
    return tmp_path, identity, calls


def test_cache_check_is_read_only_and_concurrent_build_is_published_once(cache_setup):
    root, identity, calls = cache_setup
    cache = root / 'cache'
    path, metadata, reused = masks.ensure_entry(None, 'clip', cache, check_only=True)
    assert metadata is None and not reused and not cache.exists()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: masks.ensure_entry(None, 'clip', cache), range(2)))
    assert calls == ['clip']
    assert sorted(result[2] for result in results) == [False, True]
    assert masks.validate_entry(path, identity) is not None
    assert not list(path.parent.glob('*.building-*'))


def test_corrupt_cache_is_detected_and_rebuilt_without_silent_fallback(cache_setup, monkeypatch):
    root, identity, calls = cache_setup
    path, _, _ = masks.ensure_entry(None, 'clip', root)
    original_builder = masks.build_entry
    (path / 'refined.png').write_bytes(b'broken')
    assert masks.ensure_entry(None, 'clip', root, check_only=True)[1] is None

    def fail(*args):
        raise masks.EgoMaskError('test preparation failure')

    monkeypatch.setattr(masks, 'build_entry', fail)
    with pytest.raises(masks.EgoMaskError, match='test preparation failure'):
        masks.ensure_entry(None, 'clip', root)
    assert (path / 'refined.png').read_bytes() == b'broken'
    assert not list(path.parent.glob('*.building-*'))
    monkeypatch.setattr(masks, 'build_entry', original_builder)
    _, metadata, reused = masks.ensure_entry(None, 'clip', root)
    assert not reused and metadata is not None and calls == ['clip', 'clip']
    changed = {**identity, 'method': 'different'}
    assert masks.cache_path(root, changed) != path
    assert masks.validate_entry(path, changed) is None


def test_source_identity_tracks_context_calibration_model_and_original_mask(tmp_path, monkeypatch):
    camera = masks.CAMERA
    token = 'clip'
    archive = tmp_path / 'source.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr(f'{token}.{camera}.mp4', b'video')
        z.writestr(f'{token}.{camera}.timestamps.parquet', b'timestamps')
    timestamps = pd.DataFrame({'timestamp': masks.CONTEXT_TIMES_US})
    monkeypatch.setattr(paiav_io, '_read_zip_parquet', lambda *args: timestamps)
    calibration = pd.DataFrame({'model_type': ['ftheta'],
                               'model_parameters': [json.dumps({'resolution': [1920,1080]})],
                               'ego_mask_image_png': [b'original']},
                              index=pd.MultiIndex.from_tuples([(token,camera)]))
    loader = SimpleNamespace(anchor_time_s=5.1, future_timestep=1, run_dir=tmp_path,
                             camera_intrinsics=calibration,
                             vehicle_dimensions=pd.DataFrame({'length':[5.207],'width':[2.157],'height':[1.823]}, index=[token]),
                             _feature_paths_for_token=lambda token: {camera:archive},
                             _sensor_pose=lambda *args: np.eye(4))
    original = masks.describe_inputs(loader, token, {'archive':'unused','member_sha256':'mesh'})
    loader.future_timestep = 20
    loader.run_dir = tmp_path / 'another_actor'
    assert original == masks.describe_inputs(loader, token, {'archive':'relocated','member_sha256':'mesh'})
    assert len(original['context_source_indices']) == 33
    assert original != masks.describe_inputs(loader, token, {'archive':'unused','member_sha256':'newmesh'})
    loader.camera_intrinsics.loc[(token,camera),'ego_mask_image_png'] = b'changed'
    assert original != masks.describe_inputs(loader, token, {'archive':'unused','member_sha256':'mesh'})
    loader.camera_intrinsics.loc[(token,camera),'ego_mask_image_png'] = b'original'
    loader._sensor_pose = lambda *args: np.eye(4)*2
    assert original != masks.describe_inputs(loader, token, {'archive':'unused','member_sha256':'mesh'})
    assert masks.supported(loader, token)
    loader.anchor_time_s = 4.1
    assert not masks.supported(loader, token)
    loader.anchor_time_s = 5.1
    assert not masks.supported(loader, token, 'camera_front_tele_30fov')
    loader.vehicle_dimensions.loc[token,'length'] = 4.7
    assert not masks.supported(loader, token)


def test_resolver_preserves_original_provenance_and_blocks_incompatible_resume(cache_setup, monkeypatch):
    root, _, _ = cache_setup
    monkeypatch.setattr(masks, 'supported', lambda *args: True)
    original = dict(source_field=masks.DATASET_SOURCE, sha256='original', encoding=paiav_io.EGO_MASK_ENCODING_VERSION)
    resolver = masks.Resolver(None, cache=root)
    payload, provenance = resolver.resolve('clip', masks.CAMERA, b'original', original)
    assert provenance['source_field'] == masks.SOURCE
    assert provenance['dataset_mask'] == original
    assert provenance['sha256'] == masks.sha256(payload)
    original_config = {'ego_mask_compositing': edit.ego_mask_compositing_config()}
    current_config = {'ego_mask_compositing': edit.ego_mask_compositing_config({'mode':'refined-gls','masks':{'clip':provenance}})}
    edit.write_or_validate_run_config(root, original_config)
    with pytest.raises(RuntimeError):
        edit.write_or_validate_run_config(root, current_config, allow_occlusion_migration=True)
    legacy = masks.Resolver(None, mode='dataset', cache=root)
    assert legacy.resolve('clip', masks.CAMERA, b'original', original)[0] == b'original'


@pytest.mark.parametrize('sources, fails', [([masks.SOURCE],False), ([masks.DATASET_SOURCE],True),
                                          ([masks.SOURCE,masks.DATASET_SOURCE],True)])
def test_auto_overlap_reports_refined_views_but_keeps_other_views_strict(monkeypatch, sources, fails):
    summary = dict(overlap_detected=True, overlap_views=[dict(mask_source=s) for s in sources])
    monkeypatch.setattr(classify_samples, 'candidate_ego_mask_overlap_diagnostics', lambda *args, **kwargs: summary.copy())
    candidate = {'failures':[]}
    classify_samples.apply_ego_mask_overlap_policy(candidate, [], [], policy='auto')
    assert bool(candidate['failures']) == fails
    assert candidate['ego_mask_overlap']['refined_mask_overlap_policy'] == 'report'




def test_candidate_filter_does_not_promote_incomplete_or_other_failures():
    from reactbench.compositing.paiav.build_gls_candidate_manifest import previous_category
    record = dict(classification='failure', final_failure_reasons=[dict(code='ego_mask_overlap')],
                  candidate_evaluations=[dict(status='not_evaluated', failures=[])])
    selected = previous_category(record)
    assert selected['category'] == 'mask_only'
    assert selected['incomplete_checks'] is True
    record['candidate_evaluations'].append(dict(status='evaluated', failures=[dict(code='trajectory_collision')]))
    assert previous_category(record)['category'] == 'other_failure'
    assert previous_category(dict(classification='failure'))['category'] == 'other_failure'




def test_saved_reference_masks_match_the_frozen_chain():
    """Opt-in golden check after CPU preparation; never starts expensive work here."""
    reference_root = os.environ.get('PAIAV_EGO_REFERENCE_ROOT')
    manifest = os.environ.get('PAIAV_EGO_REFERENCE_MANIFEST')
    if not reference_root or not manifest:
        pytest.skip('Set PAIAV_EGO_REFERENCE_ROOT and PAIAV_EGO_REFERENCE_MANIFEST after CPU preparation')
    selection = paiav_io.load_selection_manifest(Path(manifest))
    loader = paiav_io.PAIAVSceneLoader(edit.DEFAULT_DATA_ROOT, selection, selection.clip_ids,
                                      Path('/tmp'), anchor_time_s=5.1, future_timestep=1,
                                      ego_mask_mode='dataset')
    for token in selection.clip_ids:
        path, metadata, reused = masks.ensure_entry(loader, token, check_only=True)
        assert reused and metadata is not None, 'Run prepare_ego_masks.sh on the reference manifest first'
        reference = Path(reference_root) / token / 'masks'
        for name in ('hood','attachment_proposal','attachment_applied','refined','uncertain'):
            with Image.open(path/f'{name}.png') as new, Image.open(reference/f'{name}.png') as old:
                np.testing.assert_array_equal(np.asarray(new), np.asarray(old), err_msg=f'{token}/{name}')
        with np.load(path/'boundaries.npz') as new, np.load(reference/'boundaries.npz') as old:
            for name in new.files:
                np.testing.assert_array_equal(new[name], old[name], err_msg=f'{token}/{name}')
