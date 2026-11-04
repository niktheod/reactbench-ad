"""Versioned GLS mask preparation and atomic, actor-independent shared cache."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import zipfile
from functools import lru_cache

import numpy as np

from .. import locations
from PIL import Image

CAMERA = 'camera_front_wide_120fov'
METHOD = 'v2_outer_contour_04'
CACHE_VERSION = 'paiav_refined_gls_cache_v1'
SOURCE = 'refined_gls_ego_mask'
DATASET_SOURCE = 'camera_intrinsics.offline.ego_mask_image_png'
DEFAULT_MODE = 'refined-gls'
MODES = (DEFAULT_MODE, 'dataset')
DEFAULT_CACHE = locations.REACTPAI / '_cache' / 'ego_masks'
DEFAULT_MODEL = Path(__file__).parent / 'assets/ego_vehicle/gls_x167_mask_model.json'
CONTEXT_TIMES_US = np.unique(np.r_[np.rint(np.linspace(3., 7.5, 31)*1e6).astype(np.int64), [4800000, 4900000, 5000000, 5100000]])
MASK_NAMES = ('dataset', 'hood', 'attachment_proposal', 'attachment_applied', 'refined', 'uncertain')


class EgoMaskError(RuntimeError):
    """A supported clip cannot safely use its required refined mask."""


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), default=json_default, allow_nan=False).encode()


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def add_arguments(parser):
    parser.add_argument('--ego-mask-mode', choices=MODES, default=DEFAULT_MODE,
                        help='Default: automatically prepare/use refined masks for supported GLS clips')
    parser.add_argument('--ego-mask-cache', type=Path, default=DEFAULT_CACHE)
    parser.add_argument('--ego-mask-model', type=Path, default=DEFAULT_MODEL,
                        help='Frozen model JSON (contains external archive location and expected hashes)')


def loader_options(args):
    return dict(ego_mask_mode=getattr(args, 'ego_mask_mode', DEFAULT_MODE),
                ego_mask_cache=getattr(args, 'ego_mask_cache', DEFAULT_CACHE),
                ego_mask_model=getattr(args, 'ego_mask_model', DEFAULT_MODEL))


@lru_cache(maxsize=8)
def _read_model(path, mtime_ns, size):
    config = json.loads(Path(path).read_text())
    if config['method'] != METHOD or config['attachment_fixed_bottom_z_m'] != .75:
        raise EgoMaskError('Unsupported frozen GLS model configuration')
    return config


def read_model(path=DEFAULT_MODEL):
    path = Path(path).expanduser().resolve()
    stat = path.stat()
    return _read_model(str(path), stat.st_mtime_ns, stat.st_size)


def supported(loader, token, camera=CAMERA):
    """The accepted calibration/dimension/anchor envelope only."""
    if camera != CAMERA or not np.isclose(loader.anchor_time_s, 5.1, rtol=0, atol=1e-6):
        return False
    dims = loader.vehicle_dimensions.loc[token]
    if not np.allclose([dims[k] for k in ('length', 'width', 'height')],
                       [5.207, 2.157, 1.823], atol=.005, rtol=0):
        return False
    row = loader.camera_intrinsics.loc[(token, camera)]
    return str(row['model_type']).lower() == 'ftheta' and json.loads(row['model_parameters'])['resolution'] == [1920, 1080]


def describe_inputs(loader, token, config):
    """Identify compressed source members without decoding images during cache checks.

    ZIP CRC/size and archive stat detect source replacement; preparation also stores
    SHA-256 of every decoded context image. No actor, timestep, or run path enters
    this identity. Reading a ZIP member during preparation verifies its ZIP CRC.
    """
    from . import paiav_io as IO
    archive = loader._feature_paths_for_token(token)[CAMERA]
    stat = archive.stat()
    members = [f'{token}.{CAMERA}.{suffix}' for suffix in ('mp4', 'timestamps.parquet')]
    with zipfile.ZipFile(archive) as z:
        member_info = {name: dict(crc32=z.getinfo(name).CRC,
                                  size=z.getinfo(name).file_size,
                                  compressed_size=z.getinfo(name).compress_size) for name in members}
    timestamps = IO._read_zip_parquet(archive, members[1])
    ts = timestamps['timestamp'].to_numpy(np.int64)
    positions = IO._nearest_unique_indices(ts, CONTEXT_TIMES_US, CAMERA)
    indices = timestamps.index.to_numpy(np.int64)[positions] if timestamps.index.name == 'frame_index' else positions
    calibration = loader.camera_intrinsics.loc[(token, CAMERA)]
    config_identity = {k: v for k, v in config.items() if k not in ('archive', 'frozen_mesh')}
    return dict(cache_version=CACHE_VERSION, method=METHOD, clip_id=token, camera=CAMERA,
                raw_calibration=json.loads(calibration['model_parameters']),
                rig_from_camera=loader._sensor_pose(token, CAMERA).tolist(),
                dimensions=loader.vehicle_dimensions.loc[token].to_dict(),
                dataset_mask_sha256=sha256(bytes(calibration['ego_mask_image_png'])),
                context_nominal_times_us=CONTEXT_TIMES_US.tolist(),
                context_source_indices=indices.tolist(),
                context_source_timestamps_us=ts[positions].tolist(),
                source_archive=dict(path=str(archive.resolve()), size=stat.st_size,
                                    mtime_ns=stat.st_mtime_ns, members=member_info),
                model=config_identity,
                implementation_sha256=file_sha256(Path(__file__).with_name('_ego_mask_refinement.py')))


def cache_path(root, identity):
    # Hash the clip too, so dataset identifiers can never escape the cache root.
    return Path(root) / CACHE_VERSION / sha256(identity['clip_id'].encode())[:16] / sha256(canonical(identity))


def validate_entry(path, identity):
    """Return metadata only for complete, correctly encoded, unchanged artifacts."""
    from .paiav_io import decode_ego_mask_image_png
    try:
        metadata = json.loads((path / 'metadata.json').read_text())
        if canonical(metadata['identity']) != canonical(identity) or metadata['input_hash'] != sha256(canonical(identity)):
            return None
        required = {f'{name}.png' for name in MASK_NAMES} | {'boundaries.npz', 'diagnostics.json'}
        if set(metadata['files']) != required:
            return None
        for name, digest in metadata['files'].items():
            if sha256((path / name).read_bytes()) != digest:
                return None
        for name in MASK_NAMES:
            _, mask, _ = decode_ego_mask_image_png((path / f'{name}.png').read_bytes(), (1920, 1080), str(path))
            if name == 'refined' and int(mask.sum()) != metadata['protected_pixel_count']:
                return None
        if metadata['files']['dataset.png'] != identity['dataset_mask_sha256']:
            return None
        json.loads((path / 'diagnostics.json').read_text())
        with np.load(path / 'boundaries.npz', allow_pickle=False) as boundaries:
            if not {'wide_x', 'wide_curve'} <= set(boundaries.files):
                return None
        return metadata
    except (OSError, ValueError, KeyError, TypeError, RuntimeError):
        return None


@lru_cache(maxsize=4)
def _mesh(config_bytes):
    config = json.loads(config_bytes)
    path = Path(config['frozen_mesh']).expanduser()
    if not path.is_absolute():  # the released hood mesh lies next to the model JSON
        path = DEFAULT_MODEL.parent / path
    if file_sha256(path) != config['frozen_mesh_sha256']:
        raise EgoMaskError(f'Frozen GLS mesh hash mismatch: {path}')
    with np.load(path, allow_pickle=False) as mesh:
        vertices, faces = mesh['v'].copy(), mesh['f'].copy()
    if vertices.shape != (410, 3) or faces.ndim != 2 or faces.shape[1] != 3:
        raise EgoMaskError('Unexpected frozen hood mesh shape')
    if not np.isfinite(vertices).all() or faces.min() < 0 or faces.max() >= len(vertices):
        raise EgoMaskError('Invalid frozen hood mesh geometry')
    return vertices, faces


def build_entry(loader, token, identity, config, destination):
    from . import paiav_io as IO
    from . import _ego_mask_refinement as refinement
    import cv2
    cv2.setNumThreads(1)
    payload = bytes(loader.camera_intrinsics.loc[(token, CAMERA)]['ego_mask_image_png'])
    _, stock, _ = IO.decode_ego_mask_image_png(payload, (1920, 1080), token)
    with tempfile.TemporaryDirectory(prefix='paiav-ego-context-') as scratch:
        indices = identity['context_source_indices']
        outputs = {index: Path(scratch) / f'{index}.png' for index in indices}
        IO._decode_video_indices(loader._feature_paths_for_token(token)[CAMERA],
                                 f'{token}.{CAMERA}.mp4', indices, outputs)
        crops, histories, hashes = [], [], []
        history_times = {4800000, 4900000, 5000000, 5100000}
        for nominal, index in zip(CONTEXT_TIMES_US, indices):
            with Image.open(outputs[index]) as image:
                rgb = np.asarray(image.convert('RGB'))
            if rgb.shape != (1080, 1920, 3):
                raise EgoMaskError(f'Unexpected original image size: {rgb.shape}')
            crops.append(rgb[850:].copy())
            hashes.append(sha256(rgb.tobytes()))
            if nominal in history_times:
                histories.append(rgb)
        vertices, faces = _mesh(canonical(config))
        masks, curves, diagnostics = refinement.compute(np.stack(crops), np.stack(histories), stock,
                                                        identity, vertices, faces, config)
    (destination / 'dataset.png').write_bytes(payload)
    for name, mask in masks.items():
        Image.fromarray(mask.astype(np.uint8)*255).save(destination / f'{name}.png')
    np.savez_compressed(destination / 'boundaries.npz', **curves)
    (destination / 'diagnostics.json').write_bytes(canonical(diagnostics))
    files = {path.name: file_sha256(path) for path in destination.iterdir()}
    metadata = dict(identity=identity, input_hash=sha256(canonical(identity)), files=files,
                    context_rgb_sha256=hashes, protected_pixel_count=int(masks['refined'].sum()),
                    attachment_presence=diagnostics['attachment']['presence_prediction'],
                    resolved_rim=diagnostics['hood']['resolved_rim'])
    (destination / 'metadata.json').write_bytes(canonical(metadata))


def ensure_entry(loader, token, root=DEFAULT_CACHE, model=DEFAULT_MODEL, *, check_only=False):
    config = read_model(model)
    identity = describe_inputs(loader, token, config)
    path = cache_path(root, identity)
    metadata = validate_entry(path, identity)
    if metadata is not None:
        return path, metadata, True
    if check_only:
        return path, None, False
    path.parent.mkdir(parents=True, exist_ok=True)
    with (path.parent / (path.name + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        metadata = validate_entry(path, identity)
        if metadata is not None:
            return path, metadata, True
        # All readers validate complete immutable entries; a failed build leaves
        # any earlier entry intact. Only our invalid cache entry is replaced.
        staging = Path(tempfile.mkdtemp(prefix=path.name+'.building-', dir=path.parent))
        try:
            build_entry(loader, token, identity, config, staging)
            metadata = validate_entry(staging, identity)
            if metadata is None:
                raise EgoMaskError(f'Generated GLS cache failed validation for {token}')
            if path.exists():
                shutil.rmtree(path)
            os.replace(staging, path)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    return path, metadata, False


class Resolver:
    def __init__(self, loader, mode=DEFAULT_MODE, cache=DEFAULT_CACHE, model=DEFAULT_MODEL):
        if mode not in MODES:
            raise ValueError(f'Unsupported ego-mask mode: {mode}')
        self.loader, self.mode = loader, mode
        self.cache, self.model = Path(cache), Path(model)
        self.entries = {}

    def resolve(self, token, camera, payload, dataset_diagnostics):
        if self.mode == 'dataset' or not supported(self.loader, token, camera):
            return payload, dict(dataset_diagnostics, dataset_mask=dict(dataset_diagnostics))
        if token not in self.entries:
            self.entries[token] = ensure_entry(self.loader, token, self.cache, self.model)
        path, metadata, _ = self.entries[token]
        refined = (path / 'refined.png').read_bytes()
        if sha256(refined) != metadata['files']['refined.png']:
            raise EgoMaskError(f'Cached mask changed during editing: {token}')
        return refined, dict(source_field=SOURCE, encoding=dataset_diagnostics['encoding'],
                             resolution=[1920, 1080], sha256=sha256(refined),
                             protected_pixel_count=metadata['protected_pixel_count'],
                             dataset_mask=dict(dataset_diagnostics), method=METHOD,
                             input_hash=metadata['input_hash'],
                             attachment_presence=metadata['attachment_presence'],
                             resolved_rim=metadata['resolved_rim'])

    def run_identity(self, tokens):
        from .paiav_io import CAMERA_NAMES

        entries = {}
        for token in tokens:
            entries[token] = {camera: self.loader._camera_intrinsic(token, camera)['ego_mask_provenance']
                              for camera in CAMERA_NAMES}
        return dict(mode=self.mode, method=METHOD if self.mode != 'dataset' else None,
                    masks=entries)
