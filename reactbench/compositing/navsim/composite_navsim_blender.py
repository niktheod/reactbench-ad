#!/usr/bin/env python3
"""Render the packed MakeHuman pedestrian into one calibrated NAVSIM sample."""

from __future__ import annotations

import math
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

import composite_navsim as geometry


CAMERA_KEYS = geometry.CAMERA_KEYS


# Camera calibration is normally constant for many NAVSIM samples.  Keep only
# the small overscan descriptions here (not the dense remap arrays) so exact
# distortion bounds are computed once per calibration without retaining large
# temporary grids indefinitely.
_OVERSCAN_CACHE: Dict[Tuple[int, int, bytes, bytes, int], dict] = {}


def matrix4(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def camera_to_lidar(calibration: dict) -> np.ndarray:
    """NAVSIM stores p_lidar = R_sensor_to_lidar p_camera + t."""
    return matrix4(
        np.asarray(calibration["sensor2lidar_rotation"], dtype=np.float64),
        np.asarray(calibration["sensor2lidar_translation"], dtype=np.float64),
    )


def road_basis(
    plane: np.ndarray,
    gravity_up: Optional[np.ndarray] = None,
    max_tilt_degrees: Optional[float] = None,
) -> np.ndarray:
    """Return actor axes, optionally limiting road alignment away from gravity."""
    road_normal = np.array([-plane[0], -plane[1], 1.0], dtype=np.float64)
    road_normal /= np.linalg.norm(road_normal)
    if gravity_up is None or max_tilt_degrees is None:
        normal = road_normal
    else:
        gravity_up = np.asarray(gravity_up, dtype=np.float64)
        gravity_up /= np.linalg.norm(gravity_up)
        if np.dot(gravity_up, road_normal) < 0:
            road_normal = -road_normal
        angle = math.acos(float(np.clip(np.dot(gravity_up, road_normal), -1.0, 1.0)))
        limited = min(angle, math.radians(max_tilt_degrees))
        if angle < 1e-9 or limited < 1e-9:
            normal = gravity_up
        else:
            axis = np.cross(gravity_up, road_normal)
            axis /= np.linalg.norm(axis)
            normal = (
                gravity_up * math.cos(limited)
                + np.cross(axis, gravity_up) * math.sin(limited)
                + axis * np.dot(axis, gravity_up) * (1.0 - math.cos(limited))
            )
            normal /= np.linalg.norm(normal)
    forward = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    forward -= normal * np.dot(forward, normal)
    forward /= np.linalg.norm(forward)
    left = np.cross(normal, forward)
    left /= np.linalg.norm(left)
    return np.column_stack([forward, left, normal])


def current_ego_to_actor_frame(actor_position: np.ndarray, basis: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = basis.T
    transform[:3, 3] = -basis.T @ actor_position
    return transform


def actor_bbox_lidar(
    position: np.ndarray,
    actor_basis_lidar: np.ndarray,
    actor_heading: float,
    local_bounds_m: dict,
    scale: float = 1.0,
) -> np.ndarray:
    """Transform measured asset-local bounds exactly like the renderer root."""
    position = np.asarray(position, dtype=np.float64)
    basis = np.asarray(actor_basis_lidar, dtype=np.float64)
    scale = float(scale)
    if position.shape != (3,) or basis.shape != (3, 3):
        raise ValueError("Actor position/basis must have shapes (3,) and (3, 3)")
    if not np.isfinite(position).all() or not np.isfinite(basis).all():
        raise ValueError("Actor position/basis must be finite")
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"Actor scale must be finite and positive, got {scale}")
    try:
        local = np.array(
            [
                [x, y, z]
                for x in local_bounds_m["x"]
                for y in local_bounds_m["y"]
                for z in local_bounds_m["z"]
            ],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("local_bounds_m must contain numeric x/y/z intervals") from exc
    if local.shape != (8, 3) or not np.isfinite(local).all():
        raise ValueError("local_bounds_m must define one finite 3D box")
    angle = float(actor_heading)
    if not math.isfinite(angle):
        raise ValueError(f"actor_heading must be finite, got {angle}")
    c, s = math.cos(angle), math.sin(angle)
    local_to_actor = np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return ((local * scale) @ local_to_actor.T) @ basis.T + position


def distortion_maps(width: int, height: int, intrinsic: np.ndarray, distortion: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Map each distorted target pixel to its ideal pinhole source pixel."""
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    distorted_pixels = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 1, 2)
    ideal_pixels = cv2.undistortPoints(
        distorted_pixels,
        intrinsic.astype(np.float64),
        distortion.astype(np.float64),
        P=intrinsic.astype(np.float64),
    ).reshape(height, width, 2)
    return ideal_pixels[..., 0].astype(np.float32), ideal_pixels[..., 1].astype(np.float32)


def distortion_overscan(calibration: dict, width: int, height: int, padding: int = 3) -> dict:
    """Return the ideal-pinhole canvas needed by a distorted output image.

    Brown-Conrady remapping can request ideal pixels outside the nominal sensor
    rectangle.  Rendering only that rectangle clips actors near a side-camera
    boundary before distortion is applied.  The returned intrinsic has its
    principal point shifted into an expanded canvas whose origin is expressed
    in nominal ideal-image coordinates.
    """
    intrinsic = np.asarray(calibration["cam_intrinsic"], dtype=np.float64)
    distortion = np.asarray(calibration["distortion"], dtype=np.float64)
    key = (width, height, intrinsic.tobytes(), distortion.tobytes(), int(padding))
    cached = _OVERSCAN_CACHE.get(key)
    if cached is not None:
        return {
            **cached,
            "intrinsic": np.asarray(cached["intrinsic"], dtype=np.float64).copy(),
        }

    map_x, map_y = distortion_maps(width, height, intrinsic, distortion)
    finite = np.isfinite(map_x) & np.isfinite(map_y)
    if not np.any(finite):
        raise ValueError("Camera distortion produced no finite ideal-pixel coordinates")
    x_min = int(math.floor(float(np.min(map_x[finite])))) - padding
    x_max = int(math.ceil(float(np.max(map_x[finite])))) + padding
    y_min = int(math.floor(float(np.min(map_y[finite])))) - padding
    y_max = int(math.ceil(float(np.max(map_y[finite])))) + padding
    shifted_intrinsic = intrinsic.copy()
    shifted_intrinsic[0, 2] -= x_min
    shifted_intrinsic[1, 2] -= y_min
    result = {
        "origin": [x_min, y_min],
        "resolution": [x_max - x_min + 1, y_max - y_min + 1],
        "intrinsic": shifted_intrinsic,
        "nominal_resolution": [int(width), int(height)],
        "padding_px": int(padding),
        "ideal_bounds": [x_min, y_min, x_max, y_max],
    }
    if len(_OVERSCAN_CACHE) >= 64:
        _OVERSCAN_CACHE.clear()
    _OVERSCAN_CACHE[key] = {
        **result,
        "intrinsic": shifted_intrinsic.tolist(),
    }
    return result


def _offset_distortion_maps(
    calibration: dict,
    cache: dict,
    output_width: int,
    output_height: int,
    ideal_origin: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    intrinsic = np.asarray(calibration["cam_intrinsic"], dtype=np.float64)
    distortion = np.asarray(calibration["distortion"], dtype=np.float64)
    key = (output_width, output_height, intrinsic.tobytes(), distortion.tobytes())
    if key not in cache:
        cache[key] = distortion_maps(output_width, output_height, intrinsic, distortion)
    map_x, map_y = cache[key]
    origin_x, origin_y = ideal_origin
    return map_x - float(origin_x), map_y - float(origin_y)


def distort_rgba(
    ideal_rgba: np.ndarray,
    calibration: dict,
    cache: dict,
    output_size: Optional[Tuple[int, int]] = None,
    ideal_origin: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Distort an ideal render, accounting for an optional overscan origin."""
    ideal_height, ideal_width = ideal_rgba.shape[:2]
    output_width, output_height = output_size or (ideal_width, ideal_height)
    map_x, map_y = _offset_distortion_maps(
        calibration, cache, output_width, output_height, ideal_origin
    )
    rgba = ideal_rgba.astype(np.float32) / 255.0
    alpha = rgba[..., 3:4]
    premultiplied = np.concatenate([rgba[..., :3] * alpha, alpha], axis=2)
    remapped = cv2.remap(
        premultiplied,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0.0, 0.0, 0.0, 0.0),
    )
    remapped_alpha = remapped[..., 3:4]
    rgb = np.divide(
        remapped[..., :3],
        np.maximum(remapped_alpha, 1e-6),
        out=np.zeros_like(remapped[..., :3]),
        where=remapped_alpha > 1e-6,
    )
    return np.rint(np.clip(np.concatenate([rgb, remapped_alpha], axis=2), 0, 1) * 255).astype(np.uint8)


def distort_scalar_map(
    ideal_values: np.ndarray,
    calibration: dict,
    cache: dict,
    output_size: Tuple[int, int],
    ideal_origin: Tuple[float, float] = (0.0, 0.0),
    border_value: float = 0.0,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Apply the same lens remap to a scalar render pass."""
    output_width, output_height = output_size
    map_x, map_y = _offset_distortion_maps(
        calibration, cache, output_width, output_height, ideal_origin
    )
    return cv2.remap(
        ideal_values.astype(np.float32),
        map_x,
        map_y,
        interpolation=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float(border_value),
    )


def resize_rgba_premultiplied(rgba: np.ndarray, width: int, height: int) -> np.ndarray:
    values = rgba.astype(np.float32) / 255.0
    alpha = values[..., 3:4]
    premultiplied = np.concatenate([values[..., :3] * alpha, alpha], axis=2)
    resized = cv2.resize(premultiplied, (width, height), interpolation=cv2.INTER_AREA)
    resized_alpha = resized[..., 3:4]
    rgb = np.divide(
        resized[..., :3], np.maximum(resized_alpha, 1e-6),
        out=np.zeros_like(resized[..., :3]), where=resized_alpha > 1e-6,
    )
    return np.rint(np.clip(np.concatenate([rgb, resized_alpha], axis=2), 0, 1) * 255).astype(np.uint8)


def blur_rgba_premultiplied(rgba: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return rgba
    values = rgba.astype(np.float32) / 255.0
    alpha = values[..., 3:4]
    premultiplied = np.concatenate([values[..., :3] * alpha, alpha], axis=2)
    blurred = cv2.GaussianBlur(premultiplied, (0, 0), sigma)
    blurred_alpha = blurred[..., 3:4]
    rgb = np.divide(
        blurred[..., :3], np.maximum(blurred_alpha, 1e-6),
        out=np.zeros_like(blurred[..., :3]), where=blurred_alpha > 1e-6,
    )
    return np.rint(np.clip(np.concatenate([rgb, blurred_alpha], axis=2), 0, 1) * 255).astype(np.uint8)


def srgb_to_linear(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    return np.where(values <= 0.0031308, values * 12.92, 1.055 * values ** (1.0 / 2.4) - 0.055)


def apply_shadow_linear(background: np.ndarray, shadow_alpha: np.ndarray, strength: float) -> np.ndarray:
    amount = np.clip(strength * shadow_alpha.astype(np.float32) / 255.0, 0.0, 1.0)
    changed = amount > 0
    if not np.any(changed):
        return background.copy()
    linear = srgb_to_linear(background.astype(np.float32) / 255.0)
    linear *= (1.0 - amount[..., None])
    result = np.rint(linear_to_srgb(linear) * 255.0).astype(np.uint8)
    result[~changed] = background[~changed]
    return result


def alpha_composite(background: np.ndarray, foreground_rgba: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Composite only the alpha support while preserving dense-path pixels."""
    actor_mask = foreground_rgba[..., 3]
    changed_y, changed_x = np.nonzero(actor_mask)
    if changed_y.size == 0:
        return background.copy(), actor_mask

    y_min, y_max = int(changed_y.min()), int(changed_y.max()) + 1
    x_min, x_max = int(changed_x.min()), int(changed_x.max()) + 1
    background_roi = background[y_min:y_max, x_min:x_max]
    foreground_roi = foreground_rgba[y_min:y_max, x_min:x_max]
    alpha = foreground_roi[..., 3:4].astype(np.float32) / 255.0
    changed = alpha[..., 0] > 0
    background_linear = srgb_to_linear(
        background_roi.astype(np.float32) / 255.0
    )
    foreground_linear = srgb_to_linear(
        foreground_roi[..., :3].astype(np.float32) / 255.0
    )
    composite_linear = background_linear * (1.0 - alpha) + foreground_linear * alpha
    composite_roi = np.rint(
        linear_to_srgb(composite_linear) * 255.0
    ).astype(np.uint8)
    composite_roi[~changed] = background_roi[~changed]
    composite = background.copy()
    composite[y_min:y_max, x_min:x_max] = composite_roi
    return composite, actor_mask


def run_blender(
    blender: Path,
    blend_file: Path,
    renderer: Path,
    manifest_path: Path,
) -> None:
    command = [
        str(blender),
        "-b",
        str(blend_file),
        "--python",
        str(renderer),
        "--",
        "--manifest",
        str(manifest_path),
    ]
    completed = subprocess.run(command, text=True, capture_output=True)
    log_path = manifest_path.with_suffix(".blender.log")
    log_path.write_text(completed.stdout + "\n--- STDERR ---\n" + completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"Blender failed for {manifest_path}; see {log_path}")


def save_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path, compress_level=3)


if __name__ == "__main__":
    raise SystemExit(
        "This module now provides rendering utilities only. Use "
        "batch_composite_navsim_blender.py with --timestep."
    )
