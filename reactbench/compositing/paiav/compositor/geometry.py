#!/usr/bin/env python3
"""Calibrated, temporally consistent 3D compositing geometry for PAI-AV.

The actor trajectory is defined once in the current ego frame, transformed to
global coordinates, and projected into every historical camera using the
recorded ego pose, camera extrinsics, intrinsics, and distortion coefficients.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


CAMERA_KEYS = {
    "front_left_camera": "CAM_L0",
    "front_camera": "CAM_F0",
    "front_right_camera": "CAM_R0",
}


def homogeneous(points: np.ndarray) -> np.ndarray:
    return np.concatenate([points, np.ones((len(points), 1), dtype=points.dtype)], axis=1)


def transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return (np.asarray(matrix, dtype=np.float64) @ homogeneous(points).T).T[:, :3]


def load_lidar_xyz(data_root: Path, split: str, frame: dict) -> np.ndarray:
    try:
        from nuplan.database.utils.pointclouds.lidar import LidarPointCloud
    except ImportError as exc:
        raise RuntimeError("Run this script in the existing 'navsim' conda environment.") from exc
    path = data_root / "sensor_blobs" / split / frame["lidar_path"]
    pointcloud = LidarPointCloud.from_buffer(io.BytesIO(path.read_bytes()), "pcd").points
    return np.asarray(pointcloud[:3].T, dtype=np.float64)


def fit_road_plane(points: np.ndarray, x_range=(3.0, 20.0), y_range=(-6.0, 6.0)) -> np.ndarray:
    """Fit z = ax + by + c to low, locally smooth LiDAR returns."""
    mask = (
        (points[:, 0] > x_range[0]) & (points[:, 0] < x_range[1])
        & (points[:, 1] > y_range[0]) & (points[:, 1] < y_range[1])
        & (points[:, 2] > -1.5) & (points[:, 2] < 0.4)
    )
    candidates = points[mask]
    if len(candidates) < 100:
        raise RuntimeError("Not enough LiDAR returns to estimate the road plane")

    # A coarse grid suppresses cars/vegetation: use the lower quartile in each cell.
    samples = []
    gx = np.floor(candidates[:, 0] / 0.75).astype(int)
    gy = np.floor(candidates[:, 1] / 0.75).astype(int)
    for cell in np.unique(np.stack([gx, gy], axis=1), axis=0):
        cell_mask = (gx == cell[0]) & (gy == cell[1])
        pts = candidates[cell_mask]
        if len(pts) >= 4:
            z_cut = np.quantile(pts[:, 2], 0.35)
            low = pts[pts[:, 2] <= z_cut]
            samples.append(np.median(low, axis=0))
    samples = np.asarray(samples)
    design = np.column_stack([samples[:, 0], samples[:, 1], np.ones(len(samples))])
    keep = np.ones(len(samples), dtype=bool)
    coeff = np.zeros(3)
    for _ in range(5):
        coeff = np.linalg.lstsq(design[keep], samples[keep, 2], rcond=None)[0]
        residual = samples[:, 2] - design @ coeff
        center = np.median(residual[keep])
        mad = np.median(np.abs(residual[keep] - center)) + 1e-6
        keep = np.abs(residual - center) < max(0.07, 2.5 * 1.4826 * mad)
    if abs(coeff[0]) > 0.2 or abs(coeff[1]) > 0.2:
        raise RuntimeError(f"Implausible road plane: {coeff.tolist()}")
    return coeff


def ground_z(plane: np.ndarray, x: float, y: float) -> float:
    return float(plane[0] * x + plane[1] * y + plane[2])


def lidar_to_camera(points: np.ndarray, calibration: dict) -> np.ndarray:
    r_sensor_to_lidar = np.asarray(calibration["sensor2lidar_rotation"], dtype=float)
    t_sensor_in_lidar = np.asarray(calibration["sensor2lidar_translation"], dtype=float)
    # NAVSIM convention: p_lidar = R @ p_camera + t.
    return (points - t_sensor_in_lidar) @ r_sensor_to_lidar


def has_rolling_shutter_poses(calibration: dict) -> bool:
    required = (
        "rolling_shutter_sensor2lidar_start_rotation",
        "rolling_shutter_sensor2lidar_start_translation",
        "rolling_shutter_sensor2lidar_end_rotation",
        "rolling_shutter_sensor2lidar_end_translation",
    )
    return (
        str(calibration.get("shutter_type", "")).upper()
        == "ROLLING_TOP_TO_BOTTOM"
        and all(name in calibration for name in required)
    )


def rolling_shutter_sensor_poses(
    calibration: dict,
    row_fractions: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate camera-to-common-frame poses at normalized sensor rows."""
    fractions = np.clip(
        np.asarray(row_fractions, dtype=np.float64).reshape(-1), 0.0, 1.0
    )
    if not has_rolling_shutter_poses(calibration):
        rotation = np.asarray(
            calibration["sensor2lidar_rotation"], dtype=np.float64
        )
        translation = np.asarray(
            calibration["sensor2lidar_translation"], dtype=np.float64
        )
        return (
            np.repeat(rotation[None], len(fractions), axis=0),
            np.repeat(translation[None], len(fractions), axis=0),
        )

    start_rotation = np.asarray(
        calibration["rolling_shutter_sensor2lidar_start_rotation"],
        dtype=np.float64,
    )
    end_rotation = np.asarray(
        calibration["rolling_shutter_sensor2lidar_end_rotation"],
        dtype=np.float64,
    )
    rotations = Slerp(
        [0.0, 1.0],
        Rotation.from_matrix(np.stack([start_rotation, end_rotation])),
    )(fractions).as_matrix()
    start_translation = np.asarray(
        calibration["rolling_shutter_sensor2lidar_start_translation"],
        dtype=np.float64,
    )
    end_translation = np.asarray(
        calibration["rolling_shutter_sensor2lidar_end_translation"],
        dtype=np.float64,
    )
    translations = (
        (1.0 - fractions[:, None]) * start_translation
        + fractions[:, None] * end_translation
    )
    return rotations, translations


def project_distorted_rolling_shutter(
    points_lidar: np.ndarray,
    calibration: dict,
    iterations: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project common-frame points at their iteratively estimated sensor rows."""
    points_lidar = np.asarray(points_lidar, dtype=np.float64)
    if not has_rolling_shutter_poses(calibration) or len(points_lidar) == 0:
        return project_distorted(points_lidar, calibration)

    points_camera = lidar_to_camera(points_lidar, calibration)
    pixels = camera_rays_to_pixels(points_camera, calibration)
    _width, height = camera_resolution(calibration)
    denominator = max(height - 1, 1)
    for _ in range(max(1, int(iterations))):
        fractions = np.nan_to_num(
            pixels[:, 1] / denominator,
            nan=0.5,
            posinf=1.0,
            neginf=0.0,
        )
        rotations, translations = rolling_shutter_sensor_poses(
            calibration, fractions
        )
        points_camera = np.einsum(
            "ni,nij->nj",
            points_lidar - translations,
            rotations,
        )
        pixels = camera_rays_to_pixels(points_camera, calibration)
    return pixels, points_camera[:, 2]


def camera_model_type(calibration: dict) -> str:
    """Return the normalized lens-model name used by a calibration record."""
    return str(calibration.get("camera_model", "brown_conrady")).strip().lower()


def camera_resolution(calibration: dict) -> Tuple[int, int]:
    """Return ``(width, height)`` without assuming NAVSIM's 1920x1080 size."""
    resolution = calibration.get("resolution")
    if resolution is not None and len(resolution) == 2:
        return int(resolution[0]), int(resolution[1])
    return int(calibration.get("width", 1920)), int(calibration.get("height", 1080))


def camera_model_cache_key(calibration: dict) -> tuple:
    """Build a hashable key for cached dense lens remaps."""
    model = camera_model_type(calibration)
    intrinsic = np.asarray(calibration["cam_intrinsic"], dtype=np.float64)
    if model == "ftheta":
        return (
            model,
            intrinsic.tobytes(),
            np.asarray(
                calibration["angle_to_pixeldist_poly"], dtype=np.float64
            ).tobytes(),
            np.asarray(
                calibration["pixeldist_to_angle_poly"], dtype=np.float64
            ).tobytes(),
            float(calibration["max_angle_rad"]),
        )
    distortion = np.asarray(
        calibration.get("distortion", np.zeros(5)), dtype=np.float64
    )
    return model, intrinsic.tobytes(), distortion.tobytes()


def camera_rays_to_pixels(rays_camera: np.ndarray, calibration: dict) -> np.ndarray:
    """Project camera-frame rays through Brown-Conrady or PAI-AV F-theta lenses."""
    rays = np.asarray(rays_camera, dtype=np.float64)
    if rays.shape[-1] != 3:
        raise ValueError(f"camera rays must end in three coordinates, got {rays.shape}")
    model = camera_model_type(calibration)
    if model != "ftheta":
        original_shape = rays.shape[:-1]
        pixels, _ = cv2.projectPoints(
            rays.reshape(-1, 3),
            np.zeros(3),
            np.zeros(3),
            np.asarray(calibration["cam_intrinsic"], dtype=np.float64),
            np.asarray(
                calibration.get("distortion", np.zeros(5)), dtype=np.float64
            ),
        )
        return pixels.reshape(*original_shape, 2)

    xy_norm = np.linalg.norm(rays[..., :2], axis=-1)
    ray_norm = np.linalg.norm(rays, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        theta = np.arccos(
            np.clip(
                np.divide(
                    rays[..., 2],
                    ray_norm,
                    out=np.ones_like(ray_norm),
                    where=ray_norm > 0,
                ),
                -1.0,
                1.0,
            )
        )
    radius = np.polynomial.polynomial.polyval(
        theta,
        np.asarray(calibration["angle_to_pixeldist_poly"], dtype=np.float64),
    )
    radial = np.divide(
        rays[..., :2],
        xy_norm[..., None],
        out=np.zeros_like(rays[..., :2]),
        where=xy_norm[..., None] > 1e-12,
    )
    principal_point = np.asarray(
        calibration.get(
            "principal_point",
            [
                calibration["cam_intrinsic"][0][2],
                calibration["cam_intrinsic"][1][2],
            ],
        ),
        dtype=np.float64,
    )
    return principal_point + radius[..., None] * radial


def pixels_to_camera_rays(pixels: np.ndarray, calibration: dict) -> np.ndarray:
    """Unproject sensor pixels to normalized camera-frame rays."""
    pixels = np.asarray(pixels, dtype=np.float64)
    if pixels.shape[-1] != 2:
        raise ValueError(f"pixels must end in two coordinates, got {pixels.shape}")
    model = camera_model_type(calibration)
    if model != "ftheta":
        original_shape = pixels.shape[:-1]
        normalized = cv2.undistortPoints(
            pixels.reshape(-1, 1, 2),
            np.asarray(calibration["cam_intrinsic"], dtype=np.float64),
            np.asarray(
                calibration.get("distortion", np.zeros(5)), dtype=np.float64
            ),
        ).reshape(*original_shape, 2)
        rays = np.concatenate(
            [normalized, np.ones((*original_shape, 1), dtype=np.float64)], axis=-1
        )
        return rays / np.linalg.norm(rays, axis=-1, keepdims=True)

    principal_point = np.asarray(
        calibration.get(
            "principal_point",
            [
                calibration["cam_intrinsic"][0][2],
                calibration["cam_intrinsic"][1][2],
            ],
        ),
        dtype=np.float64,
    )
    centered = pixels - principal_point
    radius = np.linalg.norm(centered, axis=-1)
    theta = np.polynomial.polynomial.polyval(
        radius,
        np.asarray(calibration["pixeldist_to_angle_poly"], dtype=np.float64),
    )
    radial = np.divide(
        centered,
        radius[..., None],
        out=np.zeros_like(centered),
        where=radius[..., None] > 1e-12,
    )
    sine = np.sin(theta)
    rays = np.concatenate(
        [radial * sine[..., None], np.cos(theta)[..., None]], axis=-1
    )
    rays[..., 2] = np.where(radius > 1e-12, rays[..., 2], 1.0)
    return rays / np.linalg.norm(rays, axis=-1, keepdims=True)


def project_distorted(points_lidar: np.ndarray, calibration: dict) -> Tuple[np.ndarray, np.ndarray]:
    points_camera = lidar_to_camera(points_lidar, calibration)
    return camera_rays_to_pixels(points_camera, calibration), points_camera[:, 2]


def camera_frustum_mask(points_lidar: np.ndarray, calibration: dict, margin: float = 0.04) -> np.ndarray:
    """Reject rays outside the calibrated image before applying distortion.

    The polynomial distortion model is not meaningful far beyond the sensor
    boundary and can fold an off-camera object back into the image.
    """
    points_camera = lidar_to_camera(points_lidar, calibration)
    depth = points_camera[:, 2]
    if camera_model_type(calibration) == "ftheta":
        pixels = camera_rays_to_pixels(points_camera, calibration)
        width, height = camera_resolution(calibration)
        xy_norm = np.linalg.norm(points_camera[:, :2], axis=1)
        theta = np.arctan2(xy_norm, depth)
        pixel_margin = float(margin) * max(width, height)
        return (
            (depth > 0.2)
            & np.isfinite(pixels).all(axis=1)
            & (theta <= float(calibration["max_angle_rad"]) + float(margin))
            & (pixels[:, 0] >= -pixel_margin)
            & (pixels[:, 0] < width + pixel_margin)
            & (pixels[:, 1] >= -pixel_margin)
            & (pixels[:, 1] < height + pixel_margin)
        )
    normalized = points_camera[:, :2] / np.maximum(depth[:, None], 1e-6)
    k = np.asarray(calibration["cam_intrinsic"], dtype=np.float64)
    distortion = np.asarray(calibration.get("distortion", np.zeros(5)), dtype=np.float64)
    width, height = camera_resolution(calibration)
    image_corners = np.array(
        [[[0, 0]], [[width - 1, 0]], [[0, height - 1]], [[width - 1, height - 1]]],
        dtype=np.float64,
    )
    undistorted = cv2.undistortPoints(image_corners, k, distortion).reshape(-1, 2)
    x_min, y_min = undistorted.min(axis=0) - margin
    x_max, y_max = undistorted.max(axis=0) + margin
    return (
        (depth > 0.2)
        & (normalized[:, 0] >= x_min) & (normalized[:, 0] <= x_max)
        & (normalized[:, 1] >= y_min) & (normalized[:, 1] <= y_max)
    )


