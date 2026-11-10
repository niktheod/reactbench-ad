#!/usr/bin/env python3
"""Calibrated, temporally consistent 3D compositing geometry for NAVSIM.

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


def project_distorted(points_lidar: np.ndarray, calibration: dict) -> Tuple[np.ndarray, np.ndarray]:
    points_camera = lidar_to_camera(points_lidar, calibration)
    pixels, _ = cv2.projectPoints(
        points_camera.astype(np.float64), np.zeros(3), np.zeros(3),
        np.asarray(calibration["cam_intrinsic"], dtype=np.float64),
        np.asarray(calibration.get("distortion", np.zeros(5)), dtype=np.float64),
    )
    return pixels.reshape(-1, 2), points_camera[:, 2]


def camera_frustum_mask(points_lidar: np.ndarray, calibration: dict, margin: float = 0.04) -> np.ndarray:
    """Reject rays outside the calibrated image before applying distortion.

    The polynomial distortion model is not meaningful far beyond the sensor
    boundary and can fold an off-camera object back into the image.
    """
    points_camera = lidar_to_camera(points_lidar, calibration)
    depth = points_camera[:, 2]
    normalized = points_camera[:, :2] / np.maximum(depth[:, None], 1e-6)
    k = np.asarray(calibration["cam_intrinsic"], dtype=np.float64)
    distortion = np.asarray(calibration.get("distortion", np.zeros(5)), dtype=np.float64)
    image_corners = np.array([[[0, 0]], [[1919, 0]], [[0, 1079]], [[1919, 1079]]], dtype=np.float64)
    undistorted = cv2.undistortPoints(image_corners, k, distortion).reshape(-1, 2)
    x_min, y_min = undistorted.min(axis=0) - margin
    x_max, y_max = undistorted.max(axis=0) + margin
    return (
        (depth > 0.2)
        & (normalized[:, 0] >= x_min) & (normalized[:, 0] <= x_max)
        & (normalized[:, 1] >= y_min) & (normalized[:, 1] <= y_max)
    )


