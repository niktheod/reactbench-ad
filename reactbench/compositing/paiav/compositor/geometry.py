#!/usr/bin/env python3
"""Calibrated, temporally consistent 3D compositing prototype for NAVSIM.

The actor trajectory is defined once in the current ego frame, transformed to
global coordinates, and projected into every historical camera using the
recorded ego pose, camera extrinsics, intrinsics, and distortion coefficients.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation, Slerp


CAMERA_KEYS = {
    "front_left_camera": "CAM_L0",
    "front_camera": "CAM_F0",
    "front_right_camera": "CAM_R0",
}


@dataclass
class Material:
    color: Tuple[int, int, int]
    alpha: float = 1.0


@dataclass
class Mesh:
    vertices: np.ndarray
    faces: np.ndarray
    material: Material


def homogeneous(points: np.ndarray) -> np.ndarray:
    return np.concatenate([points, np.ones((len(points), 1), dtype=points.dtype)], axis=1)


def transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return (np.asarray(matrix, dtype=np.float64) @ homogeneous(points).T).T[:, :3]


def find_log_path(data_root: Path, split: str, first_image: str) -> Path:
    scene_name = Path(first_image).parents[1].name
    path = data_root / "navsim_logs" / split / f"{scene_name}.pkl"
    if not path.is_file():
        raise FileNotFoundError(f"NAVSIM log not found: {path}")
    return path


def image_stem(frame: dict, camera_name: str) -> str:
    return Path(frame["cams"][camera_name]["data_path"]).stem


def match_frames(log_frames: Sequence[dict], sample: dict) -> List[dict]:
    target_stems = [Path(p).stem for p in sample["front_camera"]]
    lookup = {image_stem(frame, "CAM_F0"): frame for frame in log_frames}
    missing = [stem for stem in target_stems if stem not in lookup]
    if missing:
        raise KeyError(f"Could not match front-camera images in log: {missing}")
    frames = [lookup[stem] for stem in target_stems]
    for annotation_key, camera_name in CAMERA_KEYS.items():
        for idx, (path, frame) in enumerate(zip(sample[annotation_key], frames)):
            if Path(path).stem != image_stem(frame, camera_name):
                raise ValueError(f"Camera/frame mismatch for {annotation_key}[{idx}]")
    return frames


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


def uv_sphere(center, radii, nu=14, nv=9, material=Material((220, 170, 135))) -> Mesh:
    vertices = []
    for iv in range(nv + 1):
        phi = math.pi * iv / nv
        for iu in range(nu):
            theta = 2 * math.pi * iu / nu
            vertices.append([
                center[0] + radii[0] * math.sin(phi) * math.cos(theta),
                center[1] + radii[1] * math.sin(phi) * math.sin(theta),
                center[2] + radii[2] * math.cos(phi),
            ])
    faces = []
    for iv in range(nv):
        for iu in range(nu):
            a = iv * nu + iu
            b = iv * nu + (iu + 1) % nu
            c = (iv + 1) * nu + (iu + 1) % nu
            d = (iv + 1) * nu + iu
            faces.extend([[a, b, c], [a, c, d]])
    return Mesh(np.asarray(vertices, np.float64), np.asarray(faces, np.int32), material)


def cylinder_between(start, end, radius, material, sides=10) -> Mesh:
    start, end = np.asarray(start, float), np.asarray(end, float)
    axis = end - start
    axis /= np.linalg.norm(axis)
    helper = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.85 else np.array([1.0, 0.0, 0.0])
    u = np.cross(axis, helper); u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    vertices = []
    for point in (start, end):
        for i in range(sides):
            angle = 2 * math.pi * i / sides
            vertices.append(point + radius * (math.cos(angle) * u + math.sin(angle) * v))
    vertices.extend([start, end])
    faces = []
    for i in range(sides):
        j = (i + 1) % sides
        faces.extend([[i, j, sides + j], [i, sides + j, sides + i]])
        faces.append([2 * sides, j, i])
        faces.append([2 * sides + 1, sides + i, sides + j])
    return Mesh(np.asarray(vertices), np.asarray(faces, np.int32), material)


def pedestrian_meshes(height: float, phase: float) -> List[Mesh]:
    """A compact articulated pedestrian model in local (right, forward, up)."""
    s = height / 1.72
    skin = Material((196, 137, 102))
    jacket = Material((166, 43, 40))
    jacket_light = Material((194, 55, 48))
    trousers = Material((35, 43, 55))
    shoes = Material((24, 22, 20))
    hair = Material((38, 27, 20))
    meshes: List[Mesh] = []
    swing = 0.20 * math.sin(phase)

    # Legs and shoes.
    for side, sign in (("left", -1), ("right", 1)):
        direction = sign * swing
        hip = np.array([sign * 0.105, 0.0, 0.90]) * s
        knee = np.array([sign * 0.10, direction * 0.55, 0.50]) * s
        ankle = np.array([sign * 0.10, -direction * 0.45, 0.10]) * s
        meshes.append(cylinder_between(hip, knee, 0.095 * s, trousers))
        meshes.append(cylinder_between(knee, ankle, 0.078 * s, trousers))
        meshes.append(uv_sphere(ankle + np.array([0, 0.055, -0.025]) * s, (0.105*s, 0.17*s, 0.07*s), 10, 6, shoes))

    # Torso, neck, head and hair cap.
    meshes.append(uv_sphere((0, 0, 1.15*s), (0.25*s, 0.16*s, 0.36*s), 16, 10, jacket))
    meshes.append(cylinder_between((0, 0, 1.42*s), (0, 0, 1.50*s), 0.075*s, skin))
    meshes.append(uv_sphere((0, 0, 1.59*s), (0.125*s, 0.115*s, 0.145*s), 16, 10, skin))
    meshes.append(uv_sphere((0, -0.006*s, 1.655*s), (0.13*s, 0.12*s, 0.095*s), 16, 7, hair))

    # Opposing arm swing; hands are visible and help the silhouette read as human.
    for sign in (-1, 1):
        arm_swing = -sign * swing
        shoulder = np.array([sign * 0.22, 0, 1.36]) * s
        elbow = np.array([sign * 0.29, arm_swing * 0.65, 1.08]) * s
        wrist = np.array([sign * 0.25, -arm_swing * 0.45, 0.87]) * s
        meshes.append(cylinder_between(shoulder, elbow, 0.065*s, jacket_light))
        meshes.append(cylinder_between(elbow, wrist, 0.052*s, jacket_light))
        meshes.append(uv_sphere(wrist, (0.06*s, 0.055*s, 0.07*s), 10, 6, skin))
    return meshes


def transform_mesh(mesh: Mesh, position: np.ndarray, yaw: float) -> Mesh:
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
    return Mesh(mesh.vertices @ rotation.T + position, mesh.faces, mesh.material)


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


def face_color(mesh: Mesh, face: np.ndarray, light_direction=np.array([-0.35, -0.45, 0.82])) -> Tuple[int, int, int]:
    tri = mesh.vertices[face]
    normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    norm = np.linalg.norm(normal)
    diffuse = abs(float(np.dot(normal / max(norm, 1e-8), light_direction / np.linalg.norm(light_direction))))
    brightness = 0.58 + 0.42 * diffuse
    return tuple(int(np.clip(channel * brightness, 0, 255)) for channel in mesh.material.color)


def render_meshes(image: np.ndarray, meshes: Sequence[Mesh], calibration: dict, supersample: int, wireframe: bool) -> Tuple[np.ndarray, np.ndarray, dict]:
    h, w = image.shape[:2]
    scale = max(1, supersample)
    canvas = np.zeros((h * scale, w * scale, 4), dtype=np.uint8)
    triangles = []
    all_pixels = []
    all_depths = []
    for mesh in meshes:
        pixels, depths = project_distorted(mesh.vertices, calibration)
        frustum_valid = camera_frustum_mask(mesh.vertices, calibration)
        all_pixels.append(pixels); all_depths.append(depths)
        for face in mesh.faces:
            face_depths = depths[face]
            if np.all(face_depths > 0.2) and np.all(frustum_valid[face]):
                triangles.append((float(np.mean(face_depths)), mesh, face, pixels[face]))

    all_vertices = np.concatenate([mesh.vertices for mesh in meshes]) if meshes else np.empty((0, 3))
    frustum_visible = camera_frustum_mask(all_vertices, calibration) if len(all_vertices) else np.empty(0, dtype=bool)
    if not np.any(frustum_visible):
        return image.copy(), np.zeros((h, w), dtype=np.uint8), {
            "vertices_in_front": int(np.count_nonzero(np.concatenate(all_depths) > 0.2)) if all_depths else 0,
            "vertices_in_frustum": 0,
            "vertices_in_image": 0,
            "pixel_bbox": None,
        }

    # Painter's algorithm is adequate for the separate convex body primitives.
    for _, mesh, face, polygon in sorted(triangles, key=lambda item: item[0], reverse=True):
        poly = np.round(polygon * scale).astype(np.int32)
        color = face_color(mesh, face)
        rgba = (*color, int(round(255 * mesh.material.alpha)))
        cv2.fillConvexPoly(canvas, poly, rgba, lineType=cv2.LINE_AA)
        if wireframe:
            cv2.polylines(canvas, [poly], True, (40, 255, 80, 220), max(1, scale), cv2.LINE_AA)

    if scale != 1:
        canvas = cv2.resize(canvas, (w, h), interpolation=cv2.INTER_AREA)
    alpha = canvas[..., 3:4].astype(np.float32) / 255.0
    rgb = canvas[..., :3].astype(np.float32)
    composite = image.astype(np.float32) * (1.0 - alpha) + rgb * alpha
    pixels = np.concatenate(all_pixels) if all_pixels else np.empty((0, 2))
    depths = np.concatenate(all_depths) if all_depths else np.empty(0)
    in_front = depths > 0.2
    visible = in_front & (pixels[:, 0] >= 0) & (pixels[:, 0] < w) & (pixels[:, 1] >= 0) & (pixels[:, 1] < h)
    stats = {
        "vertices_in_front": int(np.count_nonzero(in_front)),
        "vertices_in_frustum": int(np.count_nonzero(frustum_visible)),
        "vertices_in_image": int(np.count_nonzero(visible)),
        "pixel_bbox": None,
    }
    if np.any(visible):
        selected = pixels[visible]
        stats["pixel_bbox"] = [float(selected[:, 0].min()), float(selected[:, 1].min()), float(selected[:, 0].max()), float(selected[:, 1].max())]
    return np.clip(composite, 0, 255).astype(np.uint8), canvas[..., 3], stats


def add_ground_shadow(image: np.ndarray, position: np.ndarray, yaw: float, calibration: dict, supersample: int) -> Tuple[np.ndarray, np.ndarray]:
    # Soft, flattened ellipse approximating contact/ambient shadow on the fitted road plane.
    angles = np.linspace(0, 2 * math.pi, 48, endpoint=False)
    local = np.column_stack([0.36 * np.cos(angles), 0.60 * np.sin(angles), np.full_like(angles, 0.012)])
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    points = local @ rot.T + position
    pixels, depths = project_distorted(points, calibration)
    # Do not draw a polygon crossing the distortion model's calibrated boundary;
    # extrapolated corner rays can otherwise fold across most of the image.
    if not np.all(depths > 0.2) or not np.all(camera_frustum_mask(points, calibration)):
        return image, np.zeros(image.shape[:2], dtype=np.uint8)
    h, w = image.shape[:2]
    scale = max(1, supersample)
    mask = np.zeros((h * scale, w * scale), dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(pixels * scale).astype(np.int32)], 75, cv2.LINE_AA)
    sigma = max(2, 5 * scale)
    mask = cv2.GaussianBlur(mask, (0, 0), sigma)
    if scale != 1:
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_AREA)
    alpha = mask[..., None].astype(np.float32) / 255.0
    shadowed = np.clip(image.astype(float) * (1.0 - 0.58 * alpha), 0, 255).astype(np.uint8)
    return shadowed, mask


def make_contact_sheet(paths: Dict[Tuple[int, str], Path], output_path: Path, title: str) -> None:
    thumb_w, thumb_h = 640, 360
    margin, label_h = 12, 30
    cameras = ["CAM_L0", "CAM_F0", "CAM_R0"]
    sheet = Image.new("RGB", (3 * thumb_w + 4 * margin, 4 * (thumb_h + label_h) + 5 * margin), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    for frame_idx in range(4):
        for col, camera in enumerate(cameras):
            image = Image.open(paths[(frame_idx, camera)]).convert("RGB")
            image.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
            x = margin + col * (thumb_w + margin)
            y = margin + frame_idx * (thumb_h + label_h + margin)
            sheet.paste(image, (x, y + label_h))
            draw.text((x + 5, y + 7), f"t{frame_idx - 3:+d}  {camera}", fill=(235, 235, 235))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


if __name__ == "__main__":
    raise SystemExit(
        "This module now provides geometry utilities only. Use "
        "python -m reactbench.compositing.paiav.edit with --timestep."
    )
