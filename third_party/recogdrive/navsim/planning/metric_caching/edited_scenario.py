"""NAVSIM metric caches of edited training scenes: the inserted actor as an extra tracked object.

The inserted actor (a ReactBench actor-motion record) moves at constant velocity in the scene's
current rear-axle frame; it is overlaid on the recorded tracks, and the PDM reference and the
interpolated observations are rebuilt from the augmented scenario. nuPlan/NAVSIM imports are lazy.
"""
from __future__ import annotations

import importlib.metadata
import math
import os
import sys
from pathlib import Path

import numpy as np

from reactbench.io import digest, file_digest
from reactbench.scoring.actor_motion import validated_motion

CONTRACT = {"version": "edited-training-metric-cache-v1", "horizon_s": 4.0, "interval_s": 0.1,
            "observation_horizon_s": 5.0, "observation_time_policy": "navsim_nominal_2hz"}


def actor_box_state(motion, ego_pose, time_s):
    """World box pose/dimensions/velocity, correcting asymmetric asset roots."""
    motion = validated_motion(motion)
    if motion.get("actor_type") not in ("pedestrian", "car"):
        raise ValueError("PDMS supports inserted pedestrians and cars")
    pose = np.asarray(ego_pose, dtype=float)
    if pose.shape != (3,) or not np.isfinite(pose).all() or not math.isfinite(time_s):
        raise ValueError("Non-finite ego pose or actor time")
    heading = motion["heading_rad"]
    rotation = lambda a: np.array([[math.cos(a), -math.sin(a)],  # noqa: E731
                                   [math.sin(a), math.cos(a)]])
    offset = np.array([(motion["front_length_m"] - motion["rear_length_m"]) / 2,
                       (motion["left_width_m"] - motion["right_width_m"]) / 2])
    local = (np.asarray(motion["initial_center_xy_m"])
             + time_s * np.asarray(motion["velocity_xy_mps"]) + rotation(heading) @ offset)
    center = pose[:2] + rotation(pose[2]) @ local
    yaw = math.atan2(math.sin(pose[2] + heading), math.cos(pose[2] + heading))
    velocity = rotation(pose[2]) @ np.asarray(motion["velocity_xy_mps"])
    # Height has no effect on NAVSIM's BEV metrics. Use measured height if available.
    bounds = motion.get("geometry", {}).get("local_bounds_m", {})
    z = bounds.get("z")
    height = float(z[1] - z[0]) if z is not None else float(motion.get("height_m", 1.0))
    if not math.isfinite(height) or height <= 0:
        raise ValueError("Actor height must be finite and positive")
    return {"pose": [float(center[0]), float(center[1]), yaw],
            "length": motion["length_m"], "width": motion["width_m"], "height": height,
            "velocity": velocity.tolist(), "actor_type": motion["actor_type"]}


class AugmentedScenario:
    """Delegate the scene/map/ego interface, overlaying tracked objects in world space."""

    def __init__(self, scenario, motion, track_token):
        self.base = scenario
        self.motion = validated_motion(motion) if motion is not None else None
        self.track_token = track_token

    def __getattr__(self, name):
        return getattr(self.base, name)

    @property
    def initial_tracked_objects(self):
        return self.get_tracked_objects_at_iteration(0)

    def actor_at(self, time_s, timestamp_us):
        from nuplan.common.actor_state.agent import Agent
        from nuplan.common.actor_state.oriented_box import OrientedBox
        from nuplan.common.actor_state.scene_object import SceneObjectMetadata
        from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D
        from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType

        ego = self.base.initial_ego_state.rear_axle
        state = actor_box_state(self.motion, [ego.x, ego.y, ego.heading], time_s)
        return Agent(
            tracked_object_type=(TrackedObjectType.VEHICLE if state["actor_type"] == "car"
                                 else TrackedObjectType.PEDESTRIAN),
            oriented_box=OrientedBox(StateSE2(*state["pose"]), state["length"],
                                     state["width"], state["height"]),
            velocity=StateVector2D(*state["velocity"]),
            metadata=SceneObjectMetadata(timestamp_us, token=self.track_token,
                                         track_id=None, track_token=self.track_token))

    def get_tracked_objects_at_iteration(self, iteration, future_trajectory_sampling=None):
        from nuplan.common.actor_state.tracked_objects import TrackedObjects
        from nuplan.planning.simulation.observation.observation_type import DetectionsTracks

        original = self.base.get_tracked_objects_at_iteration(iteration, future_trajectory_sampling)
        if self.motion is None:
            return original
        objects = list(original.tracked_objects)
        if any(obj.track_token == self.track_token for obj in objects):
            raise ValueError("Inserted track token collides with a recorded track token")
        timestamp = self.base.get_time_point(iteration)
        # MetricCacheProcessor assigns its 2 Hz source frames nominal relative times.
        # Use that same clock so its 10 Hz interpolation preserves p(t)=p0+v*t.
        time_s = float(iteration) * self.base.database_interval
        objects.append(self.actor_at(time_s, timestamp.time_us))
        return DetectionsTracks(TrackedObjects(objects))

    def get_future_tracked_objects(self, iteration, time_horizon, num_samples=None,
                                   future_trajectory_sampling=None):
        from navsim.planning.scenario_builder.navsim_scenario_utils import sample_future_indices
        indices = sample_future_indices(self.base._future_sampling, iteration,
                                        time_horizon, num_samples)
        for index in indices:
            yield self.get_tracked_objects_at_iteration(index, future_trajectory_sampling)

    def get_past_tracked_objects(self, iteration, time_horizon, num_samples=None,
                                 future_trajectory_sampling=None):
        # The bundled NAVSIM scenario exposes only its initial observation as history.
        yield self.get_tracked_objects_at_iteration(0, future_trajectory_sampling)


def build_metric_cache(scenario, file_path):
    """Rebuild both reference and replay observations using the augmented scenario."""
    from navsim.planning.metric_caching.metric_cache import MetricCache
    from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor

    processor = MetricCacheProcessor(cache_path=None, force_feature_computation=True)
    planner_input, initialization = processor._get_planner_inputs(scenario)
    planner = processor._pdm_closed
    planner.initialize(initialization)
    reference = planner.compute_planner_trajectory(planner_input)
    observations = processor._interpolate_gt_observation(scenario)
    return MetricCache(Path(file_path), reference, scenario.initial_ego_state,
                       observations, planner._centerline, list(planner._route_lane_dict),
                       planner._drivable_area_map)


def reference_diagnostics(cache):
    """Stock PDMS of the cache's PDM reference trajectory in the augmented world."""
    from navsim.evaluate.pdm_score import get_trajectory_as_array
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import MultiMetricIndex
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    states = get_trajectory_as_array(cache.trajectory, sampling, cache.ego_state.time_point)
    simulated = PDMSimulator(sampling).simulate_proposals(states[None], cache.ego_state)
    scorer = PDMScorer(sampling, vehicle_parameters=cache.ego_state.car_footprint.vehicle_parameters)
    score = scorer.score_proposals(simulated, cache.observation, cache.centerline,
                                   cache.route_lane_ids, cache.drivable_area_map)
    nc = float(scorer._multi_metrics[MultiMetricIndex.NO_COLLISION, 0])
    dac = float(scorer._multi_metrics[MultiMetricIndex.DRIVABLE_AREA, 0])
    return {"score": float(score[0]), "no_at_fault_collisions": nc,
            "drivable_area_compliance": dac, "safe": nc == 1 and dac == 1,
            "raw_progress_m": float(scorer._progress_raw[0])}


def numerical_preflight():
    """Reject broken BLAS/LAPACK environments before generating seemingly valid caches."""
    identity = np.eye(40)
    if not np.allclose(np.linalg.pinv(identity), identity, atol=1e-10):
        raise RuntimeError("Numerical preflight failed: NumPy pinv(identity) is incorrect. "
                           "Use a working Python environment; nothing was computed.")
    from navsim.planning.simulation.planner.pdm_planner.simulation.batch_lqr_utils import (
        get_velocity_curvature_profiles_with_derivatives_from_poses,
    )
    poses = np.zeros((1, 41, 3))
    poses[0, :, 0] = np.arange(41)
    velocity, acceleration, curvature, _ = get_velocity_curvature_profiles_with_derivatives_from_poses(
        .1, poses, 1e-4, 1e-2)
    if not (np.allclose(velocity, 10., atol=1e-5) and np.allclose(acceleration, 0., atol=1e-5)
            and np.allclose(curvature, 0., atol=1e-5)):
        raise RuntimeError("Numerical preflight failed: constant-speed LQR profile is incorrect")


def map_identity(root):
    """Fingerprint of the nuPlan maps: GPKG paths/sizes/mtimes and JSON metadata SHA-256."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Missing NAVSIM maps: {root}")
    values = {}
    for path in sorted(root.rglob("*")):
        # Map loading creates locks and raster caches; those are not HD-map inputs.
        if path.suffix in (".gpkg", ".json") and not any(
                part.startswith(".") for part in path.relative_to(root).parts) and path.is_file():
            stat = path.stat()
            values[str(path.relative_to(root))] = ([stat.st_size, stat.st_mtime_ns, file_digest(path)]
                                                   if path.suffix == ".json" else [stat.st_size, stat.st_mtime_ns])
    if not values:
        raise FileNotFoundError(f"Empty map directory: {root}")
    return {"root": str(root), "files_fingerprint": digest(values),
            "method": "GPKG paths/sizes/mtimes and JSON metadata SHA-256; runtime caches/locks excluded"}


def backend_provenance(navsim_root, map_version):
    """Code and package versions the caches were built with."""
    import nuplan
    from reactbench.scoring import actor_motion

    navsim_root = Path(navsim_root) / "navsim"
    files = sorted({p for relative in ("common", "evaluate", "planning") for p in (navsim_root / relative).rglob("*.py")})
    if not files:
        raise ValueError(f"No NAVSIM implementation under {navsim_root}")
    versions = {}
    for package in ("nuplan-devkit", "numpy", "scipy", "shapely", "pyquaternion"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unknown"
    # nuPlan can be editable; version strings alone do not identify local edits.
    nuplan_root = Path(nuplan.__file__).parent
    nuplan_hashes = {str(p.relative_to(nuplan_root)): file_digest(p) for sub in
                     ("common/actor_state", "common/geometry", "planning/simulation", "planning/metrics/utils")
                     for p in (nuplan_root / sub).rglob("*.py")}
    return {"contract": CONTRACT,
            "navsim_sha256": digest({str(p.relative_to(navsim_root)): file_digest(p) for p in files}),
            "adapter_sha256": digest({Path(m.__file__).name: file_digest(m.__file__)
                                      for m in (sys.modules[__name__], actor_motion)}),
            "nuplan_sha256": digest(nuplan_hashes), "versions": versions, "python": sys.version.split()[0],
            "openblas_coretype": os.environ.get("OPENBLAS_CORETYPE"), "map_version": map_version}
