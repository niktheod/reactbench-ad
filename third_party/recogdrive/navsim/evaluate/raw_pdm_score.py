"""Training-only raw candidate scoring with the unchanged PDM metrics."""

from concurrent.futures import Future
import math
from threading import Lock
from typing import Callable, Hashable, List, Sequence

import numpy as np

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.common.dataclasses import PDMResults, Trajectory
from navsim.evaluate.pdm_score import get_trajectory_as_array, pdm_results_from_scorer
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import ego_state_to_state_array
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex


def validate_raw_safety_weights(safety_weight, contact_speed_weight):
    for name, value in (("training_raw_safety_weight", safety_weight),
                        ("training_raw_contact_speed_weight", contact_speed_weight)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")


def training_reward_protocol(mode: str, *, raw_safety_weight=.2, raw_contact_speed_weight=.05,
                             raw_safety_inputs=None, minimum_jerk=None) -> dict:
    """Version the state construction and fixed reference used by the reward."""
    if mode not in ("executed", "raw", "executed_raw_safety", "minimum_jerk"):
        raise ValueError("training_reward_mode must be executed, raw, executed_raw_safety or minimum_jerk.")
    if mode == "minimum_jerk":
        from navsim.evaluate.minimum_jerk_pdm_score import protocol_metadata
        return {"version": 1, "mode": mode, "interval_length": 0.1, "time_horizon": 4.0,
                "derivatives": "path-progress-minimum-jerk-v1", "reference": "clean-gt-route-progress-v1",
                "minimum_jerk": protocol_metadata(minimum_jerk)}
    protocol = {
        "version": 1,
        "mode": mode,
        "interval_length": 0.1,
        "time_horizon": 4.0,
        "derivatives": "linear-poses-backward-difference-v1" if mode == "raw" else "pdm-simulator-v1",
        "reference": "executed-pdm-reference-v1",
    }
    if mode == "executed_raw_safety":
        raise NotImplementedError("executed_raw_safety (the raw-safety RL reward) is not part of this release; use executed or minimum_jerk")
    return protocol


def _validate_sampling(sampling: TrajectorySampling) -> None:
    if sampling.num_poses != 40 or not np.isclose(sampling.interval_length, 0.1, rtol=0, atol=1e-12):
        raise ValueError("Raw PDMS requires a 4 s horizon sampled at 10 Hz (40 future poses).")


def validate_raw_observation(metric_cache: MetricCache, sampling: TrajectorySampling) -> None:
    """The stock TTC indices require aligned occupancy maps through t=4.9 s."""
    _validate_sampling(sampling)
    observation = metric_cache.observation
    if not np.isclose(observation._sample_interval, 0.1, rtol=0, atol=1e-12):
        raise ValueError("Raw PDMS observations must use the 10 Hz reward grid.")
    # A coarser observation_sample_res can silently map several ego timestamps
    # to one actor frame even if the nominal sample interval is 0.1 s.
    last_index = sampling.num_poses + 9
    if list(observation._global_to_local_idcs[:last_index + 1]) != list(range(last_index + 1)):
        raise ValueError("Raw PDMS observations must have distinct aligned 10 Hz frames through 4.9 s.")
    try:
        for index in range(last_index + 1):
            observation[index]
    except (AssertionError, IndexError, TypeError) as exc:
        raise ValueError("Raw PDMS observations must cover every 10 Hz frame through 4.9 s.") from exc


def raw_trajectory_states(
    trajectory: Trajectory, initial_ego_state: EgoState, sampling: TrajectorySampling,
) -> np.ndarray:
    """Interpolate rear-axle poses and derive dynamics without executing them.

    Future derivatives use backward differences on the 10 Hz grid. Velocity
    and acceleration fields are in each pose's body frame; position is global.
    The complete measured initial state is retained, including its dynamics.
    """
    _validate_sampling(sampling)
    poses = np.asarray(trajectory.poses, dtype=np.float64)
    native = trajectory.trajectory_sampling
    if (poses.shape != (8, 3) or native.num_poses != 8
            or not np.isclose(native.interval_length, 0.5, rtol=0, atol=1e-12)):
        raise ValueError("Raw PDMS requires eight (x, y, heading) waypoints at 0.5–4.0 s.")
    initial = ego_state_to_state_array(initial_ego_state)
    if not np.isfinite(poses).all() or not np.isfinite(initial).all():
        raise ValueError("Raw PDMS poses and initial state must be finite.")

    dt = sampling.interval_length
    times = np.arange(sampling.num_poses + 1) * dt
    source_times = np.arange(9) * native.interval_length
    local = np.vstack([np.zeros(3), poses])
    heading0 = initial[StateIndex.HEADING]
    c0, s0 = np.cos(heading0), np.sin(heading0)
    rotation = np.array([[c0, -s0], [s0, c0]])
    positions = local[:, :2] @ rotation.T + initial[StateIndex.POINT]
    source_headings = np.unwrap(local[:, 2] + heading0)
    headings = np.interp(times, source_times, source_headings)

    states = np.zeros((len(times), StateIndex.size()), dtype=np.float64)
    for axis in (StateIndex.X, StateIndex.Y):
        states[:, axis] = np.interp(times, source_times, positions[:, axis])
    states[:, StateIndex.HEADING] = (headings + np.pi) % (2 * np.pi) - np.pi

    velocity_world = np.empty((len(times), 2), dtype=np.float64)
    velocity_world[0] = initial[StateIndex.VELOCITY_2D] @ rotation.T
    velocity_world[1:] = np.diff(states[:, StateIndex.POINT], axis=0) / dt
    acceleration_world = np.diff(velocity_world, axis=0) / dt
    c, s = np.cos(headings[1:]), np.sin(headings[1:])
    for world, indices in ((velocity_world[1:], StateIndex.VELOCITY_2D),
                           (acceleration_world, StateIndex.ACCELERATION_2D)):
        states[1:, indices] = np.column_stack((c * world[:, 0] + s * world[:, 1],
                                               -s * world[:, 0] + c * world[:, 1]))
    states[0] = initial
    states[1:, StateIndex.ANGULAR_VELOCITY] = np.diff(headings) / dt
    states[1:, StateIndex.ANGULAR_ACCELERATION] = np.diff(states[:, StateIndex.ANGULAR_VELOCITY]) / dt
    # Steering is unused by PDMScorer; only t=0 retains measured steering.
    if not np.isfinite(states).all():
        raise ValueError("Raw PDMS reconstructed states must be finite.")
    return states


def executed_reference_progress(
    metric_cache: MetricCache, sampling: TrajectorySampling,
    simulator: PDMSimulator, scorer: PDMScorer,
) -> float:
    """Execute only the fixed cached reference, retaining today's EP benchmark."""
    validate_raw_observation(metric_cache, sampling)
    initial = metric_cache.ego_state
    if (metric_cache.trajectory.start_time.time_us > initial.time_point.time_us
            or metric_cache.trajectory.end_time.time_us < initial.time_point.time_us + 4_000_000):
        raise ValueError("Raw PDMS reference must cover the initial state through 4.0 s.")
    reference = get_trajectory_as_array(metric_cache.trajectory, sampling, initial.time_point)
    states = simulator.simulate_proposals(reference[None], initial)
    scorer.score_proposals(states, metric_cache.observation, metric_cache.centerline,
                           metric_cache.route_lane_ids, metric_cache.drivable_area_map)
    progress = float(scorer._progress_raw[0] * scorer._multi_metrics.prod(axis=0)[0])
    if not np.isfinite(progress) or progress < 0:
        raise ValueError("Raw PDMS reference progress must be finite and nonnegative.")
    return progress


def raw_pdm_scores(
    metric_cache: MetricCache, model_trajectories: Sequence[Trajectory],
    future_sampling: TrajectorySampling, scorer: PDMScorer, reference_progress: float,
) -> List[PDMResults]:
    """Score raw candidates with stock PDM metrics and a fixed executed reference."""
    if not model_trajectories:
        return []
    validate_raw_observation(metric_cache, future_sampling)
    if not np.isfinite(reference_progress) or reference_progress < 0:
        raise ValueError("Raw PDMS reference progress must be finite and nonnegative.")
    states = np.stack([raw_trajectory_states(trajectory, metric_cache.ego_state, future_sampling)
                       for trajectory in model_trajectories])
    scorer.score_proposals(states, metric_cache.observation, metric_cache.centerline,
                           metric_cache.route_lane_ids, metric_cache.drivable_area_map)
    return pdm_results_from_scorer(scorer, reference_progress, range(len(model_trajectories)))


class ReferenceProgressCache:
    """Process-local scalar memoization; different scenes compute concurrently.

    Only an in-flight computation keeps a Future. Completed entries are floats.
    Copies/pickles start empty so runtime locks and values never cross processes
    or become part of the teacher's state.
    """

    def __init__(self):
        self._values = {}
        self._inflight = {}
        self._lock = Lock()

    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        self.__init__()

    def get_or_compute(self, key: Hashable, compute: Callable[[], float]) -> float:
        with self._lock:
            if key in self._values:
                return self._values[key]
            future = self._inflight.get(key)
            owner = future is None
            if owner:
                future = self._inflight[key] = Future()
        if not owner:
            return future.result()
        try:
            value = float(compute())
            if not np.isfinite(value) or value < 0:
                raise ValueError("Reference progress must be finite and nonnegative.")
        except BaseException as exc:
            with self._lock:
                del self._inflight[key]
                future.set_exception(exc)
            raise
        with self._lock:
            self._values[key] = value
            del self._inflight[key]
            future.set_result(value)
        return value
