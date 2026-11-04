"""Recorded NAVSIM actors and HD-map polygons, without a reference planner."""
from __future__ import annotations
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
import fcntl, pickle, math
import numpy as np
from ..io import digest, file_digest, read_json, write_json

@contextmanager
def locked(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


class SceneRepository:
    """Persistent token index with bounded in-process log memory; sensors are never loaded."""

    def __init__(self, config):
        self.config = config
        self.logs = Path(config["log_path"])
        paths = sorted(self.logs.glob("*.pkl"))
        if not paths:
            raise FileNotFoundError(f"No NAVSIM .pkl logs in {self.logs}")
        files = {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in paths}
        identity = digest(files)
        index_path = Path(config["work_root"]) / "log_indices" / f"{identity}.json"
        with locked(index_path.with_suffix(".lock")):
            index = read_json(index_path)
            if index is None:
                tokens = {}
                for path in paths:
                    with path.open("rb") as handle:
                        frames = pickle.load(handle)
                    for idx, frame in enumerate(frames):
                        token = frame["token"]
                        if token in tokens:
                            raise ValueError(f"Duplicate NAVSIM token {token} across source logs")
                        tokens[token] = [str(path), idx]
                index = {"source_files": files, "tokens": tokens}
                write_json(index_path, index)
        self.index = index["tokens"]
        self.loaded = OrderedDict()
        self.map_identity = self._map_identity(Path(config["maps_root"]))

    @staticmethod
    def _map_identity(root):
        if not root.is_dir():
            raise FileNotFoundError(f"Missing NAVSIM maps: {root}")
        # Large map binaries are identified by path, size and mtime; metadata is content-hashed.
        values = {}
        for path in sorted(root.rglob("*")):
            # Map loading creates locks and raster caches; those are not HD-map inputs.
            if path.suffix in (".gpkg", ".json") and not any(
                    part.startswith(".") for part in path.relative_to(root).parts) and path.is_file():
                stat = path.stat()
                values[str(path.relative_to(root))] = ([stat.st_size, stat.st_mtime_ns,
                    file_digest(path)] if path.suffix == ".json" else [stat.st_size, stat.st_mtime_ns])
        if not values:
            raise FileNotFoundError(f"Empty map directory: {root}")
        return {"root": str(root), "files_fingerprint": digest(values),
                "method": "GPKG paths/sizes/mtimes and JSON metadata SHA-256; runtime caches/locks excluded"}

    def scene(self, token):
        from navsim.common.dataclasses import Scene, SensorConfig
        from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario

        if token not in self.index:
            raise KeyError(f"Scene token not found in NAVSIM logs: {token}")
        path, index = self.index[token]
        if path not in self.loaded:
            with open(path, "rb") as handle:
                self.loaded[path] = pickle.load(handle)
            while len(self.loaded) > 1:
                self.loaded.popitem(last=False)
        all_frames = self.loaded[path]
        if index < 3 or index + 8 >= len(all_frames):
            raise ValueError(f"Scene {token} lacks four history frames or four seconds of future tracks")
        frames = all_frames[index - 3:index + 9]
        times = np.asarray([f["timestamp"] for f in frames], dtype=float) / 1e6
        intervals = np.diff(times)
        irregular = np.flatnonzero(~np.isclose(intervals, 0.5, atol=0.025, rtol=0))
        source = {"log_path": path, "token": token,
                  "frames_sha256": __import__("hashlib").sha256(pickle.dumps(frames, protocol=4)).hexdigest(),
                  "map": self.map_identity}
        # Match NAVSIM's cache builder: source frames are assigned nominal 0.5 s
        # steps, even across recorded timestamp gaps. Diagnose without rejecting
        # or resampling the source frames; the same clock drives inserted actors.
        if len(irregular):
            source["timestamp_warning"] = {
                "message": "Irregular source timestamps; using NAVSIM nominal 0.5 s frame spacing",
                "nominal_interval_s": 0.5, "tolerance_s": 0.025,
                "intervals": [{"from_iteration": int(i) - 3,
                               "recorded_interval_s": float(intervals[i])} for i in irregular]}
        scene = Scene.from_scene_dict_list(frames, None, num_history_frames=4,
                                           num_future_frames=8, sensor_config=SensorConfig.build_no_sensors())
        return NavSimScenario(scene, map_root=self.config["maps_root"],
                              map_version=self.config["map_version"]), source


WORLD_CONTRACT = {"version": 1, "horizon_s": 4., "recorded_clock": "NAVSIM nominal 0.5 s frames",
                  "track_lifetime": "interpolate consecutive annotations only; split missing frames; never extrapolate",
                  "map_layers": ["ROADBLOCK", "INTERSECTION", "DRIVABLE_AREA", "CARPARK_AREA"],
                  "boundary": "OR of polygon contains predicates, as in NAVSIM DAC (boundary excluded)",
                  "drivable_area": "Only layers exposed as vector map objects; stock NAVSIM does not populate DRIVABLE_AREA"}


def build_world(scenario, source, radius=100.):
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
    state = scenario.initial_ego_state
    origin = state.rear_axle
    theta = origin.heading
    rotation = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    tracks, latest = [], {}
    for i in range(9):
        seen = set()
        for obj in scenario.get_tracked_objects_at_iteration(i).tracked_objects:
            if obj.tracked_object_type == TrackedObjectType.EGO:
                continue
            token = str(obj.track_token)
            if token in seen:
                raise ValueError(f"Duplicate recorded actor at iteration {i}: {token}")
            seen.add(token)
            box = obj.box
            xy = (np.array([box.center.x, box.center.y]) - [origin.x, origin.y]) @ rotation
            pose = [float(xy[0]), float(xy[1]), float(box.center.heading - theta)]
            if token not in latest or latest[token][0] != i-1:
                track = {"actor_id": token, "classes": [obj.tracked_object_type.name], "times_s": [], "poses": [], "sizes": []}
                tracks.append(track)
            else:
                track = latest[token][1]
            track["times_s"].append(i * .5)
            track["poses"].append(pose)
            track["sizes"].append([float(box.length), float(box.width)])
            latest[token] = (i, track)
    params = state.car_footprint.vehicle_parameters
    geometry = {"x_min": float(-params.rear_length), "x_max": float(params.front_length),
                "y_min": float(-params.width / 2), "y_max": float(params.width / 2)}
    world = {"version": 1, "contract": WORLD_CONTRACT, "horizon_s": 4., "tracks": tracks,
             "source": source, "origin": [float(origin.x), float(origin.y), float(theta)], "geometry": geometry,
             "query_radius_m": radius, "map_polygons_wkb": [], "map_status": "unavailable",
             "initial_velocity": [float(state.dynamic_car_state.rear_axle_velocity_2d.x), float(state.dynamic_car_state.rear_axle_velocity_2d.y)],
             "initial_acceleration": [float(state.dynamic_car_state.rear_axle_acceleration_2d.x), float(state.dynamic_car_state.rear_axle_acceleration_2d.y)]}
    try:
        api = scenario.map_api
        layers = [getattr(SemanticMapLayer, n) for n in WORLD_CONTRACT["map_layers"]]
        supported = api.get_available_map_objects()
        selected = [layer for layer in layers if layer in supported]
        world["unpopulated_map_layers"] = [layer.name for layer in layers if layer not in supported]
        objects = api.get_proximal_map_objects(Point2D(origin.x, origin.y), radius, selected)
        world["map_polygons_wkb"] = [obj.polygon.wkb_hex for layer in selected for obj in objects[layer]]
        if not world["map_polygons_wkb"]:
            raise ValueError("No drivable polygons in map coverage")
        world["map_status"] = "available"
    except Exception as exc:
        world["map_error"] = f"{type(exc).__name__}: {exc}"
    return world


class RawWorlds(SceneRepository):
    def load(self, token, radius=100.):
        path, _ = self.index[token]
        stat = Path(path).stat()
        identity = digest({"token": token, "log": [path, stat.st_size, stat.st_mtime_ns],
                           "map": self.map_identity, "radius": radius, "contract": WORLD_CONTRACT})
        cache = Path(self.config["work_root"]) / "worlds" / f"{identity}.json"
        with locked(cache.with_suffix(".lock")):
            stored = read_json(cache)
            if stored and stored.get("payload_sha256") == digest(stored["world"]):
                return stored["world"]
            scenario, source = self.scene(token)
            world = build_world(scenario, source, radius)
            # Do not permanently cache transient map failures.
            if world["map_status"] == "available":
                write_json(cache, {"world": world, "payload_sha256": digest(world)})
            return world
