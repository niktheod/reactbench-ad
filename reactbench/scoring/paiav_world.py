"""ReactPAI background worlds: recorded (non-reactive) PAI-AV obstacle tracks in the current ego frame."""
from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation, Slerp

from ..io import digest, file_digest, read_json, write_json

FEATURE = "obstacle.offline"
WORLD_VERSION = 1

# Reviewed self-annotation, not a proximity/initial-overlap filter. Keep this
# tied to the exact source bytes so a later annotation revision is re-reviewed.
ANNOTATION_TRACK_EXCLUSIONS = ({
    "repo_id": "nvidia/PhysicalAI-Autonomous-Vehicles",
    "revision": "b719eea7f0a63619ef51ec7f54178af0937ef050",
    "token": "82b52ac1-5d4e-4b88-a3d4-84be168fb11a",
    "obstacle_member_sha256": "809fc4d3095fcadceabd1752def8dac730e77b0efa28e2f748a4e7bed61b893e",
    "actor_id": "250",
    "reason": "Ego self-annotation: automobile footprint overlaps and follows recorded ego throughout 0–4 s; visually reviewed.",
},)


def annotation_track_exclusions(provenance):
    keys = ("repo_id", "revision", "token", "obstacle_member_sha256")
    return [dict(entry) for entry in ANNOTATION_TRACK_EXCLUSIONS
            if all(provenance.get(key) == entry[key] for key in keys)]


def apply_annotation_track_exclusions(world, exclusions):
    """Remove only reviewed source-specific tracks and disclose applied corrections."""
    ids = {entry["actor_id"] for entry in exclusions}
    removed = [t for t in world["tracks"] if t["actor_id"] in ids]
    world["tracks"] = [t for t in world["tracks"] if t["actor_id"] not in ids]
    world["coverage"].update(
        track_count=len(world["tracks"]),
        partial_lifetime_track_count=sum(t["times_s"][0] > 0 or t["times_s"][-1] < world["horizon_s"]
                                         for t in world["tracks"]),
        excluded_annotation_track_count=len(removed),
        annotation_track_exclusions=[entry for entry in exclusions
                                     if any(t["actor_id"] == entry["actor_id"] for t in removed)])


def pinned_dataset(manifest):
    dataset = manifest["dataset"]
    revision = dataset.get("catalog_hub_commit", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Selection manifest must pin catalog_hub_commit to a commit SHA")
    return dataset["repo_id"], revision


def checked_catalog(manifest, name, data_root):
    source = manifest["dataset"]["catalog_inputs"][name]
    path = Path(data_root) / name
    if file_digest(path) != source["sha256"]:
        raise ValueError(f"Catalog checksum mismatch: {path}")
    return path


def resource_plan(manifest_path, data_root, obstacle_root, tokens=None):
    """Read-only resource inventory; obey the pinned catalog's path/member layout."""
    manifest = read_json(Path(manifest_path))
    repo, revision = pinned_dataset(manifest)
    features = pd.read_csv(checked_catalog(manifest, "features.csv", data_root)).set_index("feature")
    feature = features.loc[FEATURE]
    index = pd.read_parquet(checked_catalog(manifest, "clip_index.parquet", data_root))
    presence = pd.read_parquet(checked_catalog(manifest, "metadata/feature_presence.parquet", data_root),
                               columns=[FEATURE])
    selected = list(manifest["samples"]["clip_ids"] if tokens is None else tokens)
    if len(set(selected)) != len(selected):
        raise ValueError("Duplicate requested tokens")
    if set(selected) - set(manifest["samples"]["clip_ids"]):
        raise ValueError("Requested tokens are not in the selected manifest")
    groups = {}
    for token in selected:
        if not bool(presence.loc[token, FEATURE]):
            raise ValueError(f"Obstacle labels unavailable in catalog: {token}")
        chunk = int(index.loc[token, "chunk"])
        groups.setdefault(chunk, []).append(token)
    resources = []
    for chunk, ids in sorted(groups.items()):
        relative = str(feature["chunk_path"]).format(chunk_id=chunk)
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("Unsafe resource path in feature catalog")
        template = json.loads(feature["clip_files_in_zip"])[FEATURE]
        resources.append({"chunk_id": chunk, "relative_path": relative,
                          "path": str(Path(obstacle_root).resolve() / relative),
                          "members": {token: template.format(clip_id=token) for token in ids}})
    return {"repo_id": repo, "revision": revision, "token_count": len(selected),
            "selection_manifest_sha256": file_digest(manifest_path), "resources": resources}


@lru_cache(maxsize=256)
def _hash_at_stat(path, size, mtime_ns):
    return file_digest(Path(path))


def verified_archive(path, repo_id, revision):
    path = Path(path)
    receipt = read_json(path.with_suffix(path.suffix + ".provenance.json"))
    if not path.is_file() or receipt is None:
        raise FileNotFoundError(f"Missing obstacle archive or acquisition receipt: {path}")
    stat = path.stat()
    checksum = _hash_at_stat(str(path), stat.st_size, stat.st_mtime_ns)
    if (receipt.get("repo_id"), receipt.get("revision"), receipt.get("sha256")) != (
            repo_id, revision, checksum):
        raise ValueError(f"Obstacle archive provenance/checksum mismatch: {path}")
    return checksum


def pose_matrices(egomotion, queries):
    """Interpolate offline world-from-rig poses, without extrapolation."""
    required = ["timestamp", "x", "y", "z", "qx", "qy", "qz", "qw"]
    values = egomotion[required].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite offline egomotion")
    values = values[np.argsort(values[:, 0])]
    times = values[:, 0]
    queries = np.asarray(queries, dtype=float)
    if len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError("Offline egomotion timestamps must be unique and increasing")
    if queries.min() < times[0] or queries.max() > times[-1]:
        raise ValueError("Offline egomotion does not cover obstacle reference timestamps")
    quaternions = values[:, 4:8]
    if not np.allclose(np.linalg.norm(quaternions, axis=1), 1, atol=1e-3):
        raise ValueError("Invalid offline egomotion quaternions")
    result = np.broadcast_to(np.eye(4), (len(queries), 4, 4)).copy()
    result[:, :3, :3] = Slerp(times, Rotation.from_quat(quaternions))(queries).as_matrix()
    result[:, :3, 3] = np.column_stack([np.interp(queries, times, values[:, k]) for k in (1, 2, 3)])
    return result


def build_background_world(obstacles, egomotion, current_timestamp_us, *, horizon_s=4.0,
                           max_track_gap_s=0.25):
    """Convert per-row rig-frame boxes to the prediction's current physical rig.

    obstacle.offline v2 rows name their own reference_frame_timestamp_us. They
    must NOT be interpreted as boxes already in a common world frame. Relative
    offline poses register them into the same current rig used by native ego
    predictions, without mixing online and offline world-coordinate origins.
    Empty annotations and tracks entering/leaving the window are valid; object
    timestamps do not establish whether annotation coverage is complete.
    """
    required = ["timestamp_us", "track_id", "center_x", "center_y", "center_z",
                "size_x", "size_y", "size_z", "orientation_x", "orientation_y",
                "orientation_z", "orientation_w", "label_class", "source",
                "reference_frame", "reference_frame_timestamp_us"]
    if set(required) - set(obstacles):
        raise ValueError(f"Missing obstacle columns: {sorted(set(required) - set(obstacles))}")
    if not np.isfinite(obstacles.timestamp_us.to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite obstacle timestamps")
    t0 = int(current_timestamp_us)
    end = t0 + round(horizon_s * 1e6)
    initial = pose_matrices(egomotion, [t0])[0]
    world = {"version": WORLD_VERSION, "current_timestamp_us": t0, "horizon_s": horizon_s,
             "tracks": [], "coverage": {"policy": "annotated_lifetime", "track_count": 0,
             "partial_lifetime_track_count": 0, "max_track_gap_s": max_track_gap_s,
             "scope": "supplied object annotations; no extrapolation beyond track lifetimes"}}
    margin = round(max_track_gap_s * 1e6)
    frame = obstacles.loc[(obstacles.timestamp_us >= t0 - margin) &
                          (obstacles.timestamp_us <= end + margin), required].copy()
    if frame.empty:
        return world
    if frame[required].isna().any().any():
        raise ValueError("Missing obstacle annotation values in evaluation window")
    if set(frame.reference_frame) != {"rig"}:
        raise ValueError(f"Unsupported obstacle reference frame: {set(frame.reference_frame)}")
    if set(frame.source) != {"scene:obstacles:autolabels:v2"}:
        raise ValueError(f"Unsupported obstacle label source: {set(frame.source)}")
    numeric = [c for c in required if c not in ("track_id", "label_class", "source", "reference_frame")]
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite obstacle geometry or timestamps")
    if (frame[["size_x", "size_y", "size_z"]] <= 0).any().any():
        raise ValueError("Nonpositive obstacle dimensions")
    refs = frame.reference_frame_timestamp_us.to_numpy(dtype=float)
    ref_poses = pose_matrices(egomotion, refs)
    current_from_refs = np.linalg.inv(initial)[None] @ ref_poses
    centers = frame[["center_x", "center_y", "center_z"]].to_numpy()
    centers = np.einsum("nij,nj->ni", current_from_refs[:, :3, :3], centers) + current_from_refs[:, :3, 3]
    q = frame[["orientation_x", "orientation_y", "orientation_z", "orientation_w"]].to_numpy()
    if not np.allclose(np.linalg.norm(q, axis=1), 1, atol=1e-3):
        raise ValueError("Invalid obstacle quaternions")
    rotations = current_from_refs[:, :3, :3] @ Rotation.from_quat(q).as_matrix()
    frame["x"], frame["y"] = centers[:, 0], centers[:, 1]
    frame["yaw"] = np.arctan2(rotations[:, 1, 0], rotations[:, 0, 0])
    tracks, partial = world["tracks"], 0
    for track_id, group in frame.groupby("track_id", sort=True):
        group = group.sort_values("timestamp_us")
        times = (group.timestamp_us.to_numpy(dtype=float) - t0) / 1e6
        if times[-1] < 0 or times[0] > horizon_s:
            continue
        if len(times) < 2 or np.any(np.diff(times) <= 0):
            raise ValueError(f"Track {track_id}: insufficient or duplicate timestamps")
        relevant = (times[:-1] < horizon_s) & (times[1:] > 0)
        if np.any(np.diff(times)[relevant] > max_track_gap_s + 1e-9):
            raise ValueError(f"Track {track_id}: annotation gap exceeds {max_track_gap_s}s")
        partial += int(times[0] > 0 or times[-1] < horizon_s)
        tracks.append({"actor_id": str(track_id), "classes": sorted(set(group.label_class)),
                       "times_s": times.tolist(),
                       "poses": group[["x", "y", "yaw"]].to_numpy().tolist(),
                       "sizes": group[["size_x", "size_y"]].to_numpy().tolist()})
    world["coverage"].update(track_count=len(tracks), partial_lifetime_track_count=partial)
    return world


class BackgroundWorlds:
    def __init__(self, config):
        self.config = config
        self.archives = {}

    def _member(self, path, name):
        path = Path(path)
        if path not in self.archives:
            self.archives[path] = zipfile.ZipFile(path)
        return self.archives[path].read(name)

    def load(self, context):
        cfg = self.config
        path = Path(cfg["obstacle_root"]) / context["obstacle_relative_path"]
        archive_sha = verified_archive(path, context["repo_id"], context["revision"])
        obstacle_bytes = self._member(path, context["obstacle_member"])
        ego_bytes = self._member(context["offline_egomotion_path"], context["offline_egomotion_member"])
        provenance = {**context, "archive_sha256": archive_sha,
                      "obstacle_member_sha256": hashlib.sha256(obstacle_bytes).hexdigest(),
                      "offline_egomotion_member_sha256": hashlib.sha256(ego_bytes).hexdigest(),
                      "world_implementation_sha256": file_digest(__file__),
                      "horizon_s": cfg["horizon_s"], "max_track_gap_s": cfg["max_track_gap_s"]}
        exclusions = annotation_track_exclusions(provenance)
        provenance["annotation_track_exclusions"] = exclusions
        fingerprint = digest(provenance)
        cache = Path(cfg["work_root"]) / "worlds" / f"{fingerprint}.json"
        value = read_json(cache)
        if value and value.get("fingerprint") == fingerprint and value.get("payload_sha256") == digest(value["world"]):
            return value["world"], fingerprint
        world = build_background_world(pd.read_parquet(io.BytesIO(obstacle_bytes)),
                    pd.read_parquet(io.BytesIO(ego_bytes)), context["current_timestamp_us"],
                    horizon_s=cfg["horizon_s"], max_track_gap_s=cfg["max_track_gap_s"])
        apply_annotation_track_exclusions(world, exclusions)
        world["provenance"] = provenance
        write_json(cache, {"fingerprint": fingerprint, "world": world, "payload_sha256": digest(world)})
        return world, fingerprint

    def close(self):
        for archive in self.archives.values():
            archive.close()
        self.archives.clear()
