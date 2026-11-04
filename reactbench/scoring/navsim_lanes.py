"""NAVSIM lane polygons for the stock PDMS "ego spans several lanes" test.

The polygons come from the navtest metric caches, i.e. the same ``PDMDrivableMap``
(100 m radius) that stock PDMS scores with. They are extracted once into plain
gzipped JSON under ``<work_root>/lanes/`` so scoring only needs shapely. Extraction
must run where the caches unpickle, i.e. with the NAVSIM version that built them
on ``PYTHONPATH`` (see the README). The raw worlds and their contract stay untouched.
"""
from __future__ import annotations

import argparse
import gzip
import json
from functools import lru_cache
from pathlib import Path

from .. import paths
from ..io import digest, file_digest, read_json, write_json

LANES_CONTRACT = {
    "version": 1,
    "source": "NAVSIM metric cache drivable_area_map (the PDMDrivableMap stock PDMS scores with)",
    "layers": ["LANE", "LANE_CONNECTOR"],
    "frame": "global map coordinates, like the raw-world drivable polygons",
    "use": "stock PDMS multiple-lanes test for lateral contacts",
}
LAYERS = ("LANE", "LANE_CONNECTOR")
POSE_TOLERANCE = (1e-3, 1e-4)  # m, rad: sidecar and raw world must describe the same ego pose


def lanes_path(work_root, token):
    return Path(work_root) / "lanes" / f"{token}.json.gz"


def _read(path):
    with gzip.open(path, "rt") as handle:
        return json.load(handle)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.publishing")
    try:
        with gzip.open(staging, "wt") as handle:
            json.dump(value, handle, allow_nan=False)
        staging.replace(path)
    finally:
        staging.unlink(missing_ok=True)


def extract_one(job):
    """Unpickle one metric cache (navsim env only) and write its lanes sidecar."""
    import lzma
    import pickle
    token, cache_path, work_root = job
    source = {"path": str(cache_path), "sha256": file_digest(cache_path)}
    target = lanes_path(work_root, token)
    if target.is_file():
        stored = _read(target)
        body = {k: v for k, v in stored.items() if k != "payload_sha256"}
        if stored.get("source") == source and stored.get("payload_sha256") == digest(body):
            return token, "reused", len(stored["lanes"])
    with lzma.open(cache_path, "rb") as handle:
        cache = pickle.load(handle)
    area = cache.drivable_area_map
    lanes = []
    # Index the geometries directly: map ids may repeat across layers, so token lookup is ambiguous.
    for index, (layer, geometry) in enumerate(zip(area.map_types, area._geometries)):
        name = getattr(layer, "name", str(layer).split(".")[-1])
        if name in LAYERS:
            lanes.append({"id": str(area.tokens[index]), "layer": name, "wkb": geometry.wkb_hex})
    pose = cache.ego_state.rear_axle
    body = {"version": 1, "token": token, "contract": LANES_CONTRACT, "source": source,
            "ego_rear_axle": [float(pose.x), float(pose.y), float(pose.heading)], "lanes": lanes}
    _write(target, {**body, "payload_sha256": digest(body)})
    return token, "extracted", len(lanes)


def extract(tokens, metric_cache_root, work_root, workers=1):
    from concurrent.futures import ProcessPoolExecutor
    index = {}
    for path in Path(metric_cache_root).glob("*/*/*/metric_cache.pkl"):
        token = path.parent.name
        if token in index:
            raise ValueError(f"Duplicate metric cache for {token}: {index[token]}, {path}")
        index[token] = path
    jobs = [(t, index[t], str(work_root)) for t in tokens if t in index]
    missing = sorted(t for t in tokens if t not in index)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(extract_one, jobs, chunksize=4))
    report = {"version": 1, "contract": LANES_CONTRACT, "metric_cache_root": str(metric_cache_root),
              "requested": len(tokens), "missing_metric_cache": missing,
              "extracted": sum(r[1] == "extracted" for r in results),
              "reused": sum(r[1] == "reused" for r in results),
              "tokens_without_lanes": sorted(r[0] for r in results if r[2] == 0)}
    write_json(Path(work_root) / "lanes" / "extraction_report.json", report)
    return report


@lru_cache(maxsize=64)
def _load(path, size, modified_ns):
    from shapely import wkb
    stored = _read(path)
    body = {k: v for k, v in stored.items() if k != "payload_sha256"}
    if stored.get("contract") != LANES_CONTRACT or stored.get("payload_sha256") != digest(body):
        raise ValueError(f"Corrupt or incompatible lanes sidecar: {path}")
    return {"token": stored["token"], "ego_rear_axle": stored["ego_rear_axle"], "source": stored["source"],
            "polygons": [wkb.loads(lane["wkb"], hex=True) for lane in stored["lanes"]],
            "ids": [lane["id"] for lane in stored["lanes"]]}


def load_lanes(token, work_root, origin=None):
    """Lane polygons for one scene; ``origin`` (raw-world ego pose) must match the cached ego."""
    path = lanes_path(work_root, token)
    if not path.is_file():
        raise FileNotFoundError(f"Missing NAVSIM lanes sidecar for {token}: {path}")
    stat = path.stat()
    lanes = _load(str(path), stat.st_size, stat.st_mtime_ns)
    if origin is not None:
        import math
        x, y, heading = lanes["ego_rear_axle"]
        turn = abs(math.atan2(math.sin(heading - origin[2]), math.cos(heading - origin[2])))
        if math.hypot(x - origin[0], y - origin[1]) > POSE_TOLERANCE[0] or turn > POSE_TOLERANCE[1]:
            raise ValueError(f"Lanes sidecar and raw world disagree on the ego pose for {token}")
    return lanes


def lanes_identity(work_root):
    """Provenance of the extraction (report hash), recorded with every scored artifact."""
    report = Path(work_root) / "lanes" / "extraction_report.json"
    return {"contract": LANES_CONTRACT, "report_sha256": file_digest(report) if report.is_file() else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tokens-from", type=Path, required=True,
                        help="NAVSIM annotation list (JSON list of records with a token)")
    parser.add_argument("--metric-cache", type=Path, default=None,
                        help="Default: navsim_metric_cache in configs/paths.yaml")
    parser.add_argument("--work-root", type=Path, default=None, help="Default: <outputs>/work/navsim")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    args.metric_cache = args.metric_cache or paths.get("navsim_metric_cache")
    args.work_root = args.work_root or paths.outputs("work", "navsim")
    tokens = sorted({str(r["token"]) for r in read_json(args.tokens_from)})
    report = extract(tokens, args.metric_cache, args.work_root, args.workers)
    print(json.dumps({k: (v if not isinstance(v, list) else len(v)) for k, v in report.items()
                      if k != "contract"}, indent=1))


if __name__ == "__main__":
    main()
