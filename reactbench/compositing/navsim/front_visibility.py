"""Current front-camera admission check, shared by preflight and compositing.

This measures image cropping of a projected bounding box, not visible silhouette
or occlusion by existing scene objects. Boxes crossing the camera near plane are
rejected conservatively because their corner projection is not a bounded box.
"""

from functools import lru_cache
import math

import numpy as np
from PIL import Image

VERSION = "projected_bbox_image_fraction_v1"
CAMERA = "CAM_F0"
FRAME_INDEX = 3
NEAR_M = 0.1


def fraction_argument(value):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("--min-front-visible-fraction must be finite and in [0, 1]")
    return value


def add_arguments(parser, default=0.25):
    parser.add_argument(
        "--min-front-visible-fraction", type=fraction_argument, default=default,
        help="Minimum projected bbox area fraction inside current CAM_F0; "
             "0 disables the check. Does not measure scene occlusion.",
    )


def configuration(args):
    minimum = fraction_argument(getattr(args, "min_front_visible_fraction", 0.0))
    return dict(metric=VERSION, camera=CAMERA, frame_index=FRAME_INDEX,
                min_fraction=minimum, near_plane_m=NEAR_M,
                occlusion_checked=False)


def projected_fraction(uv, depth, width, height, minimum):
    """Intersect the full distorted 2D bounding rectangle with the image."""
    minimum = fraction_argument(minimum)
    uv, depth = np.asarray(uv, dtype=float), np.asarray(depth, dtype=float)
    result = dict(metric=VERSION, camera=CAMERA, frame_index=FRAME_INDEX,
                  min_fraction=minimum, image_size_px=[int(width), int(height)],
                  projected_bounds_px=None, projected_area_px2=0.0,
                  in_image_area_px2=0.0, fraction=0.0, valid_projection=False,
                  passed=False, occlusion_checked=False)
    if (width <= 0 or height <= 0 or uv.ndim != 2 or uv.shape[1] != 2
            or len(uv) == 0 or depth.shape != (len(uv),)
            or not np.isfinite(uv).all() or not np.isfinite(depth).all()
            or not np.all(depth > NEAR_M)):
        result["projection_failure"] = "invalid_or_near_plane_projection"
        return result
    x0, y0 = uv.min(axis=0)
    x1, y1 = uv.max(axis=0)
    area = float((x1 - x0) * (y1 - y0))
    if not math.isfinite(area) or area <= 0:
        result["projection_failure"] = "degenerate_projected_bbox"
        return result
    clipped = float(max(0.0, min(x1, width) - max(x0, 0.0))
                    * max(0.0, min(y1, height) - max(y0, 0.0)))
    fraction = min(1.0, max(0.0, clipped / area))
    result.update(projected_bounds_px=[float(x0), float(y0), float(x1), float(y1)],
                  projected_area_px2=area, in_image_area_px2=clipped,
                  fraction=fraction, valid_projection=True,
                  passed=fraction >= minimum)
    return result


@lru_cache(maxsize=128)
def image_size(path):
    with Image.open(path) as source:
        return source.size


def assess_current(args, fast, frames, bbox_lidar):
    """Return None for legacy runs; otherwise inspect exactly history frame 3."""
    cfg = configuration(args)
    if cfg["min_fraction"] == 0:
        return None
    frame = frames[FRAME_INDEX]
    path = fast.frame_source_path(args.data_root, args.split, frame, CAMERA)
    width, height = image_size(str(path))
    uv, depth = fast.geometry.project_distorted(bbox_lidar, frame["cams"][CAMERA])
    return projected_fraction(uv, depth, width, height, cfg["min_fraction"])


def failure(result):
    if result is None or result["passed"]:
        return None
    return dict(
        code="insufficient_front_visibility",
        message=(f"Current {CAMERA} projected bbox fraction {result['fraction']:.6f} "
                 f"is below {result['min_fraction']:.6f} or projection is invalid"),
        front_visibility=result,
    )


def require(result, fast):
    rejected = failure(result)
    if rejected:
        raise fast.TokenFailure(rejected["code"], rejected["message"])


def check_candidate(args, fast, frames, states, candidate):
    """Attach the shared check to a pure preflight candidate before admission."""
    result = assess_current(args, fast, frames, states[FRAME_INDEX]["bbox_lidar"])
    if result is not None:
        candidate["front_visibility"] = result
        rejected = failure(result)
        if rejected:
            candidate["failures"].append(rejected)
        elif CAMERA not in states[FRAME_INDEX]["visible_cameras"]:
            # An intersecting box may have no corner inside the image.
            states[FRAME_INDEX]["visible_cameras"].append(CAMERA)
            cameras = states[FRAME_INDEX].get("camera_visibility", {})
            if CAMERA in cameras:
                cameras[CAMERA]["visible"] = True
    return result
