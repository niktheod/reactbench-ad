"""Six-camera PAI-AV adapter; model inference is shared with NAVSIM."""
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .inference import Alpamayo2Inference
from .paiav_loader import MODEL_CAMERAS


def load_camera_tensor(camera_paths, *, frames_per_camera=4):
    """Load NVIDIA's six trajectory cameras in ascending camera-ID order."""
    cameras, shape = [], None
    for camera, _ in MODEL_CAMERAS:
        paths = camera_paths.get(camera)
        if not isinstance(paths, list) or len(paths) != frames_per_camera:
            raise ValueError(f"Expected {frames_per_camera} paths for {camera!r}")
        frames = []
        for value in paths:
            path = Path(value).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"Camera image does not exist: {path}")
            with Image.open(path) as image:
                frame = torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.uint8).copy()).permute(2, 0, 1)
            if shape is not None and tuple(frame.shape) != shape:
                raise ValueError(f"All camera frames must share one shape; expected {shape}, got "
                                 f"{tuple(frame.shape)} for {path}")
            shape = tuple(frame.shape)
            frames.append(frame)
        cameras.append(torch.stack(frames))
    return torch.stack(cameras), torch.tensor([index for _, index in MODEL_CAMERAS], dtype=torch.int64)


class Alpamayo2Agent(Alpamayo2Inference):
    camera_loader = staticmethod(load_camera_tensor)
