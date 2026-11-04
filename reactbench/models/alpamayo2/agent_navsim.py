"""Three-camera NAVSIM adapter; images retain their recorded 2 Hz history."""
from ..alpamayo1_5.agent_navsim import load_camera_tensor
from .inference import Alpamayo2Inference


class Alpamayo2Agent(Alpamayo2Inference):
    camera_loader = staticmethod(load_camera_tensor)
