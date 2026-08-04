from .base import GenerationBackend
from .comfy import ComfyBackend
from .kling import KlingBackend
from .runway import RunwayBackend

BACKENDS = {
    "comfy": ComfyBackend,
    "runway": RunwayBackend,
    "kling": KlingBackend,
}

__all__ = ["GenerationBackend", "ComfyBackend", "RunwayBackend", "KlingBackend", "BACKENDS"]
