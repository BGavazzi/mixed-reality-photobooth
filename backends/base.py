from abc import ABC, abstractmethod

from PIL import Image


class GenerationBackend(ABC):
    """Anything that can turn a text prompt into an image.

    Swap backends freely (ComfyUI running locally, Runway, Kling, or
    whatever plug-and-play generation API comes next) — bridge.py only
    ever calls generate_image().
    """

    @abstractmethod
    def generate_image(self, prompt: str, timeout: float = 120.0) -> Image.Image:
        ...
