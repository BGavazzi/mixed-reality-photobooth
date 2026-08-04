import io
import os
import time

import requests
from PIL import Image

from .base import GenerationBackend


class RunwayBackend(GenerationBackend):
    """Runway's hosted API (https://docs.dev.runwayml.com).

    Requires RUNWAYML_API_SECRET in the environment and `pip install runwayml`.
    """

    def __init__(self, model: str = "gen4_image", ratio: str = "1024:1024"):
        try:
            from runwayml import RunwayML
        except ImportError as exc:
            raise RuntimeError("pip install runwayml to use the Runway backend") from exc

        api_key = os.environ.get("RUNWAYML_API_SECRET")
        if not api_key:
            raise RuntimeError("Set RUNWAYML_API_SECRET to use the Runway backend")

        self.client = RunwayML(api_key=api_key)
        self.model = model
        self.ratio = ratio

    def generate_image(self, prompt: str, timeout: float = 120.0) -> Image.Image:
        task = self.client.text_to_image.create(
            model=self.model, prompt_text=prompt, ratio=self.ratio
        )
        print(f"[runway] queued task id={task.id} text={prompt!r}")

        deadline = time.time() + timeout
        while time.time() < deadline:
            result = self.client.tasks.retrieve(task.id)
            if result.status == "SUCCEEDED":
                image_resp = requests.get(result.output[0], timeout=30)
                image_resp.raise_for_status()
                return Image.open(io.BytesIO(image_resp.content)).convert("RGBA")
            if result.status == "FAILED":
                raise RuntimeError(f"Runway task failed: {result.failure}")
            time.sleep(2)

        raise TimeoutError(f"Runway task {task.id} did not finish within {timeout}s")
