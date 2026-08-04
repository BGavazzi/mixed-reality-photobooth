import io
import json
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

import requests
from PIL import Image

from .base import GenerationBackend

DEFAULT_WORKFLOW_PATH = Path(__file__).parent.parent / "workflows" / "txt2img_api.json"
POSITIVE_PROMPT_NODE = "6"
SEED_NODE = "3"

DEFAULT_VIDEO_WORKFLOW_PATH = Path(__file__).parent.parent / "workflows" / "text_to_video_api.json"
VIDEO_POSITIVE_PROMPT_NODE = "6"
VIDEO_SEED_NODE = "72"  # SamplerCustom.noise_seed
VIDEO_EXTENSIONS = (".mp4", ".webm", ".mov")


class ComfyBackend(GenerationBackend):
    """Local Stable Diffusion via a running ComfyUI instance's REST API."""

    def __init__(self, server_address: str = "127.0.0.1:8188", workflow_path: Path = DEFAULT_WORKFLOW_PATH):
        self.server_address = server_address
        self.workflow_path = Path(workflow_path)
        self.client_id = str(uuid.uuid4())

    def _queue_prompt(self, workflow: dict) -> str:
        payload = {"prompt": workflow, "client_id": self.client_id}
        resp = requests.post(f"http://{self.server_address}/prompt", json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()["prompt_id"]

    def _get_history(self, prompt_id: str) -> dict:
        resp = requests.get(f"http://{self.server_address}/history/{prompt_id}", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def _get_output_bytes(self, filename: str, subfolder: str, folder_type: str) -> bytes:
        params = urllib.parse.urlencode(
            {"filename": filename, "subfolder": subfolder, "type": folder_type}
        )
        with urllib.request.urlopen(f"http://{self.server_address}/view?{params}") as resp:
            return resp.read()

    def generate_image(self, prompt: str, timeout: float = 120.0) -> Image.Image:
        with open(self.workflow_path) as f:
            workflow = json.load(f)

        workflow[POSITIVE_PROMPT_NODE]["inputs"]["text"] = prompt
        workflow[SEED_NODE]["inputs"]["seed"] = int.from_bytes(uuid.uuid4().bytes[:4], "big")

        prompt_id = self._queue_prompt(workflow)
        print(f"[comfy] queued prompt_id={prompt_id} text={prompt!r}")

        deadline = time.time() + timeout
        while time.time() < deadline:
            history = self._get_history(prompt_id)
            entry = history.get(prompt_id)
            if entry and entry.get("outputs"):
                for node_output in entry["outputs"].values():
                    for image in node_output.get("images", []):
                        raw = self._get_output_bytes(
                            image["filename"], image["subfolder"], image["type"]
                        )
                        return Image.open(io.BytesIO(raw)).convert("RGBA")
            time.sleep(0.5)

        raise TimeoutError(f"ComfyUI did not finish prompt_id={prompt_id} within {timeout}s")

    def generate_video(self, prompt: str, timeout: float = 600.0,
                        workflow_path: Path = DEFAULT_VIDEO_WORKFLOW_PATH) -> bytes:
        """Text-to-video via LTXV. Returns raw video file bytes (mp4).

        ComfyUI's SaveVideo node reports its output the same way SaveImage
        does — under the "images" key in /history, just with an extra
        "animated": [true] flag alongside it — so this reuses the same
        polling shape as generate_image(), filtered to video filenames.
        """
        with open(workflow_path) as f:
            workflow = json.load(f)

        workflow[VIDEO_POSITIVE_PROMPT_NODE]["inputs"]["text"] = prompt
        workflow[VIDEO_SEED_NODE]["inputs"]["noise_seed"] = int.from_bytes(uuid.uuid4().bytes[:6], "big")

        prompt_id = self._queue_prompt(workflow)
        print(f"[comfy] queued video prompt_id={prompt_id} text={prompt!r}")

        deadline = time.time() + timeout
        while time.time() < deadline:
            history = self._get_history(prompt_id)
            entry = history.get(prompt_id)
            if entry and entry.get("outputs"):
                for node_output in entry["outputs"].values():
                    for item in node_output.get("images", []):
                        if item["filename"].lower().endswith(VIDEO_EXTENSIONS):
                            return self._get_output_bytes(
                                item["filename"], item["subfolder"], item["type"]
                            )
            time.sleep(1.0)

        raise TimeoutError(f"ComfyUI did not finish video prompt_id={prompt_id} within {timeout}s")
