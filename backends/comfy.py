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

    def _get_image_bytes(self, filename: str, subfolder: str, folder_type: str) -> bytes:
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
                        raw = self._get_image_bytes(
                            image["filename"], image["subfolder"], image["type"]
                        )
                        return Image.open(io.BytesIO(raw)).convert("RGBA")
            time.sleep(0.5)

        raise TimeoutError(f"ComfyUI did not finish prompt_id={prompt_id} within {timeout}s")
