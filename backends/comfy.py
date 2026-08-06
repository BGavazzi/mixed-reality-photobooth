import io
import json
import sys
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

import requests
from PIL import Image

from .base import GenerationBackend

# Windows consoles default to cp1252, which raises on most non-ASCII prompt
# text (accents, emoji, non-English words) — reconfigure to UTF-8 so a
# prompt containing them doesn't crash the request mid-generation.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_WORKFLOW_PATH = Path(__file__).parent.parent / "workflows" / "txt2img_api.json"
POSITIVE_PROMPT_NODE = "6"
SEED_NODE = "3"

DEFAULT_VIDEO_WORKFLOW_PATH = Path(__file__).parent.parent / "workflows" / "text_to_video_api.json"
VIDEO_POSITIVE_PROMPT_NODE = "6"
VIDEO_SEED_NODE = "72"  # SamplerCustom.noise_seed
VIDEO_EXTENSIONS = (".mp4", ".webm", ".mov")

DEFAULT_PHOTOSHOOT_BG_WORKFLOW_PATH = Path(__file__).parent.parent / "workflows" / "photoshoot_bg_api.json"
PHOTOSHOOT_SUBJECT_NODE = "2"
PHOTOSHOOT_MASK_NODE = "3"
PHOTOSHOOT_DEPTH_NODE = "4"
PHOTOSHOOT_POSITIVE_PROMPT_NODE = "7"
PHOTOSHOOT_CONTROLNET_NODE = "10"
PHOTOSHOOT_SEED_NODE = "11"


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

    def _upload_image(self, image: Image.Image, name_hint: str) -> str:
        """Uploads a PIL image to ComfyUI's /upload/image so a workflow's
        LoadImage/LoadImageMask nodes can reference it by filename."""
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)
        filename = f"{name_hint}_{uuid.uuid4().hex[:8]}.png"
        resp = requests.post(
            f"http://{self.server_address}/upload/image",
            files={"image": (filename, buf, "image/png")},
            data={"type": "input", "overwrite": "true"},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["name"]

    def queue_image_generation(self, prompt: str, client_id: str = None) -> tuple[str, int]:
        """Non-blocking counterpart to generate_image(), for callers (e.g.
        the web server) that want to relay ComfyUI's own progress/preview
        websocket events instead of polling to a final result. Returns
        (prompt_id, seed) -- the seed is handed back so callers can show a
        real generation provenance record, not just "the model made this"."""
        with open(self.workflow_path) as f:
            workflow = json.load(f)

        seed = int.from_bytes(uuid.uuid4().bytes[:4], "big")
        workflow[POSITIVE_PROMPT_NODE]["inputs"]["text"] = prompt
        workflow[SEED_NODE]["inputs"]["seed"] = seed

        payload = {"prompt": workflow, "client_id": client_id or self.client_id}
        resp = requests.post(f"http://{self.server_address}/prompt", json=payload, timeout=10)
        resp.raise_for_status()
        prompt_id = resp.json()["prompt_id"]
        print(f"[comfy] queued prompt_id={prompt_id} text={prompt!r}")
        return prompt_id, seed

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

    def get_result_image(self, prompt_id: str) -> Image.Image | None:
        """Non-blocking single check of /history for a finished prompt's
        output image, or None if it isn't there yet. Used by callers that
        already know (via the websocket) that execution just finished."""
        history = self._get_history(prompt_id)
        entry = history.get(prompt_id)
        if entry and entry.get("outputs"):
            for node_output in entry["outputs"].values():
                for image in node_output.get("images", []):
                    raw = self._get_output_bytes(image["filename"], image["subfolder"], image["type"])
                    return Image.open(io.BytesIO(raw)).convert("RGB")
        return None

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

    def queue_background_generation(
        self,
        subject_photo: Image.Image,
        background_mask: Image.Image,
        depth_map: Image.Image,
        prompt: str,
        controlnet_strength: float = 0.75,
        denoise: float = 0.85,
        workflow_path: Path = DEFAULT_PHOTOSHOOT_BG_WORKFLOW_PATH,
        client_id: str = None,
    ) -> tuple[str, int]:
        """Queues a background-only regeneration: subject_photo is used as
        the init image, background_mask (255=regenerate, 0=keep original)
        is the inpaint mask, depth_map conditions the ControlNet so the new
        environment respects the subject's actual pose/perspective. Returns
        (prompt_id, seed) — async by design so a caller (e.g. the web
        server) can relay ComfyUI's own websocket progress/preview events
        while it runs, instead of blocking like generate_image()."""
        with open(workflow_path) as f:
            workflow = json.load(f)

        subject_name = self._upload_image(subject_photo.convert("RGB"), "subject")
        mask_name = self._upload_image(background_mask.convert("L"), "bgmask")
        depth_name = self._upload_image(depth_map.convert("RGB"), "depth")

        seed = int.from_bytes(uuid.uuid4().bytes[:4], "big")
        workflow[PHOTOSHOOT_SUBJECT_NODE]["inputs"]["image"] = subject_name
        workflow[PHOTOSHOOT_MASK_NODE]["inputs"]["image"] = mask_name
        workflow[PHOTOSHOOT_DEPTH_NODE]["inputs"]["image"] = depth_name
        workflow[PHOTOSHOOT_POSITIVE_PROMPT_NODE]["inputs"]["text"] = prompt
        workflow[PHOTOSHOOT_CONTROLNET_NODE]["inputs"]["strength"] = controlnet_strength
        workflow[PHOTOSHOOT_SEED_NODE]["inputs"]["seed"] = seed
        workflow[PHOTOSHOOT_SEED_NODE]["inputs"]["denoise"] = denoise

        if client_id:
            payload = {"prompt": workflow, "client_id": client_id}
            resp = requests.post(f"http://{self.server_address}/prompt", json=payload, timeout=10)
            resp.raise_for_status()
            prompt_id = resp.json()["prompt_id"]
        else:
            prompt_id = self._queue_prompt(workflow)

        print(f"[comfy] queued background prompt_id={prompt_id} text={prompt!r}")
        return prompt_id, seed

    def generate_background(self, *args, timeout: float = 180.0, **kwargs) -> Image.Image:
        """Synchronous convenience wrapper around queue_background_generation
        for scripts/tests that don't need live progress."""
        prompt_id, _seed = self.queue_background_generation(*args, **kwargs)
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
                        return Image.open(io.BytesIO(raw)).convert("RGB")
            time.sleep(0.5)
        raise TimeoutError(f"ComfyUI did not finish prompt_id={prompt_id} within {timeout}s")
