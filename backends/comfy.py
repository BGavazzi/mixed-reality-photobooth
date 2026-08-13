import io
import json
import os
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Callable

import requests
from PIL import Image

from .base import GenerationBackend

import config
import obs
import resilience
import workflow_graph
from workflow_graph import (
    CONTROLNET,
    DEPTH_IMAGE,
    MASK_IMAGE,
    NEGATIVE_PROMPT,
    POSITIVE_PROMPT,
    SAMPLER,
    SUBJECT_IMAGE,
)

# Node ids used to live here as constants -- POSITIVE_PROMPT_NODE = "6" and so
# on -- one set per workflow. They were an artifact of a particular export
# from ComfyUI's UI, so re-exporting a workflow silently repointed them at the
# wrong nodes. They are now resolved from each graph's own wiring at load
# time; see workflow_graph.py for why that is worth the extra code.
DEFAULT_WORKFLOW_PATH = Path(__file__).parent.parent / "workflows" / "txt2img_api.json"
DEFAULT_VIDEO_WORKFLOW_PATH = Path(__file__).parent.parent / "workflows" / "text_to_video_api.json"
DEFAULT_PHOTOSHOOT_BG_WORKFLOW_PATH = Path(__file__).parent.parent / "workflows" / "photoshoot_bg_api.json"

VIDEO_EXTENSIONS = (".mp4", ".webm", ".mov")

# What each entry point cannot run without. Declared per call site rather than
# globally: txt2img legitimately has no ControlNet or inpaint mask and should
# not be held to the photoshoot graph's shape.
TXT2IMG_ROLES = (POSITIVE_PROMPT, SAMPLER)
VIDEO_ROLES = (POSITIVE_PROMPT, SAMPLER)
PHOTOSHOOT_ROLES = (POSITIVE_PROMPT, NEGATIVE_PROMPT, SUBJECT_IMAGE,
                    MASK_IMAGE, DEPTH_IMAGE, CONTROLNET, SAMPLER)


class ComfyBackend(GenerationBackend):
    """Local Stable Diffusion via a running ComfyUI instance's REST API."""

    def __init__(self, server_address: str = "127.0.0.1:8188", workflow_path: Path = DEFAULT_WORKFLOW_PATH,
                 retry_policy: resilience.RetryPolicy | None = None,
                 breaker: resilience.CircuitBreaker | None = None):
        self.server_address = server_address
        self.workflow_path = Path(workflow_path)
        self.client_id = str(uuid.uuid4())
        self.retry_policy = retry_policy or resilience.RetryPolicy(
            attempts=config.env_int(
                "COMFY_RETRY_ATTEMPTS", 4, "how many times a failed ComfyUI call is retried",
                minimum=1, maximum=20),
            budget_seconds=config.env_float(
                "COMFY_RETRY_BUDGET", 20.0, "wall-clock seconds the retry ladder may spend",
                minimum=0.1, maximum=600),
        )
        # One breaker per backend instance, shared across every call type:
        # /upload/image failing and /prompt failing are the same news about the
        # same box, and counting them separately would triple the time it takes
        # to notice ComfyUI is gone.
        self.breaker = breaker if breaker is not None else resilience.CircuitBreaker(
            failure_threshold=config.env_int(
                "COMFY_BREAKER_THRESHOLD", 5, "consecutive failures before the breaker opens",
                minimum=1, maximum=100),
            recovery_seconds=config.env_float(
                "COMFY_BREAKER_RECOVERY", 30.0, "seconds an open breaker waits before probing",
                minimum=1, maximum=3600),
        )

    def _log_attempt(self, label: str):
        def log(attempt: int, exc: BaseException, delay: float | None):
            # Attributed to whichever photo is being submitted, without this
            # module knowing that photos exist: the queue worker bound the id
            # before calling in, and the context follows across the thread hop.
            obs.log("comfy", "call failed", call=label, attempt=attempt,
                    verdict=resilience.classify(exc).value, error=repr(exc),
                    next=(f"retry in {delay:.1f}s" if delay is not None else "give up"))
        return log

    def _call(self, fn, *, label: str, reconcile=None):
        return resilience.call(
            fn, policy=self.retry_policy, breaker=self.breaker,
            reconcile=reconcile, label=label, on_attempt=self._log_attempt(label),
        )

    def _queue_prompt(self, workflow: dict, client_id: str | None = None,
                       prompt_id: str | None = None) -> str:
        """Submits a workflow to /prompt and returns the prompt_id ComfyUI
        echoes back.

        prompt_id: pass your own to know it *before* ComfyUI would ever emit
        an event for it (confirmed empirically: /prompt honors a
        client-supplied prompt_id and echoes it back unchanged). That lets a
        caller register bookkeeping -- web_server.py's JOBS dict -- before
        submitting, closing the race where an `executing` event for a
        same-instant-queued job arrives before the caller's own tracking
        exists.

        It also turns out to be what makes this call safely retryable. Queueing
        work is the one call here with a side effect, so a read timeout is
        genuinely ambiguous -- ComfyUI may already be rendering. Because *we*
        chose the id, we can just go and look (`_prompt_landed`) instead of
        guessing, which is why a duplicate generation is not one of the
        outcomes here.
        """
        payload = {"prompt": workflow, "client_id": client_id or self.client_id}
        if prompt_id:
            payload["prompt_id"] = prompt_id

        def submit():
            resp = requests.post(f"http://{self.server_address}/prompt", json=payload, timeout=10)
            resp.raise_for_status()
            return resp.json()["prompt_id"]

        return self._call(
            submit, label="POST /prompt",
            # Only meaningful when the caller supplied an id; without one there
            # is nothing to look for, so an ambiguous failure stays ambiguous
            # and is retried (accepting the small duplicate risk that the
            # caller opted into by not passing an id).
            reconcile=(lambda: self._prompt_landed(prompt_id)) if prompt_id else None,
        )

    def _prompt_landed(self, prompt_id: str) -> str | None:
        """Did a prompt we may or may not have submitted actually reach the
        queue? Returns the prompt_id if so, None if not.

        Checks the running/pending queue as well as history, because a job that
        landed one second ago is in neither history nor finished -- looking only
        at /history would report "no" for the most likely case and cause exactly
        the duplicate submission this exists to prevent.
        """
        try:
            if self._get_history(prompt_id).get(prompt_id):
                return prompt_id
            resp = requests.get(f"http://{self.server_address}/queue", timeout=5)
            resp.raise_for_status()
            queue = resp.json()
            for bucket in ("queue_running", "queue_pending"):
                for entry in queue.get(bucket, []):
                    # Entries are [number, prompt_id, prompt, extra, outputs].
                    if len(entry) > 1 and entry[1] == prompt_id:
                        return prompt_id
        except Exception as exc:              # noqa: BLE001 -- best effort by design
            obs.log("comfy", "could not reconcile", prompt_id=prompt_id, error=repr(exc))
        return None

    def _get_history(self, prompt_id: str) -> dict:
        def fetch():
            resp = requests.get(f"http://{self.server_address}/history/{prompt_id}", timeout=10)
            resp.raise_for_status()
            return resp.json()

        return self._call(fetch, label="GET /history")

    def _get_output_bytes(self, filename: str, subfolder: str, folder_type: str) -> bytes:
        params = urllib.parse.urlencode(
            {"filename": filename, "subfolder": subfolder, "type": folder_type}
        )

        def fetch():
            # The timeout is not optional: urlopen without one blocks forever
            # on a half-open socket, which is how a single stalled download
            # used to be able to occupy a queue worker for the life of the
            # process.
            with urllib.request.urlopen(
                    f"http://{self.server_address}/view?{params}", timeout=30) as resp:
                return resp.read()

        return self._call(fetch, label="GET /view")

    def _iter_outputs(self, prompt_id: str):
        """Yields every output item ComfyUI recorded for a finished prompt.

        `SaveImage` and `SaveVideo` both report under the same "images" key
        (the video one just adds `"animated": true`), so stills and video
        share this one shape.
        """
        entry = self._get_history(prompt_id).get(prompt_id)
        if not entry or not entry.get("outputs"):
            return
        for node_output in entry["outputs"].values():
            yield from node_output.get("images", [])

    def _poll_for_output(self, prompt_id: str, extract: Callable, timeout: float,
                          interval: float, label: str):
        """Blocks until `extract` returns something non-None for one of the
        prompt's outputs, or `timeout` elapses.

        This poll loop existed in four near-identical copies (stills, video,
        background, and the one-shot check), each with its own subtly
        different nesting of the same `history -> outputs -> images` walk.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            for item in self._iter_outputs(prompt_id):
                result = extract(item)
                if result is not None:
                    return result
            time.sleep(interval)
        raise TimeoutError(f"ComfyUI did not finish {label} prompt_id={prompt_id} within {timeout}s")

    def _download_image(self, item: dict, mode: str = "RGBA") -> Image.Image:
        raw = self._get_output_bytes(item["filename"], item["subfolder"], item["type"])
        return Image.open(io.BytesIO(raw)).convert(mode)

    def _upload_image(self, image: Image.Image, name_hint: str) -> str:
        """Uploads a PIL image to ComfyUI's /upload/image so a workflow's
        LoadImage/LoadImageMask nodes can reference it by filename."""
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)
        filename = f"{name_hint}_{uuid.uuid4().hex[:8]}.png"

        def upload():
            # buf is rewound per attempt: a retry of a partially-consumed
            # stream uploads a truncated PNG, which ComfyUI accepts and then
            # fails on much later, inside the graph, as an unrelated-looking
            # decode error.
            buf.seek(0)
            resp = requests.post(
                f"http://{self.server_address}/upload/image",
                files={"image": (filename, buf, "image/png")},
                data={"type": "input", "overwrite": "true"},
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()["name"]

        # Safe to retry outright, including the ambiguous timeout case: the
        # filename is unique per call and `overwrite=true`, so a re-upload
        # replaces its own bytes rather than creating a second file or
        # disturbing anyone else's.
        return self._call(upload, label="POST /upload/image")

    def queue_image_generation(self, prompt: str, client_id: str | None = None,
                                prompt_id: str | None = None) -> tuple[str, int]:
        """Non-blocking counterpart to generate_image(), for callers (e.g.
        the web server) that want to relay ComfyUI's own progress/preview
        websocket events instead of polling to a final result. Returns
        (prompt_id, seed) -- the seed is handed back so callers can show a
        real generation provenance record, not just "the model made this".

        prompt_id: see _queue_prompt()'s docstring for why a caller would
        supply its own."""
        resolved = workflow_graph.load(self.workflow_path, require=TXT2IMG_ROLES)

        seed = int.from_bytes(uuid.uuid4().bytes[:4], "big")
        resolved.set(POSITIVE_PROMPT, "text", prompt)
        resolved.set_seed(seed)

        prompt_id = self._queue_prompt(resolved.workflow, client_id, prompt_id)
        obs.log("comfy", "queued", prompt_id=prompt_id, text=prompt)
        return prompt_id, seed

    def generate_image(self, prompt: str, timeout: float = 120.0) -> Image.Image:
        prompt_id, _seed = self.queue_image_generation(prompt)
        return self._poll_for_output(
            prompt_id, lambda item: self._download_image(item, "RGBA"),
            timeout=timeout, interval=0.5, label="",
        )

    def get_result_image(self, prompt_id: str) -> Image.Image | None:
        """Non-blocking single check of /history for a finished prompt's
        output image, or None if it isn't there yet. Used by callers that
        already know (via the websocket) that execution just finished."""
        for item in self._iter_outputs(prompt_id):
            return self._download_image(item, "RGB")
        return None

    def generate_video(self, prompt: str, timeout: float = 600.0,
                        workflow_path: Path = DEFAULT_VIDEO_WORKFLOW_PATH) -> bytes:
        """Text-to-video via LTXV. Returns raw video file bytes (mp4).

        ComfyUI's SaveVideo node reports its output the same way SaveImage
        does — under the "images" key in /history, just with an extra
        "animated": [true] flag alongside it — so this reuses the same
        polling shape as generate_image(), filtered to video filenames.
        """
        resolved = workflow_graph.load(workflow_path, require=VIDEO_ROLES)

        resolved.set(POSITIVE_PROMPT, "text", prompt)
        # set_seed() picks `noise_seed` here and `seed` for the image
        # workflows, off the sampler's own class -- that difference used to be
        # a comment next to a hardcoded node id.
        resolved.set_seed(int.from_bytes(uuid.uuid4().bytes[:6], "big"))

        prompt_id = self._queue_prompt(resolved.workflow)
        obs.log("comfy", "queued video", prompt_id=prompt_id, text=prompt)

        def extract(item):
            if not item["filename"].lower().endswith(VIDEO_EXTENSIONS):
                return None
            return self._get_output_bytes(item["filename"], item["subfolder"], item["type"])

        return self._poll_for_output(prompt_id, extract, timeout=timeout, interval=1.0, label="video")

    def queue_background_generation(
        self,
        subject_photo: Image.Image,
        background_mask: Image.Image,
        depth_map: Image.Image,
        prompt: str,
        controlnet_strength: float = 0.75,
        denoise: float = 0.85,
        workflow_path: Path = DEFAULT_PHOTOSHOOT_BG_WORKFLOW_PATH,
        client_id: str | None = None,
        prompt_id: str | None = None,
        negative_prompt: str | None = None,
        seed: int | None = None,
    ) -> tuple[str, int]:
        """Queues a background-only regeneration: subject_photo is used as
        the init image, background_mask (255=regenerate, 0=keep original)
        is the inpaint mask, depth_map conditions the ControlNet so the new
        environment respects the subject's actual pose/perspective. Returns
        (prompt_id, seed) — async by design so a caller (e.g. the web
        server) can relay ComfyUI's own websocket progress/preview events
        while it runs, instead of blocking like generate_image().

        prompt_id: see _queue_prompt()'s docstring -- same race-closing
        purpose.

        negative_prompt: overrides the workflow's baked-in negative. Left
        None it keeps whatever the JSON ships with, which is what every
        caller did before brand kits existed -- and the reason the string was
        effectively unreviewable: it lived only in the workflow file and no
        API could see or change it.

        seed: pins the sampler instead of drawing a fresh random one. This is
        what makes a brand's approved look reproducible across a whole event
        rather than a different image per guest.
        """
        # Resolved before the uploads: a workflow that can't satisfy these
        # roles should fail immediately, not after pushing three PNGs at
        # ComfyUI's /upload/image.
        resolved = workflow_graph.load(workflow_path, require=PHOTOSHOOT_ROLES)

        subject_name = self._upload_image(subject_photo.convert("RGB"), "subject")
        mask_name = self._upload_image(background_mask.convert("L"), "bgmask")
        depth_name = self._upload_image(depth_map.convert("RGB"), "depth")

        # Masked to 32 bits like the random path, so a caller passing a large
        # or negative number can't produce a seed ComfyUI rejects -- and so a
        # pinned seed and a random one are always reported in the same range.
        seed = int.from_bytes(uuid.uuid4().bytes[:4], "big") if seed is None else int(seed) % (2 ** 32)
        resolved.set(SUBJECT_IMAGE, "image", subject_name)
        resolved.set(MASK_IMAGE, "image", mask_name)
        resolved.set(DEPTH_IMAGE, "image", depth_name)
        resolved.set(POSITIVE_PROMPT, "text", prompt)
        if negative_prompt is not None:
            resolved.set(NEGATIVE_PROMPT, "text", negative_prompt)
        resolved.set(CONTROLNET, "strength", controlnet_strength)
        resolved.set_seed(seed)
        resolved.set(SAMPLER, "denoise", denoise)

        prompt_id = self._queue_prompt(resolved.workflow, client_id, prompt_id)
        obs.log("comfy", "queued background", prompt_id=prompt_id, text=prompt)
        return prompt_id, seed

    def generate_background(self, *args, timeout: float = 180.0, **kwargs) -> Image.Image:
        """Synchronous convenience wrapper around queue_background_generation
        for scripts/tests that don't need live progress."""
        prompt_id, _seed = self.queue_background_generation(*args, **kwargs)
        return self._poll_for_output(
            prompt_id, lambda item: self._download_image(item, "RGB"),
            timeout=timeout, interval=0.5, label="background",
        )
