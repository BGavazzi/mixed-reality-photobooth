import io
import os
import time

import jwt
import requests
from PIL import Image

from .base import GenerationBackend

DEFAULT_BASE_URL = "https://api-singapore.klingai.com"  # use https://api.klingai.com inside China


class KlingBackend(GenerationBackend):
    """Kling AI's hosted API (https://kling.ai/document-api).

    Requires KLING_ACCESS_KEY / KLING_SECRET_KEY in the environment and
    `pip install pyjwt`. Auth is a short-lived JWT signed with the secret
    key, regenerated per request rather than cached, since it's cheap and
    avoids tracking expiry.
    """

    def __init__(self, model: str = "kling-v1", base_url: str = DEFAULT_BASE_URL):
        self.access_key = os.environ.get("KLING_ACCESS_KEY")
        self.secret_key = os.environ.get("KLING_SECRET_KEY")
        if not (self.access_key and self.secret_key):
            raise RuntimeError("Set KLING_ACCESS_KEY and KLING_SECRET_KEY to use the Kling backend")
        self.model = model
        self.base_url = base_url

    def _token(self) -> str:
        payload = {
            "iss": self.access_key,
            "exp": int(time.time()) + 1800,
            "nbf": int(time.time()) - 5,
        }
        return jwt.encode(payload, self.secret_key, algorithm="HS256", headers={"alg": "HS256", "typ": "JWT"})

    def generate_image(self, prompt: str, timeout: float = 120.0) -> Image.Image:
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self._token()}"}
        resp = requests.post(
            f"{self.base_url}/v1/images/generations",
            headers=headers,
            json={"model": self.model, "prompt": prompt},
            timeout=30,
        )
        resp.raise_for_status()
        task_id = resp.json()["data"]["task_id"]
        print(f"[kling] queued task_id={task_id} text={prompt!r}")

        deadline = time.time() + timeout
        while time.time() < deadline:
            poll_headers = {"Authorization": f"Bearer {self._token()}"}
            poll = requests.get(
                f"{self.base_url}/v1/images/generations",
                headers=poll_headers,
                params={"pageSize": 500},
                timeout=30,
            ).json()

            for task in poll.get("data", []):
                if task.get("task_id") != task_id:
                    continue
                status = task.get("task_status")
                if status == "succeed":
                    url = task["task_result"]["images"][0]["url"]
                    image_resp = requests.get(url, timeout=30)
                    image_resp.raise_for_status()
                    return Image.open(io.BytesIO(image_resp.content)).convert("RGBA")
                if status == "failed":
                    raise RuntimeError(f"Kling task failed: {task}")

            time.sleep(2)

        raise TimeoutError(f"Kling task {task_id} did not finish within {timeout}s")
