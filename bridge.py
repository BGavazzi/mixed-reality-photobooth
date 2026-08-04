"""
Resolume <-> ComfyUI live bridge.

Resolume clip triggers arrive over OSC -> the matching prompt is sent to a
local ComfyUI instance -> the resulting frame is streamed out as a Spout
sender that Resolume picks up as a live video source.

Quick-and-dirty demo, not production code:
  - polls ComfyUI's /history REST endpoint instead of the websocket API
  - one generation in flight at a time (new triggers are ignored while busy)
  - fixed output canvas, resized to match if the workflow produces something else
"""

import argparse
import json
import re
import threading
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

import requests
from PIL import Image
from pythonosc import dispatcher, osc_server

# --- config ---------------------------------------------------------------

COMFYUI_SERVER = "127.0.0.1:8188"
WORKFLOW_PATH = Path(__file__).parent / "workflows" / "txt2img_api.json"
PROMPTS_PATH = Path(__file__).parent / "prompts.json"

OSC_LISTEN_IP = "0.0.0.0"
OSC_LISTEN_PORT = 9000
MANUAL_TRIGGER_ADDRESS = "/comfybridge/generate"  # arg0: freeform prompt text

SPOUT_SENDER_NAME = "ComfyBridge"
SPOUT_WIDTH = 512
SPOUT_HEIGHT = 512
SPOUT_FPS = 15

POSITIVE_PROMPT_NODE = "6"
SEED_NODE = "3"

# --- ComfyUI client ---------------------------------------------------------

class ComfyClient:
    def __init__(self, server_address: str):
        self.server_address = server_address
        self.client_id = str(uuid.uuid4())

    def queue_prompt(self, workflow: dict) -> str:
        payload = {"prompt": workflow, "client_id": self.client_id}
        resp = requests.post(f"http://{self.server_address}/prompt", json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()["prompt_id"]

    def get_history(self, prompt_id: str) -> dict:
        resp = requests.get(f"http://{self.server_address}/history/{prompt_id}", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def get_image_bytes(self, filename: str, subfolder: str, folder_type: str) -> bytes:
        params = urllib.parse.urlencode(
            {"filename": filename, "subfolder": subfolder, "type": folder_type}
        )
        with urllib.request.urlopen(f"http://{self.server_address}/view?{params}") as resp:
            return resp.read()

    def generate_image(self, prompt_text: str, timeout: float = 120.0) -> Image.Image:
        with open(WORKFLOW_PATH) as f:
            workflow = json.load(f)

        workflow[POSITIVE_PROMPT_NODE]["inputs"]["text"] = prompt_text
        workflow[SEED_NODE]["inputs"]["seed"] = int.from_bytes(uuid.uuid4().bytes[:4], "big")

        prompt_id = self.queue_prompt(workflow)
        print(f"[comfy] queued prompt_id={prompt_id} text={prompt_text!r}")

        deadline = time.time() + timeout
        while time.time() < deadline:
            history = self.get_history(prompt_id)
            entry = history.get(prompt_id)
            if entry and entry.get("outputs"):
                for node_output in entry["outputs"].values():
                    for image in node_output.get("images", []):
                        raw = self.get_image_bytes(
                            image["filename"], image["subfolder"], image["type"]
                        )
                        return Image.open(__import__("io").BytesIO(raw)).convert("RGBA")
            time.sleep(0.5)

        raise TimeoutError(f"ComfyUI did not finish prompt_id={prompt_id} within {timeout}s")


# --- shared frame state -----------------------------------------------------

class FrameBuffer:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.lock = threading.Lock()
        self.data = bytes([40, 40, 48, 255]) * (width * height)  # placeholder: dark grey
        self.busy = threading.Lock()

    def set_image(self, image: Image.Image):
        if image.size != (self.width, self.height):
            image = image.resize((self.width, self.height), Image.LANCZOS)
        with self.lock:
            self.data = image.tobytes()

    def get_bytes(self) -> bytes:
        with self.lock:
            return self.data


# --- Spout sender thread -----------------------------------------------------

def spout_sender_loop(frame_buffer: FrameBuffer, stop_event: threading.Event):
    import SpoutGL
    from OpenGL import GL

    with SpoutGL.SpoutSender() as sender:
        sender.setSenderName(SPOUT_SENDER_NAME)
        print(f"[spout] sender '{SPOUT_SENDER_NAME}' started at {frame_buffer.width}x{frame_buffer.height}")

        while not stop_event.is_set():
            sender.sendImage(
                frame_buffer.get_bytes(),
                frame_buffer.width,
                frame_buffer.height,
                GL.GL_RGBA,
                False,
                0,
            )
            sender.setFrameSync(SPOUT_SENDER_NAME)
            time.sleep(1.0 / SPOUT_FPS)


# --- OSC trigger handling -----------------------------------------------------

CLIP_CONNECT_RE = re.compile(r"/composition/layers/(\d+)/clips/(\d+)/connect")


def load_prompts() -> dict:
    with open(PROMPTS_PATH) as f:
        return json.load(f)


def run_generation(comfy: ComfyClient, frame_buffer: FrameBuffer, prompt_text: str):
    if not frame_buffer.busy.acquire(blocking=False):
        print("[bridge] busy generating, ignoring trigger")
        return

    def worker():
        try:
            image = comfy.generate_image(prompt_text)
            frame_buffer.set_image(image)
            print("[bridge] frame updated")
        except Exception as exc:
            print(f"[bridge] generation failed: {exc}")
        finally:
            frame_buffer.busy.release()

    threading.Thread(target=worker, daemon=True).start()


def make_clip_trigger_handler(comfy: ComfyClient, frame_buffer: FrameBuffer):
    prompts = load_prompts()

    def handler(address: str, *args):
        match = CLIP_CONNECT_RE.match(address)
        if not match:
            return
        value = args[0] if args else 0
        if value < 1:
            return  # ignore clip-disconnect events

        layer, clip = match.groups()
        key = f"L{layer}C{clip}"
        prompt_text = prompts.get(key, prompts.get("_default", "abstract generative art"))
        print(f"[osc] clip trigger {key} -> {prompt_text!r}")
        run_generation(comfy, frame_buffer, prompt_text)

    return handler


def make_manual_trigger_handler(comfy: ComfyClient, frame_buffer: FrameBuffer):
    def handler(address: str, *args):
        if not args:
            print("[osc] manual trigger with no prompt text, ignoring")
            return
        prompt_text = str(args[0])
        print(f"[osc] manual trigger -> {prompt_text!r}")
        run_generation(comfy, frame_buffer, prompt_text)

    return handler


# --- main ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy", default=COMFYUI_SERVER, help="ComfyUI host:port")
    parser.add_argument("--osc-port", type=int, default=OSC_LISTEN_PORT)
    args = parser.parse_args()

    comfy = ComfyClient(args.comfy)
    frame_buffer = FrameBuffer(SPOUT_WIDTH, SPOUT_HEIGHT)
    stop_event = threading.Event()

    spout_thread = threading.Thread(
        target=spout_sender_loop, args=(frame_buffer, stop_event), daemon=True
    )
    spout_thread.start()

    disp = dispatcher.Dispatcher()
    disp.map("/composition/layers/*/clips/*/connect", make_clip_trigger_handler(comfy, frame_buffer))
    disp.map(MANUAL_TRIGGER_ADDRESS, make_manual_trigger_handler(comfy, frame_buffer))

    server = osc_server.ThreadingOSCUDPServer((OSC_LISTEN_IP, args.osc_port), disp)
    print(f"[osc] listening on {OSC_LISTEN_IP}:{args.osc_port}")
    print(f"[bridge] ComfyUI at {args.comfy}, Spout sender '{SPOUT_SENDER_NAME}'")
    print("[bridge] enable OSC output in Resolume Preferences > OSC, "
          f"host=<this machine>, port={args.osc_port}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        spout_thread.join(timeout=2)


if __name__ == "__main__":
    main()
