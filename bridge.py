"""
Resolume <-> generative AI live bridge.

Two trigger paths feed the same generation-and-Spout-output pipeline:

  1. Clip triggers: Resolume clip connect events arrive over OSC, get
     mapped to a prompt from prompts.json (see CLIP_CONNECT_RE / L<n>C<m>
     keys), and fire a generation.
  2. Resync: pulls Resolume's *current* composition state over its own
     REST API (active clip names + active effects) and turns that into a
     prompt describing what the VJ actually built, then fires a
     generation from it. Trigger with OSC /comfybridge/resync.

Either way the result streams out as a Spout sender that Resolume (or any
other Spout-aware app) can pick up as a live video source.

Generation itself is pluggable — --backend selects ComfyUI (local Stable
Diffusion), Runway, or Kling; see backends/.

Quick-and-dirty demo, not production code:
  - polls REST endpoints instead of using websocket/streaming APIs
  - one generation in flight at a time (new triggers are ignored while busy)
  - fixed output canvas, resized to match if the backend returns something else
"""

import argparse
import re
import threading
import time

from PIL import Image
from pythonosc import dispatcher, osc_server

import resolume_state
from backends import BACKENDS

# --- config ---------------------------------------------------------------

OSC_LISTEN_IP = "0.0.0.0"
OSC_LISTEN_PORT = 9000
MANUAL_TRIGGER_ADDRESS = "/comfybridge/generate"  # arg0: freeform prompt text
RESYNC_TRIGGER_ADDRESS = "/comfybridge/resync"  # no args: pull Resolume state

RESYNC_STYLE_SUFFIX = "digital generative art, VJ visual, vivid colors, high detail"

SPOUT_SENDER_NAME = "ComfyBridge"
SPOUT_WIDTH = 512
SPOUT_HEIGHT = 512
SPOUT_FPS = 15


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


# --- trigger -> generation plumbing -----------------------------------------

CLIP_CONNECT_RE = re.compile(r"/composition/layers/(\d+)/clips/(\d+)/connect")


def load_prompts(prompts_path) -> dict:
    import json
    with open(prompts_path) as f:
        return json.load(f)


def run_generation(backend, frame_buffer: FrameBuffer, prompt_text: str):
    if not frame_buffer.busy.acquire(blocking=False):
        print("[bridge] busy generating, ignoring trigger")
        return

    def worker():
        try:
            image = backend.generate_image(prompt_text)
            frame_buffer.set_image(image)
            print("[bridge] frame updated")
        except Exception as exc:
            print(f"[bridge] generation failed: {exc}")
        finally:
            frame_buffer.busy.release()

    threading.Thread(target=worker, daemon=True).start()


def make_clip_trigger_handler(backend, frame_buffer, prompts_path):
    prompts = load_prompts(prompts_path)

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
        run_generation(backend, frame_buffer, prompt_text)

    return handler


def make_manual_trigger_handler(backend, frame_buffer):
    def handler(address: str, *args):
        if not args:
            print("[osc] manual trigger with no prompt text, ignoring")
            return
        prompt_text = str(args[0])
        print(f"[osc] manual trigger -> {prompt_text!r}")
        run_generation(backend, frame_buffer, prompt_text)

    return handler


def make_resync_handler(backend, frame_buffer, resolume_url: str):
    def handler(address: str, *args):
        try:
            prompt_text = resolume_state.compose_prompt(resolume_url, RESYNC_STYLE_SUFFIX)
        except Exception as exc:
            print(f"[resolume] failed to read composition from {resolume_url}: {exc}")
            return
        print(f"[resolume] composed prompt -> {prompt_text!r}")
        run_generation(backend, frame_buffer, prompt_text)

    return handler


# --- main ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=sorted(BACKENDS), default="comfy")
    parser.add_argument("--comfy", default="127.0.0.1:8188", help="ComfyUI host:port (--backend comfy)")
    parser.add_argument("--resolume-url", default=resolume_state.DEFAULT_RESOLUME_URL,
                         help="Resolume REST API base URL, for /comfybridge/resync")
    parser.add_argument("--prompts", default="prompts.json", help="clip-trigger prompt map")
    parser.add_argument("--osc-port", type=int, default=OSC_LISTEN_PORT)
    args = parser.parse_args()

    if args.backend == "comfy":
        backend = BACKENDS["comfy"](args.comfy)
    else:
        backend = BACKENDS[args.backend]()

    frame_buffer = FrameBuffer(SPOUT_WIDTH, SPOUT_HEIGHT)
    stop_event = threading.Event()

    spout_thread = threading.Thread(
        target=spout_sender_loop, args=(frame_buffer, stop_event), daemon=True
    )
    spout_thread.start()

    disp = dispatcher.Dispatcher()
    disp.map("/composition/layers/*/clips/*/connect", make_clip_trigger_handler(backend, frame_buffer, args.prompts))
    disp.map(MANUAL_TRIGGER_ADDRESS, make_manual_trigger_handler(backend, frame_buffer))
    disp.map(RESYNC_TRIGGER_ADDRESS, make_resync_handler(backend, frame_buffer, args.resolume_url))

    server = osc_server.ThreadingOSCUDPServer((OSC_LISTEN_IP, args.osc_port), disp)
    print(f"[osc] listening on {OSC_LISTEN_IP}:{args.osc_port}")
    print(f"[bridge] backend={args.backend}, Spout sender '{SPOUT_SENDER_NAME}'")
    print("[bridge] enable OSC output in Resolume Preferences > OSC, "
          f"host=<this machine>, port={args.osc_port}")
    print(f"[bridge] resync pulls live state from {args.resolume_url} "
          "(enable Preferences > Webserver in Resolume)")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        spout_thread.join(timeout=2)


if __name__ == "__main__":
    main()
