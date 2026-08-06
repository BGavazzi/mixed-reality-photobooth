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
import os
import re
import tempfile
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
VIDEO_TRIGGER_ADDRESS = "/comfybridge/generate_video"  # arg0: freeform prompt text (ComfyUI backend only)
PLAY_FILE_ADDRESS = "/comfybridge/play_file"  # arg0: local video file path, played as-is (no generation)
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


# --- video playback (loops a generated clip's frames into the FrameBuffer) --

class VideoLoopPlayer:
    def __init__(self, frame_buffer: FrameBuffer, path: str):
        self.frame_buffer = frame_buffer
        self.path = path
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=2)

    def _run(self):
        import cv2

        cap = cv2.VideoCapture(self.path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 24
        delay = 1.0 / fps
        while not self.stop_event.is_set():
            ok, frame_bgr = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop back to start
                continue
            rgba = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA)
            self.frame_buffer.set_image(Image.fromarray(rgba))
            time.sleep(delay)
        cap.release()


class VideoPlaybackManager:
    """Owns whatever video is currently looping into the FrameBuffer, so a
    new generation can cleanly replace the previous one."""

    def __init__(self, frame_buffer: FrameBuffer):
        self.frame_buffer = frame_buffer
        self.lock = threading.Lock()
        self.current: VideoLoopPlayer = None
        self.current_owned = False  # whether the bridge created current.path (and should delete it)

    def play(self, path: str, delete_on_replace: bool = True):
        with self.lock:
            self._stop_current_locked()
            self.current = VideoLoopPlayer(self.frame_buffer, path)
            self.current_owned = delete_on_replace
            self.current.start()

    def stop_current(self):
        """Stop whatever video is looping, if any, so a plain image can be
        published without the video thread immediately clobbering it on its
        next tick."""
        with self.lock:
            self._stop_current_locked()

    def _stop_current_locked(self):
        if self.current is not None:
            self.current.stop()  # joins the thread, so it can't set_image() after this returns
            if self.current_owned:
                try:
                    os.remove(self.current.path)
                except OSError:
                    pass
            self.current = None
            self.current_owned = False


# --- trigger -> generation plumbing -----------------------------------------

CLIP_CONNECT_RE = re.compile(r"/composition/layers/(\d+)/clips/(\d+)/connect")


def load_prompts(prompts_path) -> dict:
    import json
    with open(prompts_path) as f:
        return json.load(f)


def run_generation(backend, frame_buffer: FrameBuffer, prompt_text: str, video_manager: "VideoPlaybackManager" = None):
    if not frame_buffer.busy.acquire(blocking=False):
        print("[bridge] busy generating, ignoring trigger")
        return

    def worker():
        try:
            image = backend.generate_image(prompt_text)
            if video_manager is not None:
                video_manager.stop_current()  # stop any looping video before publishing the new still
            frame_buffer.set_image(image)
            print("[bridge] frame updated")
        except Exception as exc:
            print(f"[bridge] generation failed: {exc}")
        finally:
            frame_buffer.busy.release()

    threading.Thread(target=worker, daemon=True).start()


def make_clip_trigger_handler(backend, frame_buffer, prompts_path, video_manager):
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
        run_generation(backend, frame_buffer, prompt_text, video_manager)

    return handler


def run_video_generation(backend, frame_buffer: FrameBuffer, video_manager: VideoPlaybackManager, prompt_text: str):
    if not hasattr(backend, "generate_video"):
        print(f"[bridge] backend {backend.__class__.__name__} does not support generate_video, ignoring trigger")
        return
    if not frame_buffer.busy.acquire(blocking=False):
        print("[bridge] busy generating, ignoring trigger")
        return

    def worker():
        try:
            video_bytes = backend.generate_video(prompt_text)
            fd, tmp_path = tempfile.mkstemp(suffix=".mp4", prefix="comfybridge_")
            with os.fdopen(fd, "wb") as f:
                f.write(video_bytes)
            video_manager.play(tmp_path)
            print("[bridge] video playback started")
        except Exception as exc:
            print(f"[bridge] video generation failed: {exc}")
        finally:
            frame_buffer.busy.release()

    threading.Thread(target=worker, daemon=True).start()


def make_manual_trigger_handler(backend, frame_buffer, video_manager):
    def handler(address: str, *args):
        if not args:
            print("[osc] manual trigger with no prompt text, ignoring")
            return
        prompt_text = str(args[0])
        print(f"[osc] manual trigger -> {prompt_text!r}")
        run_generation(backend, frame_buffer, prompt_text, video_manager)

    return handler


def make_video_trigger_handler(backend, frame_buffer, video_manager):
    def handler(address: str, *args):
        if not args:
            print("[osc] video trigger with no prompt text, ignoring")
            return
        prompt_text = str(args[0])
        print(f"[osc] video trigger -> {prompt_text!r}")
        run_video_generation(backend, frame_buffer, video_manager, prompt_text)

    return handler


def make_play_file_handler(video_manager):
    def handler(address: str, *args):
        if not args:
            print("[osc] play_file trigger with no path, ignoring")
            return
        path = str(args[0])
        if not os.path.isfile(path):
            print(f"[osc] play_file: no such file {path!r}")
            return
        print(f"[osc] playing local file -> {path!r}")
        video_manager.play(path, delete_on_replace=False)

    return handler


def make_resync_handler(backend, frame_buffer, resolume_url: str, video_manager):
    def handler(address: str, *args):
        try:
            prompt_text = resolume_state.compose_prompt(resolume_url, RESYNC_STYLE_SUFFIX)
        except Exception as exc:
            print(f"[resolume] failed to read composition from {resolume_url}: {exc}")
            return
        print(f"[resolume] composed prompt -> {prompt_text!r}")
        run_generation(backend, frame_buffer, prompt_text, video_manager)

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
    video_manager = VideoPlaybackManager(frame_buffer)
    stop_event = threading.Event()

    spout_thread = threading.Thread(
        target=spout_sender_loop, args=(frame_buffer, stop_event), daemon=True
    )
    spout_thread.start()

    disp = dispatcher.Dispatcher()
    disp.map("/composition/layers/*/clips/*/connect", make_clip_trigger_handler(backend, frame_buffer, args.prompts, video_manager))
    disp.map(MANUAL_TRIGGER_ADDRESS, make_manual_trigger_handler(backend, frame_buffer, video_manager))
    disp.map(VIDEO_TRIGGER_ADDRESS, make_video_trigger_handler(backend, frame_buffer, video_manager))
    disp.map(PLAY_FILE_ADDRESS, make_play_file_handler(video_manager))
    disp.map(RESYNC_TRIGGER_ADDRESS, make_resync_handler(backend, frame_buffer, args.resolume_url, video_manager))

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
