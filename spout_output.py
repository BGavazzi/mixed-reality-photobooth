"""
Shared Spout video output: a thread-safe frame buffer plus a background
thread that pushes it out as a named Spout source at a fixed frame rate.

Both bridge.py (Resolume VJ bridge) and web_server.py (mixed-reality photo
booth) generate frames from a ComfyUI backend and broadcast them over Spout
independently, under different sender names, at different resolutions/FPS —
this is that shared, generation-agnostic piece, decoupled from either app's
own trigger/generation logic so a third consumer (or a future standalone
Spout-output service) can reuse it without depending on either app.
"""

import threading
import time

from PIL import Image

DEFAULT_PLACEHOLDER_RGBA = (20, 20, 24, 255)  # near-black, shown before the first frame arrives


def _cover_fit(image: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Scales to fill target_w x target_h, cropping the overflow, instead of
    stretching to fit -- a naive resize() would squash a non-square frame
    (e.g. the photo booth canvas fit to a portrait photo, see
    web/index.html's own cover-fit redraw()) into the sender's fixed
    dimensions, the same body-proportion distortion that was fixed on the
    browser-canvas side. A live Spout output should crop-to-fill like any
    normal video source, not stretch."""
    src_w, src_h = image.size
    scale = max(target_w / src_w, target_h / src_h)
    resized = image.resize((round(src_w * scale), round(src_h * scale)), Image.LANCZOS)
    left = (resized.width - target_w) // 2
    top = (resized.height - target_h) // 2
    return resized.crop((left, top, left + target_w, top + target_h))


class SpoutFrameBuffer:
    """Holds the single most recent frame as raw RGBA bytes, safe to write
    from a generation callback and read from the Spout sender thread
    concurrently. `busy` is a convenience lock for the common "one
    generation in flight at a time" pattern both apps use — unused if a
    caller doesn't need it."""

    def __init__(self, width: int, height: int, placeholder_rgba=DEFAULT_PLACEHOLDER_RGBA):
        self.width = width
        self.height = height
        self.lock = threading.Lock()
        self.busy = threading.Lock()
        self.data = bytes(placeholder_rgba) * (width * height)

    def set_image(self, image: Image.Image):
        if image.size != (self.width, self.height):
            image = _cover_fit(image, self.width, self.height)
        with self.lock:
            self.data = image.convert("RGBA").tobytes()

    def get_bytes(self) -> bytes:
        with self.lock:
            return self.data


def spout_sender_loop(frame_buffer: SpoutFrameBuffer, sender_name: str, fps: float,
                       stop_event: threading.Event, live_event: threading.Event = None):
    """Blocking loop, intended to run in its own daemon thread: republishes
    frame_buffer's current frame over Spout as sender_name at fps until
    stop_event is set.

    live_event, if given, is set once the sender is actually publishing and
    cleared when it stops, so a caller can tell the difference between "sent
    to Spout" and "wrote a frame into a buffer nothing is reading."

    Every failure here is reported rather than allowed to kill the thread
    quietly. This runs as a daemon thread nobody joins, so an unhandled
    exception -- most likely `import SpoutGL` on a machine without it, since
    Spout is Windows-only -- printed a traceback into the startup noise and
    then left the app looking completely healthy while "Send to Spout"
    reported success and no Spout source ever appeared. Silence is the worst
    possible outcome for an output whose whole job is to be picked up by
    another application.
    """
    try:
        import SpoutGL
        from OpenGL import GL
    except ImportError as exc:
        print(f"[spout] DISABLED: {exc}. Spout is Windows-only (pip install SpoutGL PyOpenGL); "
              f"everything else works, but no '{sender_name}' source will be published.")
        return

    try:
        with SpoutGL.SpoutSender() as sender:
            sender.setSenderName(sender_name)
            print(f"[spout] sender '{sender_name}' started at {frame_buffer.width}x{frame_buffer.height}")
            if live_event is not None:
                live_event.set()
            while not stop_event.is_set():
                sender.sendImage(
                    frame_buffer.get_bytes(),
                    frame_buffer.width,
                    frame_buffer.height,
                    GL.GL_RGBA,
                    False,
                    0,
                )
                sender.setFrameSync(sender_name)
                time.sleep(1.0 / fps)
    except Exception as exc:
        print(f"[spout] sender '{sender_name}' stopped with an error: {exc!r}")
    finally:
        if live_event is not None:
            live_event.clear()
