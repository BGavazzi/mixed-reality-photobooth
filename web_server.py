"""
Browser front-end for the ComfyUI pipeline: mixed-reality photo booth.

Upload a real photo -> rotoscope / pose / depth / illumination extracted
-> generate a new background around the subject (ComfyUI + SDXL ControlNet
depth-conditioned inpaint) -> composite the untouched subject back on top.
The whole generation step streams live over a websocket relay of ComfyUI's
own progress + denoising-preview events, so the browser shows the image
actually forming instead of a spinner.

Multiple browser tabs/sessions can be connected and have jobs in flight at
once -- each /ws connection gets its own session_id, and ComfyUI's events
route back by prompt_id -> session rather than to a single global "whoever
connected last." ComfyUI itself still renders one graph at a time (a real
GPU constraint), so concurrent sessions queue naturally through its own
/prompt queue; what multi-session routing fixes is that each session
correctly gets *its own* results instead of racing a shared global.

    python web_server.py
    -> http://127.0.0.1:8000
"""

import argparse
import asyncio
import base64
import io
import json
import struct
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import websockets
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, ImageOps

from backends.comfy import ComfyBackend
from spout_output import SpoutFrameBuffer, spout_sender_loop
import photoshoot_pipeline

COMFY_ADDRESS = "127.0.0.1:8188"
COMFY_CLIENT_ID = "web-photoshoot-bridge"
BINARY_PREVIEW_EVENT = 1  # ComfyUI's BinaryEventTypes.PREVIEW_IMAGE

# Matches workflows/txt2img_api.json and workflows/photoshoot_bg_api.json's
# hardcoded ckpt_name/control_net_name — surfaced here too so the
# provenance record doesn't have to re-parse the workflow JSON to know
# what model actually produced a given image.
CHECKPOINT_NAME = "RealVisXL_V5.0_fp16.safetensors"
CONTROLNET_NAME = "diffusers_xl_depth_full.safetensors"

# Virtual-production tie-in: pushes a generated composite out as its own
# Spout source, so a real LED wall / projector on set can show it live
# behind the model during the actual shoot — the same technique bridge.py
# uses, kept as an independent sender here (own name, own thread) so this
# app doesn't depend on bridge.py running at all, and never collides with
# its "ComfyBridge" sender if both happen to be up at once.
PHOTOBOOTH_SPOUT_NAME = "PhotoBooth"
PHOTOBOOTH_SPOUT_WIDTH = 768
PHOTOBOOTH_SPOUT_HEIGHT = 768
PHOTOBOOTH_SPOUT_FPS = 15


photobooth_frame_buffer = SpoutFrameBuffer(PHOTOBOOTH_SPOUT_WIDTH, PHOTOBOOTH_SPOUT_HEIGHT)
spout_stop_event = threading.Event()


def _warm_up_pipeline_models():
    """Runs each CV stage once on a throwaway image before the server takes
    real traffic. Without this, the *first* real upload pays for lazy model
    construction (rembg session, OpenposeDetector, MidasDetector) inline —
    and MiDaS's first-ever forward pass in a freshly loaded process has been
    observed to fail outright with a tensor-size mismatch inside its DPT
    skip connections (a cold-load race, not an input-size problem: a second
    call in the same process succeeds every time). Warming up here moves
    that cost and that failure mode out of the user-facing request path.

    Retries once on failure precisely because it's the *first* call that's
    been observed to trip the race — if warmup itself hits it, a second
    attempt in the same now-partially-loaded process is the empirically
    reliable recovery. If both attempts fail, log and let the server start
    anyway: failing startup entirely over a cold-load race would be a worse
    outcome than the pre-warmup behavior of only the first real request
    failing."""
    dummy = Image.new("RGB", (512, 512), (128, 128, 128))
    try:
        photoshoot_pipeline.analyze(dummy)
    except Exception as exc:
        print(f"[warmup] first attempt failed ({exc!r}), retrying once...")
        try:
            photoshoot_pipeline.analyze(dummy)
        except Exception as exc2:
            print(f"[warmup] retry also failed ({exc2!r}) — starting anyway, "
                  f"first real upload may hit this instead")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[warmup] loading rotoscope/pose/depth models...")
    await asyncio.to_thread(_warm_up_pipeline_models)
    print("[warmup] done")
    asyncio.create_task(comfy_relay_loop())
    threading.Thread(
        target=spout_sender_loop,
        args=(photobooth_frame_buffer, PHOTOBOOTH_SPOUT_NAME, PHOTOBOOTH_SPOUT_FPS, spout_stop_event),
        daemon=True,
    ).start()
    yield
    spout_stop_event.set()


app = FastAPI(lifespan=lifespan)
backend = ComfyBackend(COMFY_ADDRESS)

# Multi-session job routing. Each browser tab that opens /ws gets its own
# session_id; ComfyUI's own websocket events carry a prompt_id, and JOBS
# maps that back to the session that queued it, so results route to the
# *right* client instead of a single global "whoever's currently active" --
# the earlier design meant a second concurrent tab would silently steal or
# corrupt the first tab's in-flight generation. ComfyUI itself still only
# executes one graph at a time (a real GPU constraint, not a shortcut taken
# here) — multiple sessions queueing concurrently now queue correctly
# through ComfyUI's own /prompt queue instead of racing a shared global.
SESSIONS: dict[str, WebSocket] = {}  # session_id -> websocket
JOBS: dict[str, dict] = {}  # prompt_id -> {"session_id", "kind", "provenance"}

# Binary preview frames carry no prompt_id in ComfyUI's wire protocol (see
# "Why a websocket relay instead of just polling" in README.md) -- this is
# the one place a single "currently active" value is actually correct
# rather than a limitation, since it mirrors ComfyUI's real one-graph-at-
# a-time execution: whichever prompt_id we last saw in a JSON event is
# unambiguously the one whose binary frames are arriving right now.
executing_prompt_id: str | None = None


# --- helpers ----------------------------------------------------------------

def pil_to_b64(image: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def b64_to_pil(data: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(data)))


async def send_json_to(session_id: str, payload: dict):
    ws = SESSIONS.get(session_id)
    if ws is not None:
        try:
            await ws.send_json(payload)
        except Exception as exc:
            print(f"[ws] send to session {session_id} failed: {exc}")


async def send_bytes_to(session_id: str, data: bytes):
    ws = SESSIONS.get(session_id)
    if ws is not None:
        try:
            await ws.send_bytes(data)
        except Exception as exc:
            print(f"[ws] send to session {session_id} failed: {exc}")


async def broadcast_json(payload: dict):
    for ws in list(SESSIONS.values()):
        try:
            await ws.send_json(payload)
        except Exception as exc:
            print(f"[ws] broadcast failed: {exc}")


# --- ComfyUI websocket relay -------------------------------------------------

async def comfy_relay_loop():
    uri = f"ws://{COMFY_ADDRESS}/ws?clientId={COMFY_CLIENT_ID}"
    while True:
        try:
            async with websockets.connect(uri, max_size=None) as ws:
                print(f"[relay] connected to ComfyUI at {uri}")
                async for message in ws:
                    await handle_comfy_message(message)
        except Exception as exc:
            print(f"[relay] comfy ws error: {exc!r}, reconnecting in 2s")
            await asyncio.sleep(2)


async def handle_comfy_message(message):
    global executing_prompt_id

    if isinstance(message, (bytes, bytearray)):
        if len(message) < 8:
            return
        event_type = struct.unpack(">I", message[:4])[0]
        if event_type == BINARY_PREVIEW_EVENT and executing_prompt_id is not None:
            job = JOBS.get(executing_prompt_id)
            if job is not None:
                await send_bytes_to(job["session_id"], message[8:])  # strip 4-byte event + 4-byte format header
        return

    try:
        event = json.loads(message)
    except json.JSONDecodeError:
        return

    etype = event.get("type")
    data = event.get("data", {})
    prompt_id = data.get("prompt_id")

    if etype == "status":
        # Not job-specific -- every connected session sees overall queue
        # depth, not just whichever session happens to have a job in flight.
        queue_remaining = data.get("status", {}).get("exec_info", {}).get("queue_remaining")
        await broadcast_json({"type": "status", "queue_remaining": queue_remaining})
        return

    job = JOBS.get(prompt_id)
    if job is None:
        return  # not a job we queued (or already finished/errored and cleaned up)
    session_id = job["session_id"]

    if etype == "progress":
        executing_prompt_id = prompt_id
        await send_json_to(session_id, {"type": "progress", "value": data.get("value"), "max": data.get("max")})

    elif etype == "executing":
        node = data.get("node")
        if node is None:
            # None node for a job we're tracking means this run just finished.
            image = await asyncio.to_thread(backend.get_result_image, prompt_id)
            if image is not None:
                await send_json_to(session_id, {
                    "type": "done",
                    "image_base64": pil_to_b64(image),
                    "kind": job["kind"],
                    "provenance": job["provenance"],
                })
            JOBS.pop(prompt_id, None)
            if executing_prompt_id == prompt_id:
                executing_prompt_id = None
        else:
            executing_prompt_id = prompt_id
            await send_json_to(session_id, {"type": "executing", "node": node})

    elif etype == "execution_error":
        await send_json_to(session_id, {"type": "error", "message": str(data.get("exception_message", "generation failed"))})
        JOBS.pop(prompt_id, None)
        if executing_prompt_id == prompt_id:
            executing_prompt_id = None

    elif etype == "execution_interrupted":
        # Fires if a job is cancelled from ComfyUI's own UI/API (queue
        # cleared, manually interrupted) rather than through this app --
        # without handling it, JOBS would never get cleaned up for that
        # prompt_id and would sit there for the life of the process, a slow
        # per-cancelled-job memory leak on what's meant to be a long-running
        # server.
        await send_json_to(session_id, {"type": "error", "message": "generation was interrupted"})
        JOBS.pop(prompt_id, None)
        if executing_prompt_id == prompt_id:
            executing_prompt_id = None


# --- HTTP routes --------------------------------------------------------------

@app.get("/")
def index():
    return FileResponse("web/index.html")


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    raw = await file.read()
    image = Image.open(io.BytesIO(raw)).convert("RGB")

    result = await asyncio.to_thread(photoshoot_pipeline.analyze, image)

    return JSONResponse({
        # result["image"] is `image` downscaled if it was oversized -- this,
        # not the raw upload, becomes the browser's canonical "original" and
        # is what later generation calls send back as the subject photo.
        "original": pil_to_b64(result["image"]),
        "cutout": pil_to_b64(result["cutout"]),
        "mask": pil_to_b64(result["mask"]),
        "pose": pil_to_b64(result["pose"]),
        "depth": pil_to_b64(result["depth"]),
        "shadow": pil_to_b64(result["shadow"]),
        "illumination": result["illumination"].to_dict(),
        "suggested_controlnet_strength": result["suggested_controlnet_strength"],
        "width": result["image"].width,
        "height": result["image"].height,
    })


# --- websocket: browser control + live event relay ---------------------------

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    session_id = str(uuid.uuid4())
    await websocket.accept()
    SESSIONS[session_id] = websocket
    try:
        while True:
            msg = await websocket.receive_json()
            action = msg.get("action")

            if action == "generate_image":
                asyncio.create_task(handle_generate_image(session_id, msg))
            elif action == "generate_background":
                asyncio.create_task(handle_generate_background(session_id, msg))
            elif action == "edit_region":
                asyncio.create_task(handle_edit_region(session_id, msg))
            elif action == "send_to_spout":
                asyncio.create_task(handle_send_to_spout(session_id, msg))

    except WebSocketDisconnect:
        pass
    finally:
        SESSIONS.pop(session_id, None)
        # Jobs this session queued are left in JOBS -- ComfyUI still renders
        # them, and send_json_to()/send_bytes_to() just no-op once the
        # session is gone rather than erroring, so a disconnect mid-generation
        # can't crash the relay loop or leak into another session's results.


async def handle_generate_image(session_id: str, msg: dict):
    prompt = (msg.get("prompt") or "").strip()
    if not prompt:
        return
    prompt_id, seed = await asyncio.to_thread(backend.queue_image_generation, prompt, COMFY_CLIENT_ID)
    JOBS[prompt_id] = {
        "session_id": session_id,
        "kind": "image",
        "provenance": {
            "kind": "image", "prompt": prompt, "seed": seed,
            "checkpoint": CHECKPOINT_NAME, "controlnet": None,
            "controlnet_strength": None, "denoise": None,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    await send_json_to(session_id, {"type": "queued", "prompt_id": prompt_id})


async def handle_generate_background(session_id: str, msg: dict):
    try:
        subject = b64_to_pil(msg["subject"]).convert("RGB")
        mask = b64_to_pil(msg["mask"]).convert("L")
        depth = b64_to_pil(msg["depth"]).convert("RGB")
        # UI sends the subject mask (255=subject); the workflow needs the
        # inverse (255=background=regenerate).
        background_mask = ImageOps.invert(mask)
        prompt = (msg.get("prompt") or "").strip()
        controlnet_strength = float(msg.get("controlnet_strength", 0.75))
        denoise = float(msg.get("denoise", 0.85))
    except Exception as exc:
        await send_json_to(session_id, {"type": "error", "message": f"bad request: {exc}"})
        return

    prompt_id, seed = await asyncio.to_thread(
        backend.queue_background_generation,
        subject, background_mask, depth, prompt,
        controlnet_strength, denoise,
        client_id=COMFY_CLIENT_ID,
    )
    JOBS[prompt_id] = {
        "session_id": session_id,
        "kind": "background",
        "provenance": {
            "kind": "background", "prompt": prompt, "seed": seed,
            "checkpoint": CHECKPOINT_NAME, "controlnet": CONTROLNET_NAME,
            "controlnet_strength": controlnet_strength, "denoise": denoise,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    await send_json_to(session_id, {"type": "queued", "prompt_id": prompt_id})


async def handle_edit_region(session_id: str, msg: dict):
    """Regenerates only a user-drawn region of the CURRENT composite (not
    the original photo) — used by the "place an object" draw tool. Unlike
    generate_background, the mask here already means "regenerate this"
    directly (no inversion), since it's drawn by the user with that
    meaning, not derived from a subject-segmentation mask.

    ControlNet depth defaults to near-zero here (unlike generate_background's
    0.75): the depth map reflects the scene *before* the new object exists,
    so conditioning on it fights the model trying to introduce new geometry
    that wasn't there — empirically this suppressed the object entirely at
    background-regen strength."""
    try:
        composite = b64_to_pil(msg["subject"]).convert("RGB")
        region_mask = b64_to_pil(msg["mask"]).convert("L")
        depth = b64_to_pil(msg["depth"]).convert("RGB")
        label = (msg.get("prompt") or "").strip()
        prompt = f"{label}, seamlessly blended, matching lighting and perspective"
        controlnet_strength = float(msg.get("controlnet_strength", 0.1))
        denoise = float(msg.get("denoise", 1.0))
    except Exception as exc:
        await send_json_to(session_id, {"type": "error", "message": f"bad request: {exc}"})
        return

    prompt_id, seed = await asyncio.to_thread(
        backend.queue_background_generation,
        composite, region_mask, depth, prompt,
        controlnet_strength, denoise,
        client_id=COMFY_CLIENT_ID,
    )
    JOBS[prompt_id] = {
        "session_id": session_id,
        "kind": "region",
        "provenance": {
            "kind": "region", "prompt": prompt, "seed": seed,
            "checkpoint": CHECKPOINT_NAME, "controlnet": CONTROLNET_NAME,
            "controlnet_strength": controlnet_strength, "denoise": denoise,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    await send_json_to(session_id, {"type": "queued", "prompt_id": prompt_id})


async def handle_send_to_spout(session_id: str, msg: dict):
    """Pushes the current flattened composite out over the 'PhotoBooth'
    Spout sender, for a real LED wall / projector on set to pick up as a
    live source — same mechanism as bridge.py's Resolume tie-in, just
    fired manually from the browser instead of an OSC trigger. Still one
    shared Spout output regardless of how many sessions are connected --
    whichever session sends last wins the physical output, same as a real
    LED wall can only show one thing at a time.
    """
    try:
        image = b64_to_pil(msg["image"]).convert("RGBA")
    except Exception as exc:
        await send_json_to(session_id, {"type": "error", "message": f"bad request: {exc}"})
        return
    # set_image() does a PIL cover-fit resize/crop -- CPU-bound, and calling
    # it directly here would block the single event loop thread for its
    # duration, stalling comfy_relay_loop's message processing (and every
    # other connected session's progress/preview delivery) right when
    # multi-session routing is supposed to keep sessions independent.
    await asyncio.to_thread(photobooth_frame_buffer.set_image, image)
    await send_json_to(session_id, {"type": "spout_sent"})


def main():
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
