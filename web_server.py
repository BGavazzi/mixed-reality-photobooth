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
import os
import struct
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import websockets
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, ImageOps, UnidentifiedImageError

import brand_kit
from backends.comfy import (
    ComfyBackend,
    DEFAULT_PHOTOSHOOT_BG_WORKFLOW_PATH,
    PHOTOSHOOT_NEGATIVE_PROMPT_NODE,
)
from brand_kit import BrandKitError
from console_encoding import use_utf8_console
from spout_output import SpoutFrameBuffer, spout_sender_loop
import photoshoot_pipeline

# Env-configurable so ComfyUI can live on another box (a GPU workstation on
# the same LAN) without editing source. The relay URL, the REST backend, and
# the /prompt submissions all derive from this one value.
COMFY_ADDRESS = os.environ.get("COMFY_ADDRESS", "127.0.0.1:8188")
COMFY_CLIENT_ID = "web-photoshoot-bridge"
BINARY_PREVIEW_EVENT = 1  # ComfyUI's BinaryEventTypes.PREVIEW_IMAGE

# Resolved against this file, not the process's working directory -- the
# route used to serve the literal relative path "web/index.html", so
# starting the server from anywhere but the repo root returned a 404 for
# the entire app with no hint as to why.
INDEX_HTML_PATH = Path(__file__).parent / "web" / "index.html"

# Rejected before decoding rather than after. Pillow will happily start
# allocating for a huge upload, and a decompression-bomb PNG can exhaust
# memory during Image.open() itself -- this is a local demo, but "one bad
# upload takes the server down mid-shoot" is a bad failure mode regardless.
MAX_UPLOAD_BYTES = 40 * 1024 * 1024


def _read_model_names_from_workflow(workflow_path) -> tuple[str, str]:
    """Reads the checkpoint/ControlNet names straight out of the workflow
    JSON that's actually used at generation time, by node class_type rather
    than a hardcoded node ID (more robust to the workflow being re-exported
    with different node numbering). Previously these were separate
    hardcoded constants that duplicated the workflow file — if someone
    swapped the model inside the workflow JSON without also updating the
    constants here, every provenance record would silently keep reporting
    the old model name even though a different one produced the image."""
    with open(workflow_path) as f:
        workflow = json.load(f)
    checkpoint_name = controlnet_name = None
    for node in workflow.values():
        class_type = node.get("class_type")
        if class_type == "CheckpointLoaderSimple":
            checkpoint_name = node["inputs"]["ckpt_name"]
        elif class_type == "ControlNetLoader":
            controlnet_name = node["inputs"]["control_net_name"]
    # Warn rather than raise: a missing name only degrades the provenance
    # record, and refusing to import the module over it would take the whole
    # app down for a cosmetic field. Silence, though, would let every
    # disclosure card quietly read "checkpoint: null".
    if checkpoint_name is None or controlnet_name is None:
        print(f"[provenance] warning: {workflow_path} has no "
              f"{'CheckpointLoaderSimple' if checkpoint_name is None else 'ControlNetLoader'} node; "
              f"generated images will have an incomplete provenance record")
    return checkpoint_name, controlnet_name


CHECKPOINT_NAME, CONTROLNET_NAME = _read_model_names_from_workflow(DEFAULT_PHOTOSHOOT_BG_WORKFLOW_PATH)


def _read_base_negative_prompt(workflow_path) -> str:
    """Reads the quality-guard negative prompt out of the workflow.

    Brand kits *extend* this rather than replace it: the baked string handles
    generic failure modes (blurry, extra limbs, visible seams) that have
    nothing to do with any client, and a brand manager writing a brand.json
    should not have to re-type them to avoid losing them. Reading it here
    instead of duplicating it as a constant keeps one copy -- editing the
    workflow in ComfyUI's own UI stays the way to change it.
    """
    try:
        with open(workflow_path) as f:
            workflow = json.load(f)
        return workflow[PHOTOSHOOT_NEGATIVE_PROMPT_NODE]["inputs"]["text"]
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        print(f"[brands] warning: could not read the base negative prompt from "
              f"{workflow_path} ({exc}); brand kits will contribute theirs alone")
        return ""


BASE_NEGATIVE_PROMPT = _read_base_negative_prompt(DEFAULT_PHOTOSHOOT_BG_WORKFLOW_PATH)

# Loaded once at import. A brand pack is static configuration for the length
# of an event, and reloading it per request would mean a half-saved JSON edit
# could take out a generation mid-shoot.
BRANDS = brand_kit.load_brands()
print(f"[brands] loaded {len(BRANDS)} brand kit(s): {', '.join(BRANDS) or 'none'}")

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
# Set by the sender thread once it's genuinely publishing. Without this the
# "Send to Spout" button reported success even on a machine where SpoutGL
# isn't installed at all, which is the one situation where the user most
# needs to be told why no source shows up in Resolume.
spout_live_event = threading.Event()


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
        args=(photobooth_frame_buffer, PHOTOBOOTH_SPOUT_NAME, PHOTOBOOTH_SPOUT_FPS,
              spout_stop_event, spout_live_event),
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
        await _fail_orphaned_jobs("lost the connection to ComfyUI mid-generation")
        await asyncio.sleep(2)


async def _fail_orphaned_jobs(reason: str):
    """Anything still in JOBS when the ComfyUI websocket drops can never
    reach a terminal event: ComfyUI doesn't replay events on reconnect, so
    those prompt_ids are unobservable from here even if the GPU goes on to
    finish them. Left alone they were a double failure -- the JOBS entries
    accumulated for the life of the process, and every affected browser sat
    on a disabled Generate button forever waiting for a `done` that had
    already been missed."""
    global executing_prompt_id
    if not JOBS:
        return
    orphaned = list(JOBS.items())
    JOBS.clear()
    executing_prompt_id = None
    for prompt_id, job in orphaned:
        print(f"[relay] orphaning job {prompt_id} ({reason})")
        await send_json_to(job["session_id"], {"type": "error", "message": reason})


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
            # None node for a job we're tracking means this run just
            # finished. Every exit from here must tell the client something:
            # an unhandled exception used to propagate out of this coroutine
            # into comfy_relay_loop's `async for`, tearing down and
            # reconnecting the ComfyUI websocket while the browser sat on a
            # disabled Generate button waiting for a "done" that could never
            # arrive. A silent `image is None` had the same effect.
            try:
                image = await asyncio.to_thread(backend.get_result_image, prompt_id)
            except Exception as exc:
                print(f"[relay] fetching result for {prompt_id} failed: {exc!r}")
                image = None
            if image is not None:
                await send_json_to(session_id, {
                    "type": "done",
                    "image_base64": pil_to_b64(image),
                    "kind": job["kind"],
                    "provenance": job["provenance"],
                })
            else:
                await send_json_to(session_id, {
                    "type": "error",
                    "message": "generation finished but its output image could not be retrieved from ComfyUI",
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
    return FileResponse(INDEX_HTML_PATH)


@app.get("/api/config")
def config():
    """Everything the browser needs that only the server knows.

    `comfy_address` is here because COMFY_ADDRESS is server-side
    configuration -- ComfyUI may well be on a GPU box across the LAN, and
    hardcoding a link to 127.0.0.1:8188 in the page would send every operator
    to their own machine instead. The browser rewrites a loopback address to
    whatever host it reached *this* server on, which is the only address it
    can be sure is reachable from where it is sitting.
    """
    return JSONResponse({
        "comfy_address": COMFY_ADDRESS,
        "base_negative_prompt": BASE_NEGATIVE_PROMPT,
        "brands": [b.to_dict() for b in BRANDS.values()],
    })


@app.get("/api/brands/{brand_id}/logo")
def brand_logo(brand_id: str):
    """Serves a brand's logo artwork.

    Note what this route does *not* do: it never joins a client-supplied
    string onto a filesystem path. `brand_id` is only ever used as a
    dictionary key, and the path comes from the already-validated kit -- so
    there is no traversal to defend against rather than a defence to get
    right.
    """
    brand = BRANDS.get(brand_id)
    if brand is None or brand.logo is None:
        raise HTTPException(status_code=404, detail=f"no logo for brand {brand_id!r}")
    return FileResponse(brand.logo_path)


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty upload")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"photo is {len(raw) // (1024 * 1024)}MB; the limit is {MAX_UPLOAD_BYTES // (1024 * 1024)}MB",
        )
    try:
        # EXIF orientation matters here specifically: phone cameras store
        # portrait shots as landscape pixels plus a rotation flag, so
        # without exif_transpose the pose/depth/rotoscope stages all run on
        # a sideways subject -- and the browser, which honours the flag when
        # displaying the same file, would show an upright photo whose
        # extracted layers are inexplicably rotated 90 degrees.
        image = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        # Previously an unhandled exception here returned a 500 with a
        # traceback; the browser's only clue was "analysis failed".
        raise HTTPException(status_code=400, detail=f"could not read that file as an image: {exc}") from exc

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


async def _queue_job(session_id: str, kind: str, submit, provenance: dict):
    """Shared submit-and-track path for every generation action.

    The three handlers below were the same eleven lines of bookkeeping three
    times over -- pre-register, submit, roll back on failure, attach
    provenance, notify -- differing only in what they submit. Keeping the
    ordering correct in one place matters more than usual here, because two
    separate bugs already lived in it (see the comments below); triplicating
    it meant triplicating both.

    `submit` is a zero-arg callable returning (prompt_id, seed); it runs in a
    worker thread because the backend's HTTP calls are blocking.
    """
    # Registered *before* the submit call, with our own pre-generated
    # prompt_id, not after -- the ComfyUI relay is a concurrent task and can
    # start delivering events for a same-instant-queued job before this
    # coroutine's own await returns; without pre-registering, that first
    # event would look up a JOBS entry that doesn't exist yet and be
    # silently dropped. ComfyUI's /prompt honors a client-supplied prompt_id
    # and echoes it back unchanged (confirmed empirically).
    prompt_id = str(uuid.uuid4())
    JOBS[prompt_id] = {"session_id": session_id, "kind": kind, "provenance": None}
    try:
        _, seed = await asyncio.to_thread(submit, prompt_id)
    except Exception as exc:
        # Pre-registering means a failed submission (ComfyUI unreachable,
        # bad workflow) would otherwise leave this JOBS entry orphaned for
        # the life of the process -- roll it back and tell the client.
        JOBS.pop(prompt_id, None)
        await send_json_to(session_id, {"type": "error", "message": f"failed to queue generation: {exc}"})
        return
    JOBS[prompt_id]["provenance"] = {
        "kind": kind, "seed": seed, "checkpoint": CHECKPOINT_NAME,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **provenance,
    }
    await send_json_to(session_id, {"type": "queued", "prompt_id": prompt_id})


async def handle_generate_image(session_id: str, msg: dict):
    prompt = (msg.get("prompt") or "").strip()
    if not prompt:
        return
    await _queue_job(
        session_id, "image",
        lambda prompt_id: backend.queue_image_generation(prompt, COMFY_CLIENT_ID, prompt_id),
        {"prompt": prompt, "controlnet": None, "controlnet_strength": None, "denoise": None},
    )


def _compose_for_request(msg: dict) -> brand_kit.ComposedPrompt:
    """Applies the requested brand kit to an incoming generation message.

    This is the enforcement point, and it is on the server for a reason: the
    browser sends a brand id and a look id, never the finished prompt. A
    client that has been modified -- or simply left open on an older version
    of the page -- therefore cannot strip the brand's locked negative or
    quietly widen its own free text into the mandated styling. The worst it
    can do is name a brand that doesn't exist, which is an error, not a
    silent downgrade.
    """
    brand_id = msg.get("brand_id") or None
    brand = None
    if brand_id:
        brand = BRANDS.get(brand_id)
        if brand is None:
            raise BrandKitError(f"unknown brand kit {brand_id!r}")
    return brand_kit.compose(
        brand,
        msg.get("look_id") or None,
        msg.get("prompt") or "",
        base_negative=BASE_NEGATIVE_PROMPT,
    )


async def handle_generate_background(session_id: str, msg: dict):
    try:
        subject = b64_to_pil(msg["subject"]).convert("RGB")
        mask = b64_to_pil(msg["mask"]).convert("L")
        depth = b64_to_pil(msg["depth"]).convert("RGB")
        # UI sends the subject mask (255=subject); the workflow needs the
        # inverse (255=background=regenerate).
        background_mask = ImageOps.invert(mask)
        composed = _compose_for_request(msg)
        controlnet_strength = float(msg.get("controlnet_strength", 0.75))
        denoise = float(msg.get("denoise", 0.85))
    except Exception as exc:
        await send_json_to(session_id, {"type": "error", "message": f"bad request: {exc}"})
        return

    if not composed.positive:
        await send_json_to(session_id, {
            "type": "error",
            "message": "nothing to generate: pick an approved look or type a scene prompt",
        })
        return

    await _queue_job(
        session_id, "background",
        lambda prompt_id: backend.queue_background_generation(
            subject, background_mask, depth, composed.positive, controlnet_strength, denoise,
            client_id=COMFY_CLIENT_ID, prompt_id=prompt_id,
            negative_prompt=composed.negative, seed=composed.seed,
        ),
        {"prompt": composed.positive, "controlnet": CONTROLNET_NAME,
         "controlnet_strength": controlnet_strength, "denoise": denoise,
         **composed.to_provenance()},
    )


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
    background-regen strength.

    A brand kit's *negative* applies here, but its positive styling does not.
    The distinction is deliberate: "never generate a competitor's logo" is a
    rule about every pixel in the frame, while "rugged outdoor lifestyle
    photography, film grain" describes a scene and reads as noise when
    what's being generated is a single prop dropped into an existing one.
    This is also the highest-risk input in the whole app -- the region label
    is free text typed live at a booth, so it is the one prompt a brand's
    blocklist most needs to reach."""
    try:
        composite = b64_to_pil(msg["subject"]).convert("RGB")
        region_mask = b64_to_pil(msg["mask"]).convert("L")
        depth = b64_to_pil(msg["depth"]).convert("RGB")
        label = (msg.get("prompt") or "").strip()
        prompt = f"{label}, seamlessly blended, matching lighting and perspective"
        # look_id is dropped on purpose -- see the docstring; only the
        # blocklist half of the kit is wanted here.
        composed = _compose_for_request({"brand_id": msg.get("brand_id"), "prompt": ""})
        controlnet_strength = float(msg.get("controlnet_strength", 0.1))
        denoise = float(msg.get("denoise", 1.0))
    except Exception as exc:
        await send_json_to(session_id, {"type": "error", "message": f"bad request: {exc}"})
        return

    await _queue_job(
        session_id, "region",
        lambda prompt_id: backend.queue_background_generation(
            composite, region_mask, depth, prompt, controlnet_strength, denoise,
            client_id=COMFY_CLIENT_ID, prompt_id=prompt_id,
            negative_prompt=composed.negative,
        ),
        {"prompt": prompt, "controlnet": CONTROLNET_NAME,
         "controlnet_strength": controlnet_strength, "denoise": denoise,
         "brand": composed.brand_name, "brand_id": composed.brand_id,
         "brand_version": composed.brand_version,
         "negative_prompt": composed.negative, "operator_text": label},
    )


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
    if not spout_live_event.is_set():
        await send_json_to(session_id, {
            "type": "error",
            "message": (f"the composite was staged, but the '{PHOTOBOOTH_SPOUT_NAME}' Spout sender "
                        f"is not running (see the server log) — no receiver will see it"),
        })
        return
    await send_json_to(session_id, {"type": "spout_sent"})


def main():
    import uvicorn
    use_utf8_console()  # prompts are logged verbatim and can contain non-ASCII
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
