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
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import websockets
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from PIL import Image, ImageOps, UnidentifiedImageError

import attention
import batch
import brand_kit
import config
import consent
import job_queue
import obs
import resilience
import watchdog
import workflow_graph
from backends.comfy import ComfyBackend, DEFAULT_PHOTOSHOOT_BG_WORKFLOW_PATH
from brand_kit import BrandKitError
from console_encoding import use_utf8_console
from spout_output import SpoutFrameBuffer, spout_sender_loop
import photoshoot_pipeline

# Env-configurable so ComfyUI can live on another box (a GPU workstation on
# the same LAN) without editing source. The relay URL, the REST backend, and
# the /prompt submissions all derive from this one value.
COMFY_ADDRESS = config.env_str("COMFY_ADDRESS", "127.0.0.1:8188",
                               "host:port of the ComfyUI this app drives")
# Unique per process, and deliberately so. ComfyUI keys its websocket clients
# by clientId and keeps only the most recent socket per id, so two instances of
# this app sharing one constant meant the second to connect silently stole the
# first's events: the older instance's generations still ran to completion, but
# it never heard about them and its browser sat on "queued..." forever. That is
# not exotic -- it happens the moment the Docker image is run alongside a
# native `python web_server.py`, which is exactly how you'd compare the two.
# Override only if something external needs to predict the id.
COMFY_CLIENT_ID = config.env_str(
    "COMFY_CLIENT_ID", f"web-photoshoot-bridge-{uuid.uuid4().hex[:8]}",
    "websocket client id; unique per process unless pinned")
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

# A batch is bounded for the same reason the queue is: fifty photos is
# already ~30 minutes of serial GPU time, and accepting a thousand would be
# promising something the machine cannot deliver in any useful timeframe.
MAX_BATCH_FILES = 50

SHUTDOWN_DRAIN_SECONDS = config.env_float(
    "SHUTDOWN_DRAIN_SECONDS", 30.0,
    "seconds a shutdown waits for queued work to reach ComfyUI",
    minimum=0, maximum=3600)

# Three separate facts, deliberately not one "healthy" flag.
#
# MODELS_READY: the ~40s of rotoscope/pose/depth loading at startup. Until it
#   is set the process answers HTTP but every generation would block on the
#   import lock, which looks like a hang rather than a warm-up.
# READY: the composite answer /readyz gives. Cleared first at shutdown so a
#   load balancer stops sending work *before* anything is torn down, which is
#   the entire point of having readiness separate from liveness.
# ACCEPTING: whether new work may be admitted. A draining server is alive and
#   still finishing what it has -- it just must not take more.
MODELS_READY = threading.Event()
READY = threading.Event()
ACCEPTING = threading.Event()
ACCEPTING.set()


def _read_workflow_facts(workflow_path) -> tuple[str | None, str | None, str]:
    """Reads the model names and the quality-guard negative prompt out of the
    workflow that actually runs at generation time.

    All three used to be hardcoded constants duplicating the workflow file --
    swap a model inside the JSON without editing here and every provenance
    record would keep reporting the old one. They are now read from the file,
    and the negative prompt is located by *role* rather than by node id, so a
    re-export can't repoint it at the positive conditioning (see
    workflow_graph.py).

    Warns rather than raises: an unreadable workflow here only degrades the
    disclosure card and the brand-kit base negative, and taking the whole app
    down at import over a cosmetic field would be the worse trade. Silence
    would not -- that would let every card quietly read "checkpoint: null".
    """
    try:
        with open(workflow_path) as f:
            workflow = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[provenance] warning: could not read {workflow_path} ({exc}); "
              f"provenance records will be incomplete and brand kits will "
              f"contribute their blocklist alone")
        return None, None, ""

    names = workflow_graph.model_names(workflow)
    if names["checkpoint"] is None or names["controlnet"] is None:
        print(f"[provenance] warning: {workflow_path} has no "
              f"{'CheckpointLoaderSimple' if names['checkpoint'] is None else 'ControlNetLoader'} "
              f"node; generated images will have an incomplete provenance record")

    # Resolved separately, and allowed to fail on its own: reading a model
    # name only needs a loader node, while locating the negative prompt needs
    # the graph to be drivable. A workflow too broken for the second is still
    # worth reporting the first from.
    negative = ""
    try:
        resolved = workflow_graph.resolve(workflow, source=workflow_path)
        if resolved.has(workflow_graph.NEGATIVE_PROMPT):
            node_id = resolved.node_id(workflow_graph.NEGATIVE_PROMPT)
            negative = resolved.workflow[node_id]["inputs"].get("text", "")
    except workflow_graph.WorkflowSchemaError as exc:
        print(f"[brands] warning: no negative prompt resolved from {workflow_path} "
              f"({exc}); brand kits will contribute their blocklist alone")

    return names["checkpoint"], names["controlnet"], negative


CHECKPOINT_NAME, CONTROLNET_NAME, BASE_NEGATIVE_PROMPT = _read_workflow_facts(
    DEFAULT_PHOTOSHOOT_BG_WORKFLOW_PATH)

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


async def retention_loop(interval_seconds: float = 3600):
    """Expires old batch runs, hourly.

    The startup sweep matters more than the loop: a booth that crashed
    mid-evening leaves a directory of photographs that nothing in the app
    remembers, so nothing would ever delete them. An hourly tick then keeps a
    long-running instance honest without waiting for a restart.
    """
    while True:
        try:
            await asyncio.to_thread(batch.sweep_expired)
        except Exception as exc:                    # noqa: BLE001
            # A failing sweep must not take the server down, but it must be
            # noticed: silently retaining faces is exactly the failure this
            # whole feature exists to prevent.
            print(f"[batch] retention sweep failed: {exc!r}")
            attention.raise_item(
                attention.DEPENDENCY_DOWN, "retention sweep is failing",
                detail=f"{exc!r} -- batch photos may be retained past their expiry")
        await asyncio.sleep(interval_seconds)


async def _probe_comfy(prompt_id: str):
    """What does ComfyUI say about a prompt that has gone quiet?

    Ordered by consequence. `/history` first, because "it finished and we
    missed the event" is both the most likely cause and the only one where
    getting it wrong destroys a picture that exists. Only if history has
    nothing do we ask whether it is still queued.
    """
    image = await asyncio.to_thread(backend.get_result_image, prompt_id)
    if image is not None:
        return watchdog.FINISHED, image
    landed = await asyncio.to_thread(backend._prompt_landed, prompt_id)
    return (watchdog.QUEUED if landed else watchdog.UNKNOWN), None


async def _recover_job(prompt_id: str, image: Image.Image):
    job = JOBS.pop(prompt_id, None)
    if job is None:
        return
    await _deliver_result(job, prompt_id, image)
    # Worth an operator's attention even though it ended well: a recovered job
    # means events were lost, and losing events silently is how the clientId
    # collision went unnoticed for a day. Low severity, because nobody has to
    # do anything -- but it should be visible that it happened.
    attention.raise_item(
        attention.BATCH_ITEM_FAILED, "a render finished but its event was missed",
        detail=f"recovered {prompt_id} from /history; check for a second process "
               f"sharing COMFY_CLIENT_ID, or a websocket that reconnected mid-render")


async def _abandon_job(prompt_id: str, item: watchdog.Overdue):
    job = JOBS.pop(prompt_id, None)
    if job is None:
        return
    global executing_prompt_id
    if executing_prompt_id == prompt_id:
        executing_prompt_id = None
    await _report_job_error(job, prompt_id, item.message)
    if item.reason == "stall":
        attention.raise_item(
            attention.DEPENDENCY_DOWN, "ComfyUI accepted work and went silent",
            detail=f"no events for {watchdog.STALL_SECONDS:.0f}s with jobs in flight; "
                   f"{prompt_id} was given up after {item.age:.0f}s")


async def watchdog_loop():
    """The only loop here that exists because of something that does *not*
    happen. Failures announce themselves; a job that is simply never spoken of
    again does not, so it takes a timer to notice."""
    while True:
        await asyncio.sleep(watchdog.INTERVAL_SECONDS)
        try:
            if not JOBS:
                continue
            tally = await watchdog.sweep(
                dict(JOBS), probe=_probe_comfy, recover=_recover_job, fail=_abandon_job)
            if tally["checked"]:
                obs.log("watchdog", "swept overdue jobs", **tally)
        except Exception as exc:                        # noqa: BLE001
            # Must not take the server down, must not go unnoticed: this is the
            # component whose whole job is spotting silence.
            obs.log("watchdog", "sweep failed", error=repr(exc))
            attention.raise_item(
                attention.DEPENDENCY_DOWN, "the job watchdog is failing",
                detail=f"{exc!r} -- stuck generations will not be detected")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(config.banner())
    print("[warmup] loading rotoscope/pose/depth models...")
    await asyncio.to_thread(_warm_up_pipeline_models)
    MODELS_READY.set()
    print("[warmup] done")
    swept = await asyncio.to_thread(batch.sweep_expired)
    print(f"[batch] retention: {batch.DEFAULT_RETAIN_DAYS:g} day(s), "
          f"{len(swept)} expired run(s) removed at startup")
    await GENERATION_QUEUE.start()
    background = [
        asyncio.create_task(retention_loop()),
        asyncio.create_task(comfy_relay_loop()),
        asyncio.create_task(watchdog_loop()),
    ]
    threading.Thread(
        target=spout_sender_loop,
        args=(photobooth_frame_buffer, PHOTOBOOTH_SPOUT_NAME, PHOTOBOOTH_SPOUT_FPS,
              spout_stop_event, spout_live_event),
        daemon=True,
    ).start()
    READY.set()

    yield

    await _shutdown(background)


async def _shutdown(background: list[asyncio.Task]):
    """Stop taking work, finish what was taken, then stop.

    The old shutdown was one line -- `await GENERATION_QUEUE.stop()` -- which
    cancels the worker tasks. Anything mid-submission died there and then, and
    anything still waiting in the queue was dropped without so much as an
    error frame: `docker stop` or a Ctrl-C during a batch lost a guest's photo
    and told nobody. `drain()` existed on the queue interface the whole time
    and nothing called it.

    The boundary is worth being precise about, because it is not "wait for
    every photo to be finished". Draining waits for queued work to be *handed
    to ComfyUI*, which is where the app stops being the thing that can lose it:
    once ComfyUI has the prompt it renders regardless of this process, and the
    result is recoverable from /history afterwards. Waiting for renders instead
    would mean a shutdown that takes half an hour on a full queue, which is how
    you teach an operator to use `kill -9`.

    Bounded, because a drain that cannot finish must not become a hang: past
    the deadline the remaining jobs are failed loudly rather than dropped
    silently, which is the whole difference from before.
    """
    READY.clear()          # /readyz starts failing here, before anything is torn down
    ACCEPTING.clear()      # new submissions are refused with a 503, not queued into a dying process
    stats = GENERATION_QUEUE.stats()
    print(f"[shutdown] draining {stats['waiting']} waiting + {stats['running']} running "
          f"(deadline {SHUTDOWN_DRAIN_SECONDS:g}s)")

    started = time.monotonic()
    try:
        await asyncio.wait_for(GENERATION_QUEUE.drain(), timeout=SHUTDOWN_DRAIN_SECONDS)
        print(f"[shutdown] queue drained in {time.monotonic() - started:.1f}s")
    except asyncio.TimeoutError:
        stats = GENERATION_QUEUE.stats()
        print(f"[shutdown] drain deadline passed with {stats['waiting']} job(s) still "
              f"waiting; failing them rather than dropping them")
        await _fail_orphaned_jobs("the server shut down before this generation was submitted")

    await GENERATION_QUEUE.stop()
    for task in background:
        task.cancel()
    await asyncio.gather(*background, return_exceptions=True)
    spout_stop_event.set()
    print("[shutdown] done")


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

    # Every ComfyUI event proves the dependency is alive, whichever job it
    # names. The watchdog reads this to tell "our job is stuck" apart from
    # "ComfyUI is wedged", which are different problems with different fixes.
    watchdog.note_event()

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
                obs.log("relay", "fetching result failed", error=repr(exc))
                image = None
            if image is not None:
                await _deliver_result(job, prompt_id, image)
            else:
                await _report_job_error(
                    job, prompt_id,
                    "generation finished but its output image could not be retrieved from ComfyUI")
            JOBS.pop(prompt_id, None)
            if executing_prompt_id == prompt_id:
                executing_prompt_id = None
        else:
            executing_prompt_id = prompt_id
            await send_json_to(session_id, {"type": "executing", "node": node})

    elif etype == "execution_error":
        await _report_job_error(job, prompt_id, str(data.get("exception_message", "generation failed")))
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
        await _report_job_error(job, prompt_id, "generation was interrupted")
        JOBS.pop(prompt_id, None)
        if executing_prompt_id == prompt_id:
            executing_prompt_id = None


async def _deliver_result(job: dict, prompt_id: str, image: Image.Image):
    """Hands a finished render to whichever sink the job belongs to.

    Two sinks, one path. A batch run outlives the page that started it -- and
    can be started with no page at all -- so its results are written to the run
    directory rather than pushed at a websocket that may not exist.

    Extracted from the relay because the watchdog needs exactly this: a render
    it found sitting in /history after its event was missed has to reach the
    same place by the same route, or "recovered" would quietly mean something
    different from "completed".
    """
    if job.get("batch_run_id"):
        await asyncio.to_thread(_finish_batch_item, job, prompt_id, image)
    else:
        await send_json_to(job["session_id"], {
            "type": "done",
            "image_base64": pil_to_b64(image),
            "kind": job["kind"],
            "provenance": job["provenance"],
        })


async def _report_job_error(job: dict, prompt_id: str, message: str):
    """One error path for both sinks. Before batch mode this was three inline
    `send_json_to(...)` calls; a batch job routed through those would fail
    silently, since its session id belongs to no socket and send_json_to
    no-ops rather than raising."""
    run_id = job.get("batch_run_id")
    if run_id:
        run = batch.RUNS.get(run_id)
        item = run.item_by_prompt(prompt_id) if run else None
        if run and item:
            run.set_status(item, batch.FAILED, message)
            run.write_manifest()
            print(f"[batch] {run_id} item {item.stem} failed: {message}")
            # Summary deliberately excludes the item, so fifty photos failing
            # against one dead ComfyUI merge into one line with a count of
            # fifty instead of burying the panel.
            attention.raise_item(
                attention.BATCH_ITEM_FAILED,
                f"batch run {run_id}: photo(s) failed to generate",
                detail=message, run_id=run_id, last_item=item.filename)
        return
    await send_json_to(job["session_id"], {"type": "error", "message": message})
    attention.raise_item(
        attention.GENERATION_FAILED,
        "a guest's generation failed after retries",
        detail=message, prompt_id=prompt_id, job_kind=job.get("kind"))


def _finish_batch_item(job: dict, prompt_id: str, image: Image.Image):
    """Composites the untouched subject back over its generated background and
    writes the frame. Runs in a thread: it is PIL work plus disk I/O, and
    doing it on the event loop would stall the relay -- which during a batch
    is delivering another item's progress at the same time."""
    run = batch.RUNS.get(job["batch_run_id"])
    if run is None:
        return  # run was deleted mid-flight; the frame has nowhere to go
    item = run.item_by_prompt(prompt_id)
    if item is None:
        return
    try:
        composite = batch.composite_subject_over(image, run.path_for("cutout", item))
        composite.save(run.path_for("output", item))
        with run.lock:
            item.provenance = job["provenance"]
            item.status = batch.DONE
        run.write_manifest()
        print(f"[batch] {run.run_id} item {item.stem} done "
              f"({run.counts()[batch.DONE]}/{len(run.items)})")
    except Exception as exc:
        run.set_status(item, batch.FAILED, f"compositing failed: {exc}")
        run.write_manifest()


def _batch_submit(run: batch.BatchRun, item: batch.BatchItem, composed, settings: dict):
    """The whole per-photo pipeline, run inside a queue worker.

    Analysis lives here rather than in the request handler on purpose: it is
    ~20s of CPU per photo, and doing it up front would mean a fifty-photo
    batch sits silent for fifteen minutes before the GPU sees anything. Inside
    the job, the worker pool overlaps one photo's rotoscope with another
    photo's generation.
    """
    run.set_status(item, batch.ANALYZING)
    raw = run.path_for("input", item, ".orig")
    image = ImageOps.exif_transpose(Image.open(raw)).convert("RGB")
    result = photoshoot_pipeline.analyze(image)

    # Persisted rather than kept in memory: fifty subjects' worth of decoded
    # images waiting on a serial GPU is how a long batch becomes an OOM.
    result["cutout"].save(run.path_for("cutout", item))
    if batch.KEEP_INTERMEDIATES:
        result["image"].save(run.path_for("analyzed", item))

    run.set_status(item, batch.GENERATING)
    # The generation is conditioned on the analysed image, but that copy stays
    # in this worker's memory for the length of the call -- it does not need to
    # exist on disk, and neither does the original photograph now that a cutout
    # exists. Deleted here rather than at the end of the run because the window
    # in which the app holds a stranger's full photograph is the thing being
    # minimised, and a fifty-photo batch runs for half an hour.
    run.drop_original(item)
    return backend.queue_background_generation(
        result["image"],
        ImageOps.invert(result["mask"].convert("L")),
        result["depth"].convert("RGB"),
        composed.positive,
        settings["controlnet_strength"],
        settings["denoise"],
        client_id=COMFY_CLIENT_ID,
        prompt_id=item.prompt_id,
        negative_prompt=composed.negative,
        seed=composed.seed,
    )


# --- HTTP routes --------------------------------------------------------------

@app.get("/")
def index():
    return FileResponse(INDEX_HTML_PATH)


@app.get("/healthz")
def healthz():
    """Is the process alive? Nothing else.

    Deliberately checks nothing: a liveness probe that consults a dependency
    is a restart loop waiting for that dependency to have a bad minute.
    ComfyUI being down is not a reason to kill this process -- it is the exact
    situation the breaker, the attention queue and the guest-facing message
    were built for, and all of them need the process to still be running.
    """
    return JSONResponse({"status": "alive", **config.version_info()})


@app.get("/readyz")
def readyz():
    """Should this instance be given work right now?

    Every "no" carries a reason, because a probe that fails without saying why
    turns into somebody reading source at 2am. 503 rather than 200-with-a-flag
    so the answer is legible to anything that speaks HTTP and nothing else.
    """
    breaker = getattr(backend, "breaker", None)
    breaker_stats = breaker.stats() if breaker else {}
    queue_stats = GENERATION_QUEUE.stats()

    reasons = []
    if not MODELS_READY.is_set():
        reasons.append("the CPU pipeline models are still loading")
    if not READY.is_set():
        reasons.append("the server has not finished starting, or is shutting down")
    if not ACCEPTING.is_set():
        reasons.append("the server is draining and is not accepting new work")
    if breaker_stats.get("state") == "open":
        reasons.append("ComfyUI is not answering; the circuit breaker is open")
    if queue_stats["waiting"] >= queue_stats["max_depth"]:
        reasons.append(f"the queue is full ({queue_stats['waiting']} waiting)")

    payload = {
        "ready": not reasons,
        "reasons": reasons,
        **config.version_info(),
        "queue": queue_stats,
        "breaker": breaker_stats,
        "attention": {"open": attention.snapshot()["open"]},
        "jobs_in_flight": len(JOBS),
        "comfy_silent_for": round(watchdog.silence(), 1) if JOBS else None,
        "settings": config.describe(),
    }
    return JSONResponse(payload, status_code=200 if not reasons else 503)


@app.get("/api/config")
def browser_config():
    """Everything the browser needs that only the server knows.

    Named `browser_config` rather than `config` because the module of that name
    is imported here; the route path is unchanged. FastAPI never uses the
    function name for routing, but Python does -- as a shadowed global it made
    every later `config.env_int(...)` an AttributeError at import.

    `comfy_address` is here because COMFY_ADDRESS is server-side
    configuration -- ComfyUI may well be on a GPU box across the LAN, and
    hardcoding a link to 127.0.0.1:8188 in the page would send every operator
    to their own machine instead. The browser rewrites a loopback address to
    whatever host it reached *this* server on, which is the only address it
    can be sure is reachable from where it is sitting.
    """
    return JSONResponse({
        **config.version_info(),
        "comfy_address": COMFY_ADDRESS,
        "base_negative_prompt": BASE_NEGATIVE_PROMPT,
        "brands": [b.to_dict() for b in BRANDS.values()],
        "queue": GENERATION_QUEUE.stats(),
        # Served rather than duplicated in the page, so the list an operator
        # picks from and the list the server accepts cannot drift apart.
        "consent_bases": consent.options(),
        # Off by default. The page uses this to decide whether the consent
        # fields block the file picker or merely offer to record something;
        # both states are the same form, so an operator who wants the record
        # never has to be told about a flag.
        "consent_required": consent.REQUIRED,
        "retention": {"retain_days": batch.DEFAULT_RETAIN_DAYS,
                      "keep_intermediates": batch.KEEP_INTERMEDIATES},
    })


@app.post("/api/batch")
async def start_batch(
    files: list[UploadFile] = File(...),
    brand_id: str = Form(""),
    look_id: str = Form(""),
    prompt: str = Form(""),
    controlnet_strength: float = Form(0.75),
    denoise: float = Form(0.85),
    consent_basis: str = Form(""),
    consent_by: str = Form(""),
    consent_note: str = Form(""),
):
    """Starts a batch: N photos, one approved look, one consistent set.

    Returns as soon as the work is queued rather than when it is done -- a
    fifty-photo run is half an hour of GPU time, which is not an HTTP request.
    Poll GET /api/batch/{run_id} and download the zip when it reports finished.
    """
    if not ACCEPTING.is_set():
        # 503 before reading a single upload. A draining server that accepted
        # fifty photographs and then dropped them would be worse than one that
        # never took them, and it would write them to disk on the way.
        raise HTTPException(
            status_code=503,
            detail="the server is shutting down and is not accepting new batches")
    if not files:
        raise HTTPException(status_code=400, detail="no files uploaded")
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(
            status_code=413,
            detail=f"{len(files)} photos; the limit is {MAX_BATCH_FILES} per run")

    # Checked before a single byte is written. A run that is going to be
    # rejected for a half-filled consent form should not first spend thirty
    # seconds putting fifty photographs of strangers on disk. With the gate off
    # (the default) an empty declaration is recorded as `not_recorded` and the
    # run proceeds -- see consent.py for why that default is what it is.
    try:
        consent_record = consent.parse(consent_basis, consent_by, consent_note)
    except consent.ConsentError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        composed = _compose_for_request({"brand_id": brand_id, "look_id": look_id, "prompt": prompt})
    except BrandKitError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not composed.positive:
        raise HTTPException(
            status_code=400,
            detail="nothing to generate: pick an approved look or supply a scene prompt")

    run = batch.create_run([f.filename or f"photo_{i}" for i, f in enumerate(files)],
                           composed.brand_id, composed.look_id, composed.look_label,
                           consent=consent_record.to_dict())

    # Uploads are read and written to disk here, in the request, because the
    # UploadFile objects are only valid for its lifetime -- a queue worker
    # picking one up later would find a closed file.
    for item, upload in zip(run.items, files):
        raw = await upload.read()
        if len(raw) > MAX_UPLOAD_BYTES:
            run.set_status(item, batch.FAILED,
                           f"{len(raw) // (1024 * 1024)}MB exceeds the "
                           f"{MAX_UPLOAD_BYTES // (1024 * 1024)}MB limit")
            continue
        try:
            # Validated now rather than inside the worker: a directory of
            # mixed files should reject its non-images immediately, not
            # twenty minutes into a run.
            Image.open(io.BytesIO(raw)).verify()
        except Exception as exc:
            run.set_status(item, batch.FAILED, f"not a readable image: {exc}")
            continue
        run.path_for("input", item, ".orig").write_bytes(raw)

    settings = {"controlnet_strength": float(controlnet_strength), "denoise": float(denoise)}
    queued = 0
    for item in run.items:
        if item.status == batch.FAILED:
            continue
        item.prompt_id = str(uuid.uuid4())
        JOBS[item.prompt_id] = {
            "session_id": f"batch:{run.run_id}",  # no socket; keeps the shape uniform
            "kind": "background",
            "provenance": None,
            "batch_run_id": run.run_id,
        }
        job = job_queue.GenerationJob(
            session_id=f"batch:{run.run_id}",
            kind="background",
            submit=lambda _pid, it=item: _batch_submit(run, it, composed, settings),
            provenance={"prompt": composed.positive, "controlnet": CONTROLNET_NAME,
                        "controlnet_strength": settings["controlnet_strength"],
                        "denoise": settings["denoise"], **composed.to_provenance()},
            job_id=item.prompt_id,
        )
        try:
            await GENERATION_QUEUE.submit(job)
            queued += 1
        except job_queue.QueueFullError as exc:
            JOBS.pop(item.prompt_id, None)
            run.set_status(item, batch.FAILED, str(exc))

    run.write_manifest()
    print(f"[batch] {run.run_id}: queued {queued}/{len(run.items)} photo(s), "
          f"brand={composed.brand_id} look={composed.look_id} seed={composed.seed}")
    return JSONResponse({**run.to_dict(), "queued": queued}, status_code=202)


@app.get("/api/batch/{run_id}")
def batch_status(run_id: str):
    run = batch.RUNS.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"no batch run {run_id!r}")
    return JSONResponse(run.to_dict())


@app.get("/api/batch/{run_id}/download")
def batch_download(run_id: str):
    """The finished frames plus the manifest, as a zip.

    Allowed before the run finishes on purpose: an operator who has to leave
    should be able to take what is ready rather than lose it.
    """
    run = batch.RUNS.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"no batch run {run_id!r}")
    if run.counts()[batch.DONE] == 0:
        raise HTTPException(status_code=409, detail="no frames have finished yet")
    buffer = batch.zip_run(run)
    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="batch_{run_id}.zip"'},
    )


@app.delete("/api/batch/{run_id}")
def batch_delete(run_id: str):
    """Deletes a run and its files -- including the subjects' photographs,
    which is why this exists rather than letting a booth accumulate them."""
    if not batch.delete_run(run_id):
        raise HTTPException(status_code=404, detail=f"no batch run {run_id!r}")
    return JSONResponse({"deleted": run_id})


@app.get("/api/queue")
def queue_stats():
    """Live queue depth and dependency health. Separate from /api/config --
    config is read once at page load, this is the bit that changes, and a
    batch run polls it.

    The breaker's state is here rather than in its own endpoint because "is
    there a queue?" and "is ComfyUI answering?" are the same question from the
    operator's side: both explain why nothing is coming out.
    """
    stats = GENERATION_QUEUE.stats()
    breaker = getattr(backend, "breaker", None)
    if breaker is not None:
        stats["comfyui"] = breaker.stats()
    return JSONResponse(stats)


@app.get("/api/attention")
def attention_queue():
    """What currently needs a person. Polled by the header badge."""
    return JSONResponse(attention.snapshot())


@app.post("/api/attention/{item_id}/resolve")
def attention_resolve(item_id: int, by: str = Form("operator")):
    """Marks one item handled. Deliberately one at a time and attributed:
    resolving is a record of who looked at what, and a "clear all" button is
    how that record stops existing."""
    item = attention.resolve(item_id, by=by)
    if item is None:
        raise HTTPException(status_code=404, detail=f"no open attention item {item_id}")
    return JSONResponse(item.to_dict())


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


async def _on_job_accepted(job: job_queue.GenerationJob, prompt_id: str, seed: int):
    """A worker got the job into ComfyUI. Attach provenance and tell the
    browser it is really queued -- not when it was *submitted to us*, which
    is what the client used to be told."""
    # ComfyUI just answered, so any standing "it is unreachable" alert is
    # stale. Closing it here rather than waiting for an operator keeps the
    # panel trustworthy: an alert that outlives its problem is how people
    # learn to ignore the panel entirely.
    attention.resolve_kind(attention.DEPENDENCY_DOWN, by="recovered")

    entry = JOBS.get(prompt_id)
    if entry is None:
        # The session vanished and cleanup already ran; nothing to attach to.
        return
    # The watchdog's clock starts here, not when the job was registered:
    # waiting in *our* queue is bounded by admission control and reported to
    # the client as a position, and timing that out would fail exactly the
    # work the queue exists to hold. What has no bound until now is the time
    # between ComfyUI accepting a prompt and saying anything about it again.
    entry["accepted_at"] = time.monotonic()
    entry["provenance"] = {
        "kind": job.kind, "seed": seed, "checkpoint": CHECKPOINT_NAME,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **job.provenance,
    }
    await send_json_to(job.session_id, {
        "type": "queued", "prompt_id": prompt_id, "waited_seconds": round(job.waited, 1),
    })


async def _on_job_failed(job: job_queue.GenerationJob, exc: Exception):
    """Submission failed (ComfyUI unreachable, bad workflow). The JOBS entry
    was registered before submission, so it has to be rolled back or it sits
    there for the life of the process while the browser waits on a `done`
    that can never arrive."""
    JOBS.pop(job.job_id, None)

    # An open breaker is a different sentence to a guest than one bad render:
    # it means the booth is down, not that their photo was unlucky, and it is
    # the case where an operator can actually do something.
    if isinstance(exc, resilience.CircuitOpenError):
        message = ("the render service is not responding — an operator has been "
                   "alerted, your photo is safe")
        attention.raise_item(
            attention.DEPENDENCY_DOWN, "ComfyUI is not responding",
            detail=str(exc), breaker=getattr(backend, "breaker", None)
            and backend.breaker.stats())
    else:
        message = f"failed to queue generation: {exc}"
        attention.raise_item(
            attention.GENERATION_FAILED, "a generation could not be queued",
            detail=str(exc), job_kind=job.kind)

    await send_json_to(job.session_id, {"type": "error", "message": message})
    if job.session_id.startswith("batch:"):
        # A batch job has no socket to receive that message, so without this
        # its item would sit in GENERATING until the run was deleted.
        await _report_job_error({"batch_run_id": job.session_id.split(":", 1)[1],
                                 "session_id": job.session_id}, job.job_id, message)


GENERATION_QUEUE = job_queue.InProcessJobQueue(
    on_accepted=_on_job_accepted,
    on_failed=_on_job_failed,
    workers=config.env_int("GENERATION_WORKERS", job_queue.DEFAULT_WORKERS,
                           "concurrent submitters in front of the serial GPU",
                           minimum=1, maximum=32),
    max_depth=config.env_int("GENERATION_QUEUE_DEPTH", job_queue.DEFAULT_MAX_DEPTH,
                             "jobs allowed to wait before submission is refused",
                             minimum=1, maximum=10000),
)


async def _queue_job(session_id: str, kind: str, submit, provenance: dict):
    """Shared enqueue-and-track path for every generation action.

    The three handlers below were the same eleven lines of bookkeeping three
    times over -- pre-register, submit, roll back on failure, attach
    provenance, notify -- differing only in what they submit. Keeping the
    ordering correct in one place matters more than usual here, because two
    separate bugs already lived in it; triplicating it meant triplicating both.

    Submission itself now goes through GENERATION_QUEUE rather than straight
    onto a thread, so there is admission control and a real queue position to
    report. What has *not* changed is the ordering below, which is load-bearing.
    """
    # Registered *before* the job is handed to the queue, with our own
    # pre-generated prompt_id -- the ComfyUI relay is a concurrent task and
    # can start delivering events for a same-instant-queued job before the
    # accept callback runs; without pre-registering, that first event would
    # look up a JOBS entry that doesn't exist yet and be silently dropped.
    # ComfyUI's /prompt honors a client-supplied prompt_id and echoes it back
    # unchanged (confirmed empirically), which is what lets job_id double as
    # the prompt_id here.
    if not ACCEPTING.is_set():
        await send_json_to(session_id, {
            "type": "error",
            "message": "the server is shutting down and is not taking new generations"})
        return

    prompt_id = str(uuid.uuid4())
    JOBS[prompt_id] = {"session_id": session_id, "kind": kind, "provenance": None}
    job = job_queue.GenerationJob(
        session_id=session_id, kind=kind, submit=submit,
        provenance=provenance, job_id=prompt_id,
    )
    try:
        position = await GENERATION_QUEUE.submit(job)
    except job_queue.QueueFullError as exc:
        JOBS.pop(prompt_id, None)
        await send_json_to(session_id, {"type": "error", "message": str(exc)})
        return

    if position > 1:
        # Only when there is genuinely a line. Announcing "position 1" on an
        # idle booth would invent a wait that isn't there.
        await send_json_to(session_id, {
            "type": "queue_position", "prompt_id": prompt_id, "position": position,
        })


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
    parser.add_argument("--retain-days", type=float, default=None,
                        help="how long batch photos survive (0 = keep indefinitely, "
                             f"default {batch.DEFAULT_RETAIN_DAYS:g})")
    parser.add_argument("--keep-intermediates", action="store_true",
                        help="keep the original uploads and analysed copies for "
                             "debugging; off by default because they are the most "
                             "sensitive thing this app writes down")
    parser.add_argument("--require-consent", action="store_true",
                        help="refuse a batch that declares no consent basis. Off "
                             "by default -- the fields are still there and still "
                             "written to the manifest, they just do not block "
                             "(see consent.py for the reasoning)")
    args = parser.parse_args()

    if args.retain_days is not None:
        batch.DEFAULT_RETAIN_DAYS = args.retain_days
    if args.keep_intermediates:
        batch.KEEP_INTERMEDIATES = True
        print("[batch] --keep-intermediates: original photographs will be retained")
    if args.require_consent:
        consent.REQUIRED = True
    print(f"[consent] declaration is "
          f"{'required' if consent.REQUIRED else 'optional (--require-consent to enforce)'}"
          f"; runs without one are recorded as '{consent.NOT_RECORDED}'")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
