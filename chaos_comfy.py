"""A fake ComfyUI that fails on purpose.

    python chaos_comfy.py --port 8188 --failure-rate 0.3 --seed 42
    python web_server.py --port 8010          # then point a browser at it

Why this exists: the app's failure handling used to be untestable in any
honest way. The unit tests prove the retry ladder is correct in isolation, but
"a guest at the booth still gets a coherent outcome when ComfyUI is having a
bad night" is a claim about the whole system, and there was no way to make
ComfyUI have a bad night on demand.

The design is lifted from a technical challenge whose mock quote service does
exactly this -- deliberate instability with a seeded RNG -- because it is the
right shape: reproducible failures beat real ones, since a bug you can only
reproduce one time in five is a bug you cannot fix.

It speaks enough of ComfyUI's protocol to drive the real app end to end with no
GPU and no models: the REST surface the backend calls, and the websocket event
stream the relay consumes, including binary preview frames.

Three faithfulness decisions worth knowing about:

- **One socket per clientId wins.** Real ComfyUI keeps a single websocket per
  client id and silently drops the older one, which is how two instances of
  this app sharing an id made one of them go permanently deaf. That behaviour
  is reproduced here on purpose, so `tests/test_chaos_comfy.py` can assert the
  app no longer trips over it.
- **Failures are rolled before validation**, like the real thing under load: a
  perfectly valid request can still get a 503.
- **The slow path is a real sleep**, not an immediate error, because a
  dependency that is slow and a dependency that is down need different
  handling and the app has to tell them apart.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import random
import struct
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response
from PIL import Image, ImageDraw, ImageFilter

FAILURE_RATE = float(os.environ.get("COMFY_FAILURE_RATE", "0.20"))
SLOW_RATE = float(os.environ.get("COMFY_SLOW_RATE", "0.10"))
SLOW_SECONDS = float(os.environ.get("COMFY_SLOW_SECONDS", "8"))
RENDER_SECONDS = float(os.environ.get("COMFY_RENDER_SECONDS", "4"))
STEPS = int(os.environ.get("COMFY_STEPS", "12"))
_seed = os.environ.get("COMFY_SEED")
_rng = random.Random(int(_seed)) if _seed else random.Random()

# Which endpoints are allowed to misbehave. /queue and /history stay reliable
# by default: they are how the app *reconciles* an ambiguous submission, and a
# harness where reconciliation is also broken tests despair rather than
# resilience. Override with COMFY_CHAOS_ENDPOINTS to exercise that too.
CHAOS_ENDPOINTS = set(os.environ.get(
    "COMFY_CHAOS_ENDPOINTS", "prompt,upload,view").split(","))

app = FastAPI(title="chaos-comfy (fake ComfyUI)")

STORE = Path(os.environ.get("COMFY_CHAOS_DIR", Path(__file__).parent / ".chaos_comfy"))
(STORE / "input").mkdir(parents=True, exist_ok=True)
(STORE / "output").mkdir(parents=True, exist_ok=True)

HISTORY: dict[str, dict] = {}
QUEUE: list[list] = []
SOCKETS: dict[str, WebSocket] = {}
STATS = {"prompts": 0, "failures": 0, "slow": 0, "uploads": 0, "views": 0}


# --- the instability ----------------------------------------------------------

class Injected(Exception):
    def __init__(self, status: int):
        self.status = status


# A scripted sequence of outcomes, consumed before the dice are rolled: an int
# is a status code to fail with, None is "let this one through". Random failure
# is right for a soak run and wrong for a test -- "survives a 30% failure rate"
# is a claim about a distribution, while "survives two 503s then succeeds" is
# one a test can make fail. Empty by default, so a real run is purely random.
SCRIPT: list[int | None] = []


def maybe_fail(endpoint: str) -> float:
    """Rolls for this call. Returns how long to sleep (0 for a normal call),
    raises Injected for a failure.

    One roll decides both outcomes, so raising the failure rate genuinely
    squeezes out the slow path rather than adding to it -- the same arithmetic
    the real service's knobs have.
    """
    if SCRIPT:
        outcome = SCRIPT.pop(0)
        if outcome is not None:
            STATS["failures"] += 1
            raise Injected(outcome)
        return 0.0
    if endpoint not in CHAOS_ENDPOINTS:
        return 0.0
    roll = _rng.random()
    if roll < FAILURE_RATE:
        STATS["failures"] += 1
        raise Injected(_rng.choice([500, 502, 503]))
    if roll < FAILURE_RATE + SLOW_RATE:
        STATS["slow"] += 1
        return SLOW_SECONDS
    return 0.0


@app.exception_handler(Injected)
async def injected_handler(request, exc: Injected):
    return JSONResponse(
        status_code=exc.status,
        content={"error": "upstream_unavailable",
                 "message": "ComfyUI is temporarily unavailable (injected by chaos_comfy)."})


# --- the REST surface the backend actually calls -------------------------------

@app.get("/system_stats")
def system_stats():
    return {"system": {"comfyui_version": "chaos-0.1"},
            "devices": [{"name": "chaos:0 fake GPU", "type": "cuda",
                         "vram_total": 8 * 1024 ** 3, "vram_free": 6 * 1024 ** 3}]}


@app.get("/object_info")
def object_info():
    """Only the two loaders the app checks models against. Reporting the real
    filenames means start_demo.ps1's preflight passes against the fake."""
    return {
        "CheckpointLoaderSimple": {"input": {"required": {
            "ckpt_name": [["RealVisXL_V5.0_fp16.safetensors"]]}}},
        "ControlNetLoader": {"input": {"required": {
            "control_net_name": [["diffusers_xl_depth_full.safetensors"]]}}},
    }


@app.post("/upload/image")
async def upload_image(image: UploadFile = File(...), type: str = Form("input"),
                       overwrite: str = Form("true")):
    delay = maybe_fail("upload")
    if delay:
        await asyncio.sleep(delay)
    raw = await image.read()
    (STORE / "input" / image.filename).write_bytes(raw)
    STATS["uploads"] += 1
    return {"name": image.filename, "subfolder": "", "type": "input"}


@app.post("/prompt")
async def prompt(payload: dict):
    delay = maybe_fail("prompt")
    if delay:
        await asyncio.sleep(delay)

    prompt_id = payload.get("prompt_id") or str(uuid.uuid4())
    client_id = payload.get("client_id")
    workflow = payload.get("prompt") or {}
    STATS["prompts"] += 1

    QUEUE.append([len(QUEUE), prompt_id, workflow, {}, []])
    asyncio.create_task(_execute(prompt_id, client_id, workflow))
    return {"prompt_id": prompt_id, "number": len(QUEUE), "node_errors": {}}


@app.get("/history/{prompt_id}")
def history(prompt_id: str):
    delay = maybe_fail("history")
    if delay:
        time.sleep(delay)
    entry = HISTORY.get(prompt_id)
    return {prompt_id: entry} if entry else {}


@app.get("/queue")
def queue():
    return {"queue_running": QUEUE[:1], "queue_pending": QUEUE[1:]}


@app.get("/view")
def view(filename: str, subfolder: str = "", type: str = "output"):
    delay = maybe_fail("view")
    if delay:
        time.sleep(delay)
    path = STORE / type / filename
    if not path.exists():
        return JSONResponse(status_code=404, content={"error": "not found"})
    STATS["views"] += 1
    return Response(content=path.read_bytes(), media_type="image/png")


@app.get("/chaos/stats")
def chaos_stats():
    """Not part of ComfyUI's API -- it is how a test asserts that the app
    really did retry, rather than inferring it from a log line."""
    return {**STATS, "failure_rate": FAILURE_RATE, "slow_rate": SLOW_RATE,
            "seed": _seed, "chaos_endpoints": sorted(CHAOS_ENDPOINTS),
            "sockets": list(SOCKETS)}


# --- the websocket the relay consumes ------------------------------------------

@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    client_id = websocket.query_params.get("clientId") or str(uuid.uuid4())
    # Faithful to the real server, and the reason this file exists as much as
    # the chaos does: a second connection with the same id silently replaces
    # the first, which then never receives another event for the rest of its
    # life. See tests/test_chaos_comfy.py.
    SOCKETS[client_id] = websocket
    try:
        await websocket.send_json({"type": "status", "data": {
            "status": {"exec_info": {"queue_remaining": len(QUEUE)}}}})
        while True:
            await websocket.receive_text()
    except (WebSocketDisconnect, RuntimeError):
        if SOCKETS.get(client_id) is websocket:
            SOCKETS.pop(client_id, None)


async def _send(client_id: str | None, message: dict):
    socket = SOCKETS.get(client_id) if client_id else None
    targets = [socket] if socket else list(SOCKETS.values())
    for target in targets:
        try:
            await target.send_json(message)
        except Exception:            # noqa: BLE001 -- a dead socket is not our problem
            pass


async def _send_preview(client_id: str | None, image: Image.Image):
    """A binary preview frame in ComfyUI's own undocumented framing:
    4-byte event type, 4-byte image format, then the JPEG."""
    buf = io.BytesIO()
    image.convert("RGB").resize((256, 256)).save(buf, format="JPEG", quality=60)
    payload = struct.pack(">I", 1) + struct.pack(">I", 1) + buf.getvalue()
    socket = SOCKETS.get(client_id) if client_id else None
    for target in ([socket] if socket else list(SOCKETS.values())):
        try:
            await target.send_bytes(payload)
        except Exception:            # noqa: BLE001
            pass


# --- the "render" ---------------------------------------------------------------

def _subject_size(workflow: dict) -> tuple[int, int]:
    """Matches the output to whatever image the workflow loads, the way a real
    graph would -- the app composites the untouched subject over this, and a
    size mismatch is a different bug than the one being tested."""
    for node in workflow.values():
        if isinstance(node, dict) and node.get("class_type", "").startswith("LoadImage"):
            name = (node.get("inputs") or {}).get("image")
            path = STORE / "input" / str(name)
            if path.exists():
                try:
                    with Image.open(path) as im:
                        return im.size
                except Exception:    # noqa: BLE001
                    pass
    return (1024, 1024)


def _fake_background(size: tuple[int, int], seed: int) -> Image.Image:
    """Something that reads as a generated scene at a glance without pretending
    to be one: a graded sky, a horizon, and a few soft shapes."""
    rng = random.Random(seed)
    w, h = size
    image = Image.new("RGB", size)
    draw = ImageDraw.Draw(image)
    top = (rng.randint(90, 160), rng.randint(120, 190), rng.randint(150, 220))
    bottom = (rng.randint(180, 240), rng.randint(170, 230), rng.randint(160, 210))
    for y in range(h):
        t = y / max(1, h - 1)
        draw.line([(0, y), (w, y)],
                  fill=tuple(int(a + (b - a) * t) for a, b in zip(top, bottom)))
    for _ in range(rng.randint(4, 9)):
        x = rng.randint(0, w)
        bw, bh = rng.randint(w // 12, w // 5), rng.randint(h // 6, h // 2)
        shade = rng.randint(60, 140)
        draw.rectangle([x, h - bh, x + bw, h], fill=(shade, shade, shade + 15))
    return image.filter(ImageFilter.GaussianBlur(radius=max(1, w // 300)))


async def _execute(prompt_id: str, client_id: str | None, workflow: dict):
    """Emits the same event sequence the real server does, at a plausible pace."""
    nodes = [k for k in workflow] or ["1", "2", "3"]
    size = _subject_size(workflow)
    image = _fake_background(size, seed=hash(prompt_id) & 0xFFFF)
    per_step = RENDER_SECONDS / max(1, STEPS)

    await _send(client_id, {"type": "execution_start", "data": {"prompt_id": prompt_id}})
    for node in nodes[:4]:
        await _send(client_id, {"type": "executing",
                                "data": {"node": node, "prompt_id": prompt_id}})
        await asyncio.sleep(per_step)

    for step in range(1, STEPS + 1):
        await _send(client_id, {"type": "progress",
                                "data": {"value": step, "max": STEPS, "prompt_id": prompt_id}})
        if step % 4 == 0:
            await _send_preview(client_id, image)
        await asyncio.sleep(per_step)

    filename = f"chaos_{prompt_id[:8]}.png"
    image.save(STORE / "output" / filename)
    HISTORY[prompt_id] = {
        "prompt": [0, prompt_id, workflow, {}, []],
        "outputs": {"9": {"images": [{"filename": filename, "subfolder": "",
                                      "type": "output"}]}},
        "status": {"status_str": "success", "completed": True, "messages": []},
    }
    QUEUE[:] = [q for q in QUEUE if q[1] != prompt_id]

    # node=None is how ComfyUI says "this prompt is done"; the app's relay keys
    # its whole completion path off it.
    await _send(client_id, {"type": "executing", "data": {"node": None, "prompt_id": prompt_id}})
    await _send(client_id, {"type": "status", "data": {
        "status": {"exec_info": {"queue_remaining": len(QUEUE)}}}})


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8188)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--failure-rate", type=float, help="fraction of calls that fail")
    parser.add_argument("--slow-rate", type=float, help="fraction that stall")
    parser.add_argument("--slow-seconds", type=float)
    parser.add_argument("--render-seconds", type=float, help="how long a fake generation takes")
    parser.add_argument("--seed", type=int, help="fixes the failures so a run is reproducible")
    args = parser.parse_args()

    global FAILURE_RATE, SLOW_RATE, SLOW_SECONDS, RENDER_SECONDS, _rng, _seed
    if args.failure_rate is not None:
        FAILURE_RATE = args.failure_rate
    if args.slow_rate is not None:
        SLOW_RATE = args.slow_rate
    if args.slow_seconds is not None:
        SLOW_SECONDS = args.slow_seconds
    if args.render_seconds is not None:
        RENDER_SECONDS = args.render_seconds
    if args.seed is not None:
        _seed, _rng = str(args.seed), random.Random(args.seed)

    print(f"[chaos] fake ComfyUI on {args.host}:{args.port} -- "
          f"failure_rate={FAILURE_RATE} slow_rate={SLOW_RATE} "
          f"slow_seconds={SLOW_SECONDS} seed={_seed} endpoints={sorted(CHAOS_ENDPOINTS)}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
