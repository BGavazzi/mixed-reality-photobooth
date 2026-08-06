"""
Browser front-end for the ComfyUI pipeline: mixed-reality photo booth.

Upload a real photo -> rotoscope / pose / depth / illumination extracted
-> generate a new background around the subject (ComfyUI + SDXL ControlNet
depth-conditioned inpaint) -> composite the untouched subject back on top.
The whole generation step streams live over a websocket relay of ComfyUI's
own progress + denoising-preview events, so the browser shows the image
actually forming instead of a spinner.

    python web_server.py
    -> http://127.0.0.1:8000
"""

import argparse
import asyncio
import base64
import io
import json
import struct
import uuid
from contextlib import asynccontextmanager

import websockets
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, ImageOps

from backends.comfy import ComfyBackend
import photoshoot_pipeline

COMFY_ADDRESS = "127.0.0.1:8188"
COMFY_CLIENT_ID = "web-photoshoot-bridge"
BINARY_PREVIEW_EVENT = 1  # ComfyUI's BinaryEventTypes.PREVIEW_IMAGE


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(comfy_relay_loop())
    yield


app = FastAPI(lifespan=lifespan)
backend = ComfyBackend(COMFY_ADDRESS)

# Single-generation-at-a-time, single-demo-user design (matches bridge.py's
# own busy-lock philosophy) — one active browser socket, one active job.
current_ws: WebSocket | None = None
active_prompt_id: str | None = None
active_kind: str | None = None  # "image" | "background" | "region" — tells the client how to apply the result


# --- helpers ----------------------------------------------------------------

def pil_to_b64(image: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def b64_to_pil(data: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(data)))


async def send_json(payload: dict):
    if current_ws is not None:
        try:
            await current_ws.send_json(payload)
        except Exception as exc:
            print(f"[ws] send failed: {exc}")


async def send_bytes(data: bytes):
    if current_ws is not None:
        try:
            await current_ws.send_bytes(data)
        except Exception as exc:
            print(f"[ws] send failed: {exc}")


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
    global active_prompt_id, active_kind

    if isinstance(message, (bytes, bytearray)):
        if len(message) < 8:
            return
        event_type = struct.unpack(">I", message[:4])[0]
        if event_type == BINARY_PREVIEW_EVENT:
            await send_bytes(message[8:])  # strip 4-byte event + 4-byte format header
        return

    try:
        event = json.loads(message)
    except json.JSONDecodeError:
        return

    etype = event.get("type")
    data = event.get("data", {})
    prompt_id = data.get("prompt_id")

    if etype == "status":
        queue_remaining = data.get("status", {}).get("exec_info", {}).get("queue_remaining")
        await send_json({"type": "status", "queue_remaining": queue_remaining})
        return

    if active_prompt_id is None or prompt_id != active_prompt_id:
        return

    if etype == "progress":
        await send_json({"type": "progress", "value": data.get("value"), "max": data.get("max")})

    elif etype == "executing":
        node = data.get("node")
        if node is None:
            # None node with our active prompt_id means this run just finished.
            image = await asyncio.to_thread(backend.get_result_image, prompt_id)
            if image is not None:
                await send_json({"type": "done", "image_base64": pil_to_b64(image), "kind": active_kind})
            active_prompt_id = None
            active_kind = None
        else:
            await send_json({"type": "executing", "node": node})

    elif etype == "execution_error":
        await send_json({"type": "error", "message": str(data.get("exception_message", "generation failed"))})
        active_prompt_id = None
        active_kind = None


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
        "original": pil_to_b64(image),
        "cutout": pil_to_b64(result["cutout"]),
        "mask": pil_to_b64(result["mask"]),
        "pose": pil_to_b64(result["pose"]),
        "depth": pil_to_b64(result["depth"]),
        "illumination": result["illumination"].to_dict(),
    })


# --- websocket: browser control + live event relay ---------------------------

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    global current_ws
    await websocket.accept()
    current_ws = websocket
    try:
        while True:
            msg = await websocket.receive_json()
            action = msg.get("action")

            if action == "generate_image":
                asyncio.create_task(handle_generate_image(msg))
            elif action == "generate_background":
                asyncio.create_task(handle_generate_background(msg))
            elif action == "edit_region":
                asyncio.create_task(handle_edit_region(msg))

    except WebSocketDisconnect:
        if current_ws is websocket:
            current_ws = None


async def handle_generate_image(msg: dict):
    global active_prompt_id, active_kind
    prompt = (msg.get("prompt") or "").strip()
    if not prompt:
        return
    prompt_id = await asyncio.to_thread(backend.queue_image_generation, prompt, COMFY_CLIENT_ID)
    active_prompt_id = prompt_id
    active_kind = "image"
    await send_json({"type": "queued", "prompt_id": prompt_id})


async def handle_generate_background(msg: dict):
    global active_prompt_id, active_kind
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
        await send_json({"type": "error", "message": f"bad request: {exc}"})
        return

    prompt_id = await asyncio.to_thread(
        backend.queue_background_generation,
        subject, background_mask, depth, prompt,
        controlnet_strength, denoise,
        client_id=COMFY_CLIENT_ID,
    )
    active_prompt_id = prompt_id
    active_kind = "background"
    await send_json({"type": "queued", "prompt_id": prompt_id})


async def handle_edit_region(msg: dict):
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
    global active_prompt_id, active_kind
    try:
        composite = b64_to_pil(msg["subject"]).convert("RGB")
        region_mask = b64_to_pil(msg["mask"]).convert("L")
        depth = b64_to_pil(msg["depth"]).convert("RGB")
        label = (msg.get("prompt") or "").strip()
        prompt = f"{label}, seamlessly blended, matching lighting and perspective"
        controlnet_strength = float(msg.get("controlnet_strength", 0.1))
        denoise = float(msg.get("denoise", 1.0))
    except Exception as exc:
        await send_json({"type": "error", "message": f"bad request: {exc}"})
        return

    prompt_id = await asyncio.to_thread(
        backend.queue_background_generation,
        composite, region_mask, depth, prompt,
        controlnet_strength, denoise,
        client_id=COMFY_CLIENT_ID,
    )
    active_prompt_id = prompt_id
    active_kind = "region"
    await send_json({"type": "queued", "prompt_id": prompt_id})


def main():
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
