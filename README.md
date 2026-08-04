# Resolume <-> ComfyUI live bridge

Quick-and-dirty local demo: triggering a clip in Resolume fires a ComfyUI
generation, and the result streams back into Resolume as a live Spout
source.

```
Resolume (OSC output, on clip trigger)
        |
        v
   bridge.py  --OSC listener-->  picks prompt for that clip
        |
        v
   ComfyUI REST API (/prompt, /history, /view)
        |
        v
   generated frame --> Spout sender "ComfyBridge"
        |
        v
Resolume Sources > Spout > ComfyBridge  (live layer)
```

## Prerequisites

- Windows (Spout is Windows-only)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) running locally with
  at least one checkpoint installed (default `python main.py`, listens on
  `127.0.0.1:8188`)
- Resolume Arena or Avenue
- Python 3.10+ (SpoutGL ships prebuilt wheels for common CPython versions;
  if `pip install` fails on SpoutGL, check that your Python version has a
  wheel available)

## Setup

```
pip install -r requirements.txt
```

Edit `workflows/txt2img_api.json`:
- set `"ckpt_name"` (node `4`) to a checkpoint you actually have installed
  in ComfyUI (check ComfyUI's model dropdown for the exact filename)

Edit `prompts.json` to map Resolume clip slots (`L<layer>C<clip>`) to the
prompts you want each clip to generate. `_default` is used for any
untracked clip.

## Run

```
python bridge.py
```

You should see it start an OSC listener and a Spout sender named
`ComfyBridge`.

In Resolume:
1. **Preferences > OSC** — enable OSC output, set host to `127.0.0.1`
   (or the bridge machine's IP) and port `9000` (must match `bridge.py`'s
   `OSC_LISTEN_PORT`). Resolume broadcasts OSC feedback for every clip
   connect/disconnect automatically once this is on — no per-clip mapping
   needed.
2. Add a layer, **Sources > Spout > ComfyBridge**.
3. Trigger any clip. The bridge looks up that clip's prompt, generates it
   in ComfyUI, and the live layer updates with the result once it's ready
   (generation takes a few seconds depending on your GPU/model/step count —
   this is not truly real-time, it's request/response).

## Testing without Resolume open

```
python test_trigger.py --clip 1 1
python test_trigger.py --prompt "a cat made of stained glass"
```

Watch the bridge's console output and the Spout receiver (e.g. Resolume,
or any Spout-capable viewer) to confirm frames arrive.

## Known limitations (this is a demo, not production)

- Polls ComfyUI's `/history` REST endpoint on a timer instead of using the
  websocket API — simpler, slightly higher latency.
- One generation in flight at a time; triggers that arrive while busy are
  dropped rather than queued.
- Fixed output canvas (`SPOUT_WIDTH`/`SPOUT_HEIGHT` in `bridge.py`, default
  512x512) — anything ComfyUI returns gets resized to fit. Keep the
  workflow's `EmptyLatentImage` width/height matching those constants to
  avoid quality loss from resizing.
- No retry/backoff on ComfyUI errors, no auth, single-machine only.

## Where this goes next

- Swap `/history` polling for the websocket API for lower-latency "done"
  detection.
- Use a fast/distilled pipeline (SDXL Turbo, LCM) to get generation time
  down toward something that feels closer to live.
- Queue triggers instead of dropping them while busy.
- Support img2img so a Resolume layer's current frame can feed back in as
  ComfyUI's init image (real generative feedback loop).
