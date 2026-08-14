# Resolume live VJ bridge

*This is the build the repo started as, kept because it still runs and because
the ComfyUI plumbing underneath it is shared with the photo booth. The photo
booth is the primary demo — see the [main README](../README.md).*

A Resolume-to-generative-AI bridge in both directions.

- **Trigger → generate**: a Resolume clip trigger (or a manual/test call) fires
  a still-image or video generation and streams the result back live as a Spout
  source.
- **Resolume → prompt**: pull whatever's currently live in Resolume (active clip
  names, active effects, and the clip's actual thumbnail image) and turn it into
  a text prompt, so you can regenerate/reinterpret what the VJ actually built
  instead of typing a prompt by hand.

Generation is backend-agnostic — swap between local Stable Diffusion (ComfyUI),
Runway, or Kling with one flag. Video generation (ComfyUI/LTXV) is also
available alongside stills.

```
                    +-------------------------------+
Resolume clip  ---> |  bridge.py                     |
  trigger (OSC)     |   - clip -> prompts.json       |
                     |   - OR resync -> Resolume      |---> backend.generate_image(prompt)
Resolume state <---- |     REST API -> name+effects   |         or
  (REST API +        |     + thumbnail pixel stats    |     backend.generate_video(prompt)
   thumbnail)        |     -> prompt                  |         |
                     +-------------------------------+          v
                                    |                     ComfyUI / Runway / Kling
                                    v                            |
                          Spout sender "ComfyBridge" <-----------+
                                    |                    (video loops frame-by-frame)
                                    v
                    Resolume Sources > Spout > ComfyBridge (live layer)
                       or  python spout_viewer.py  (no Resolume needed)
```

## Prerequisites

```
powershell -ExecutionPolicy Bypass -File install.ps1 -Resolume
python doctor.py --resolume
```

This is the one path that needs the extra packages (`python-osc`, `SpoutGL`,
`PyOpenGL`) and the GUI OpenCV build for `spout_viewer.py`'s window — the
installer handles both, and `doctor.py --resolume` reports which OpenCV build
you actually ended up with.

- Windows for the **Spout output specifically** — the OSC trigger path and
  generation work fine on macOS/Linux, and the requirement markers skip the
  Spout packages there rather than failing the install. `spout_sender_loop()`
  reports the missing module and returns instead of dying in a background
  thread, so the app stays honest about publishing nothing.
- A generation backend, at least one of:
  - [ComfyUI](https://github.com/comfyanonymous/ComfyUI) running locally with a
    checkpoint installed (default `127.0.0.1:8188`) — needed for video
    generation specifically, Runway/Kling are stills-only here
  - Runway API access (`RUNWAYML_API_SECRET` env var) — https://docs.dev.runwayml.com
  - Kling AI API access (`KLING_ACCESS_KEY` / `KLING_SECRET_KEY` env vars) — https://kling.ai/document-api
- Resolume Arena or Avenue (optional — everything works against
  `send_trigger.py` and `spout_viewer.py` without it)
- Python 3.10+ (SpoutGL ships prebuilt wheels for common CPython versions)

## Setup

If you're using the ComfyUI backend:

- Stills: edit `workflows/txt2img_api.json`, set `"ckpt_name"` to a checkpoint
  you have installed. (Which node that is no longer needs to be memorised —
  `workflow_graph.py` resolves node roles from the graph's own wiring; see the
  main README.)
- Video: uses `workflows/text_to_video_api.json` (LTXV 2B). Needs
  `ltx-video-2b-v0.9.1.safetensors` in `models/checkpoints/` and
  `t5xxl_fp8_e4m3fn.safetensors` in `models/text_encoders/` —
  [LTXV checkpoint](https://huggingface.co/Lightricks/LTX-Video/blob/main/ltx-video-2b-v0.9.1.safetensors),
  [T5 text encoder](https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/t5xxl_fp8_e4m3fn.safetensors).
  Runs comfortably on 8GB VRAM; ~1-4 seconds of 768x512 video per generation,
  roughly a minute to render.

Edit `prompts.json` to map Resolume clip slots (`L<layer>C<clip>`) to prompts
for the clip-trigger path. `_default` covers any untracked clip. The resync path
doesn't need this file — it builds its prompt from Resolume's live state
instead.

## Run

```
python bridge.py --backend comfy              # local Stable Diffusion / LTXV via ComfyUI
python bridge.py --backend runway
python bridge.py --backend kling
```

This starts an OSC listener (`:9000` by default) and a Spout sender named
`ComfyBridge`.

**See it without Resolume:**

```
python spout_viewer.py
```

opens a window showing whatever the bridge is currently sending. Or use the
typing-box GUI instead of raw OSC calls:

```
python gui.py
```

**In Resolume** (once you're ready):

1. **Preferences > OSC** — enable OSC output, host `127.0.0.1`, port `9000`
   (must match `bridge.py`'s `--osc-port`). Resolume broadcasts OSC for every
   clip connect/disconnect automatically once this is on.
2. **Preferences > Webserver** — enable it (default port `8080`) if you want the
   resync path to read live composition state.
3. Add a layer, **Sources > Spout > ComfyBridge**.
4. Trigger any clip to fire a generation from `prompts.json`.

## Testing without Resolume open

```
python send_trigger.py --clip 1 1        # simulate a clip trigger
python send_trigger.py --prompt "a cat made of stained glass"
python send_trigger.py --video "a cat made of stained glass, slow pan"  # ComfyUI backend only
python send_trigger.py --resync           # pull live state from Resolume's REST API
```

`--resync` needs Resolume actually running (with the webserver enabled), since
it reads real composition state; the other three work with just `bridge.py` and
a backend running.

## How resync builds a prompt

`resolume_state.py` calls Resolume's REST API (`GET /api/v1/composition`), walks
the visible layers, and for each one's active (connected) clip pulls from two
sources:

- **Name + effects**: the clip's name as a prompt fragment, plus a short
  descriptor for each non-bypassed effect on that clip or its layer
  (`kaleidoscope` → "kaleidoscopic, symmetrical fractal patterns", etc. — see
  `EFFECT_DESCRIPTORS`). Only as good as how the VJ named things.
- **The clip's actual thumbnail** (`GET .../clips/{n}/thumbnail`, forcing a
  fresh capture first via `POST .../thumbnail/update`): resized to 48x48 and
  read directly for dominant hue, saturation, brightness, contrast, and edge
  density, converted to adjectives (`describe_image()` in `resolume_state.py`).
  This is what makes the prompt reproducible — the same visual content always
  produces the same descriptors, regardless of what the clip happens to be
  named.

Fragments are joined, de-duplicated, and a fixed style suffix is appended.

## How video generation works

`--backend comfy` gets a `generate_video()` in addition to `generate_image()`.
It queues `workflows/text_to_video_api.json` (LTXV text-to-video: `CLIPLoader` →
`CLIPTextEncode` → `LTXVConditioning` → `EmptyLTXVLatentVideo` → `LTXVScheduler`
→ `SamplerCustom` → `VAEDecode` → `CreateVideo` → `SaveVideo`), polls `/history`
the same way stills do (ComfyUI's `SaveVideo` reports its output under the same
`"images"` key as `SaveImage`, just flagged `"animated": true`), downloads the
resulting mp4, and hands it to `VideoLoopPlayer` — a background thread that
decodes frames with OpenCV and feeds them into the same `FrameBuffer` the Spout
sender reads from, looping until the next generation replaces it. Triggered via
OSC `/comfybridge/generate_video` (arg0: prompt text).

## Known limitations (this is a demo, not production)

- Polls REST endpoints on a timer instead of using websocket/streaming APIs —
  simpler, slightly higher latency. (The photo booth does use ComfyUI's
  websocket — see the main README.)
- One generation in flight at a time; triggers that arrive while busy are
  dropped rather than queued. The photo booth's `job_queue.py` is the answer to
  this and the bridge does not use it yet.
- Fixed output canvas (`SPOUT_WIDTH`/`SPOUT_HEIGHT` in `bridge.py`, default
  512x512) — anything a backend returns (image or video frame) gets fit to it.
  `SpoutFrameBuffer.set_image()` (shared by both apps via `spout_output.py`)
  cover-fits and crops to the sender's aspect ratio rather than stretching, so
  non-square sources no longer get squashed — they get cropped to fill instead,
  same as any normal video source.
- `EFFECT_DESCRIPTORS` in `resolume_state.py` is a small hand-picked table, not
  a mapping of Resolume's full effect library. The thumbnail descriptors are
  deterministic pixel stats, not learned image understanding — coarse but
  consistent.
- Video generation is ComfyUI/LTXV-only; Runway and Kling backends here only
  implement stills, though their APIs support video too.
- No retry/backoff on backend errors, no auth on the bridge's own OSC listener,
  single-machine only.

## Where this half goes next

- Runway/Kling `generate_video()` implementations (their APIs already support
  image-to-video/text-to-video).
- Feed effect *parameter values* (not just names) into the resync prompt — e.g.
  a Colorize hue value as an actual color descriptor.
- Put triggers through `job_queue.py` so a burst during a set queues instead of
  being dropped.
