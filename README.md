# resolume-genai-bridge

Two demos in one repo, both built around orchestrating ComfyUI's real APIs
(REST + its own progress/preview websocket) rather than just calling a
`/generate` endpoint and waiting:

1. **Mixed-Reality Photo Booth** (primary) — a browser app that takes a
   real photo of a person, extracts everything needed to regenerate the
   environment around them (rotoscope, pose, depth, lighting), generates a
   new depth-conditioned background or adds objects into it, and
   composites the untouched original subject back on top. Live denoising
   preview streams into the browser as it generates.
2. **Resolume live VJ bridge** (original build, still included) — a
   Resolume-to-generative-AI bridge that turns clip triggers or Resolume's
   own live composition state into image/video generations, streamed back
   as a Spout source.

---

## 1. Mixed-Reality Photo Booth

```
Browser (web/index.html)
  |  upload photo
  v
POST /api/analyze  -----------------------------------------------+
  |                                                                |
  |  photoshoot_pipeline.py:                                      |
  |    segment_subject()  -> rembg/birefnet-portrait -> cutout+mask
  |    estimate_pose()    -> OpenPose/DWPose (controlnet_aux)     |
  |    estimate_depth()   -> MiDaS (controlnet_aux)               |
  |    estimate_illumination() -> plain CV on the subject pixels  |
  |         (light direction / warmth / softness -> text descriptor)
  v                                                                |
Browser: layer stack (background / subject / pose / depth, <------+
          each independently visible/opaque/movable/scalable)
  |
  |  scene prompt + "Generate Background"
  v
WS /ws  ==(relays ComfyUI's own websocket)==>  ComfyUI
  |         - progress (step N/M)                  |
  |         - executing (which node)                RealVisXL V5.0 (SDXL)
  |         - binary preview frames (live denoise)   + ControlNet depth
  |         - done -> final image                    inpaint, masked to
  v                                                   background-only
Browser: new background layer, or (region-draw tool)
          a masked object inserted as its own layer
  |
  v
Flatten & export PNG, or export a project .json (all layers +
generation params) to resume/tweak later
```

The core idea: the subject's actual pixels are never re-generated. Only
the region *around* them is, conditioned on their real pose/depth so the
new environment stays geometrically consistent, then composited back
with the original cutout on top.

### Why a websocket relay instead of just polling

ComfyUI already exposes a websocket with real-time `progress`,
`executing`, and (with `--preview-method auto`) binary JPEG preview
frames of the image mid-denoise. `web_server.py` keeps one persistent
websocket connection to ComfyUI and relays those events straight to
whichever browser tab is active — the browser shows the image actually
forming, not a spinner. The binary preview frame format is
undocumented-but-stable: `struct.pack(">I", event_type) + struct.pack(">I", image_type) + image_bytes`;
verified against ComfyUI's own `server.py` (`encode_bytes`/`send_image`).

### Setup

```
pip install -r requirements.txt
```

Models (ComfyUI's `models/` folder):
- Checkpoint: [`RealVisXL_V5.0_fp16.safetensors`](https://huggingface.co/SG161222/RealVisXL_V5.0) in `models/checkpoints/`
- ControlNet: [`diffusers_xl_depth_full.safetensors`](https://huggingface.co/lllyasviel/sd_control_collection) in `models/controlnet/`
- Rotoscope/pose/depth models (`birefnet-portrait`, OpenPose, MiDaS) download automatically on first use via `rembg`/`controlnet_aux`.

### Run

```
powershell -ExecutionPolicy Bypass -File start_demo.ps1
```

Starts ComfyUI (with `--preview-method auto`), waits for it to be ready,
starts `web_server.py`, waits for that, opens the browser. Run
`stop_demo.ps1` to shut both down. (If you double-click the script rather
than running it from an already-open PowerShell window and the services
don't seem to stay up, open PowerShell yourself and run it directly —
that path is the one that's fully verified end-to-end.)

Or manually:
```
python web_server.py     # http://127.0.0.1:8000, needs ComfyUI already running
```

### The region-draw "add object" tool

Draws a box on the canvas, asks what belongs there, and inpaints just
that masked region — added as its own layer, everything else untouched.
Two things that matter for this to actually work, found empirically:

- **ControlNet depth strength defaults near-zero for object insertion**
  (vs. 0.75 for background regen). The depth map reflects the scene
  *before* the new object exists, so conditioning on it fights the model
  trying to introduce geometry that isn't there — at background-regen
  strength the object just didn't appear at all.
- **Minimum region size matters.** SDXL can't render a recognizable
  object into a small masked patch — a tiny corner box just blends into
  the surrounding background. ~30% of the canvas per side is the point
  where this reliably stopped failing in testing; the UI enforces that
  as a minimum.

### Known limitations

- Rotoscope (`birefnet-portrait`) runs CPU-only, ~20s/photo — this
  machine's `onnxruntime-gpu` wants CUDA 13 libraries that aren't
  published as pip wheels yet (checked: `nvidia-cublas-cu13` on PyPI is
  a version-`0.0.1` placeholder).
- Single generation in flight at a time, single browser session assumed
  (matches the Resolume bridge's own busy-lock philosophy below).
- The illumination estimate is classic CV (per-quadrant luminance,
  highlight color, contrast), not a learned model — a useful heuristic
  for prompt-grounding, not a physically accurate light probe.

---

## 2. Resolume live VJ bridge

The original build this repo started as: a Resolume-to-generative-AI
bridge in both directions.

- **Trigger -> generate**: a Resolume clip trigger (or a manual/test
  call) fires a still-image or video generation and streams the result
  back live as a Spout source.
- **Resolume -> prompt**: pull whatever's currently live in Resolume
  (active clip names, active effects, and the clip's actual thumbnail
  image) and turn it into a text prompt, so you can regenerate/reinterpret
  what the VJ actually built instead of typing a prompt by hand.

Generation is backend-agnostic — swap between local Stable Diffusion
(ComfyUI), Runway, or Kling with one flag. Video generation (ComfyUI/LTXV)
is also available alongside stills.

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

### Prerequisites

- Windows (Spout is Windows-only)
- A generation backend, at least one of:
  - [ComfyUI](https://github.com/comfyanonymous/ComfyUI) running locally
    with a checkpoint installed (default `127.0.0.1:8188`) — needed for
    video generation specifically, Runway/Kling are stills-only here
  - Runway API access (`RUNWAYML_API_SECRET` env var) — https://docs.dev.runwayml.com
  - Kling AI API access (`KLING_ACCESS_KEY` / `KLING_SECRET_KEY` env vars) — https://kling.ai/document-api
- Resolume Arena or Avenue (optional — everything works against
  `test_trigger.py` and `spout_viewer.py` without it)
- Python 3.10+ (SpoutGL ships prebuilt wheels for common CPython versions)

### Setup

If you're using the ComfyUI backend:
- Stills: edit `workflows/txt2img_api.json`, set `"ckpt_name"` (node `4`)
  to a checkpoint you have installed.
- Video: uses `workflows/text_to_video_api.json` (LTXV 2B). Needs
  `ltx-video-2b-v0.9.1.safetensors` in `models/checkpoints/` and
  `t5xxl_fp8_e4m3fn.safetensors` in `models/text_encoders/` —
  [LTXV checkpoint](https://huggingface.co/Lightricks/LTX-Video/blob/main/ltx-video-2b-v0.9.1.safetensors),
  [T5 text encoder](https://huggingface.co/comfyanonymous/flux_text_encoders/blob/main/t5xxl_fp8_e4m3fn.safetensors).
  Runs comfortably on 8GB VRAM; ~1-4 seconds of 768x512 video per
  generation, roughly a minute to render.

Edit `prompts.json` to map Resolume clip slots (`L<layer>C<clip>`) to
prompts for the clip-trigger path. `_default` covers any untracked clip.
The resync path doesn't need this file — it builds its prompt from
Resolume's live state instead.

### Run

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

opens a window showing whatever the bridge is currently sending. Or use
the typing-box GUI instead of raw OSC calls:

```
python gui.py
```

**In Resolume** (once you're ready):
1. **Preferences > OSC** — enable OSC output, host `127.0.0.1`, port
   `9000` (must match `bridge.py`'s `--osc-port`). Resolume broadcasts
   OSC for every clip connect/disconnect automatically once this is on.
2. **Preferences > Webserver** — enable it (default port `8080`) if you
   want the resync path to read live composition state.
3. Add a layer, **Sources > Spout > ComfyBridge**.
4. Trigger any clip to fire a generation from `prompts.json`.

### Testing without Resolume open

```
python test_trigger.py --clip 1 1        # simulate a clip trigger
python test_trigger.py --prompt "a cat made of stained glass"
python test_trigger.py --video "a cat made of stained glass, slow pan"  # ComfyUI backend only
python test_trigger.py --resync           # pull live state from Resolume's REST API
```

`--resync` needs Resolume actually running (with the webserver enabled),
since it reads real composition state; the other three work with just
`bridge.py` and a backend running.

### How resync builds a prompt

`resolume_state.py` calls Resolume's REST API (`GET /api/v1/composition`),
walks the visible layers, and for each one's active (connected) clip pulls
from two sources:

- **Name + effects**: the clip's name as a prompt fragment, plus a short
  descriptor for each non-bypassed effect on that clip or its layer
  (`kaleidoscope` → "kaleidoscopic, symmetrical fractal patterns", etc. —
  see `EFFECT_DESCRIPTORS`). Only as good as how the VJ named things.
- **The clip's actual thumbnail** (`GET .../clips/{n}/thumbnail`, forcing
  a fresh capture first via `POST .../thumbnail/update`): resized to
  48x48 and read directly for dominant hue, saturation, brightness,
  contrast, and edge density, converted to adjectives (`describe_image()`
  in `resolume_state.py`). This is what makes the prompt reproducible —
  the same visual content always produces the same descriptors,
  regardless of what the clip happens to be named.

Fragments are joined, de-duplicated, and a fixed style suffix is
appended.

### How video generation works

`--backend comfy` gets a `generate_video()` in addition to
`generate_image()`. It queues `workflows/text_to_video_api.json` (LTXV
text-to-video: `CLIPLoader` → `CLIPTextEncode` → `LTXVConditioning` →
`EmptyLTXVLatentVideo` → `LTXVScheduler` → `SamplerCustom` → `VAEDecode` →
`CreateVideo` → `SaveVideo`), polls `/history` the same way stills do
(ComfyUI's `SaveVideo` reports its output under the same `"images"` key
as `SaveImage`, just flagged `"animated": true`), downloads the resulting
mp4, and hands it to `VideoLoopPlayer` — a background thread that decodes
frames with OpenCV and feeds them into the same `FrameBuffer` the Spout
sender reads from, looping until the next generation replaces it.
Triggered via OSC `/comfybridge/generate_video` (arg0: prompt text).

### Known limitations (this is a demo, not production)

- Polls REST endpoints on a timer instead of using websocket/streaming
  APIs — simpler, slightly higher latency. (The photo booth app above
  does use ComfyUI's websocket — see part 1.)
- One generation in flight at a time; triggers that arrive while busy are
  dropped rather than queued.
- Fixed output canvas (`SPOUT_WIDTH`/`SPOUT_HEIGHT` in `bridge.py`,
  default 512x512) — anything a backend returns (image or video frame)
  gets resized to fit, so non-square sources get squashed.
- `EFFECT_DESCRIPTORS` in `resolume_state.py` is a small hand-picked
  table, not a mapping of Resolume's full effect library. The thumbnail
  descriptors are deterministic pixel stats, not learned image
  understanding — coarse but consistent.
- Video generation is ComfyUI/LTXV-only; Runway and Kling backends here
  only implement stills, though their APIs support video too.
- No retry/backoff on backend errors, no auth on the bridge's own OSC
  listener, single-machine only.

## Where this goes next

- Runway/Kling `generate_video()` implementations (their APIs already
  support image-to-video/text-to-video).
- Feed effect *parameter values* (not just names) into the resync prompt
  — e.g. a Colorize hue value as an actual color descriptor.
- Photo booth: multi-region batch edits in one generation pass instead of
  one masked region at a time; GPU-accelerated rotoscope once CUDA 13
  onnxruntime wheels are published.
