# resolume-genai-bridge

[![tests](https://github.com/BGavazzi/mixed-reality-photobooth/actions/workflows/tests.yml/badge.svg)](https://github.com/BGavazzi/mixed-reality-photobooth/actions/workflows/tests.yml)

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
  |    cap_resolution()   -> downscale if oversized (see below)   |
  |    segment_subject()  -> rembg/birefnet-portrait -> cutout+mask
  |                          (+ drop tiny disconnected mask blobs)|
  |    estimate_pose()    -> OpenPose/DWPose (controlnet_aux)     |
  |    estimate_depth()   -> MiDaS (controlnet_aux)               |
  |    estimate_illumination() -> plain CV on the subject pixels  |
  |         (light direction / warmth / softness -> text descriptor)
  |    suggest_controlnet_strength() -> lower default if the      |
  |         background has real depth structure to fight against |
  v                                                                |
Browser: layer stack (background / subject / pose / depth, <------+
          each independently visible/opaque/movable/scalable/
          rotatable; canvas can fit the photo's own aspect ratio
          instead of always cropping to a fixed square)
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
pip install --force-reinstall opencv-python
```

The second line matters: `controlnet_aux` pulls in `opencv-python-headless`,
which silently overwrites the GUI-capable `opencv-python` build on disk
(same `cv2` module name, one wins) — without this, `spout_viewer.py` and
the Resolume bridge's video preview fail with a `waitKeyImpl` error.

Models (ComfyUI's `models/` folder):
- Checkpoint: [`RealVisXL_V5.0_fp16.safetensors`](https://huggingface.co/SG161222/RealVisXL_V5.0) in `models/checkpoints/`
- ControlNet: [`diffusers_xl_depth_full.safetensors`](https://huggingface.co/lllyasviel/sd_control_collection) in `models/controlnet/`
- Rotoscope/pose/depth models (`birefnet-portrait`, OpenPose, MiDaS) download automatically on first use via `rembg`/`controlnet_aux`.

### Run

```
powershell -ExecutionPolicy Bypass -File start_demo.ps1
```

Starts ComfyUI (with `--preview-method auto`), waits for it to be ready,
checks that the checkpoint and ControlNet the workflow names are actually
installed (via ComfyUI's `/object_info`, so a missing model is reported at
startup instead of ~40s into the first generation), starts `web_server.py`,
waits for that, opens the browser. Run `stop_demo.ps1` to shut both down.

Paths are derived from the script's own location; ComfyUI is assumed to be a
sibling directory of this repo, overridable with `COMFYUI_DIR`:

```
$env:COMFYUI_DIR = "C:\path\to\ComfyUI"; .\start_demo.ps1
``` (If you double-click the script rather
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

### Real-photo hardening

Everything above was originally validated against one curated 1024x1024
test image. Running real (non-square, full-resolution) photos through it
with a specific target shot in mind — not just checking that requests
succeed — surfaced four issues that don't show up on a clean square test
photo:

- **No resize node in `photoshoot_bg_api.json`** means SDXL's `VAEEncode`
  runs at the uploaded photo's *native* resolution. A realistic 22MP
  camera photo drove ComfyUI into a VAE out-of-memory fallback
  (`retrying with tiled VAE encoding`) and dropped sampling from
  ~1.2s/step to ~41s/step — on track for ~20 minutes with zero error
  shown to the user. `cap_resolution()` downscales to 1536px max before
  any model sees the image; generation resolution otherwise still tracks
  the photo's own aspect ratio, just capped.
- **The canvas was a fixed 768x768 square**, so any non-square photo got
  stretched (not cropped — `drawImage` with an explicit target size
  distorts) to fill it, visibly warping body proportions. The canvas now
  cover-fits each layer to its real aspect ratio, and the UI offers to
  resize the canvas itself to the photo's own proportions so nothing gets
  cropped at all (opt-in — square stays the default).
- **rembg occasionally classifies a small disconnected patch of a busy
  background as subject** — a real floating artifact (e.g. a chair-leg
  sliver), not a body part, since alpha thresholding has no notion of
  connectivity. Fixed with a connected-component pass that drops blobs
  below an area-ratio threshold relative to the main subject blob, rather
  than naively keeping only the single largest one (which would also
  discard legitimately-disconnected parts like a held object or jewelry).
- **ControlNet depth strength (0.75 default) can over-anchor to the
  original scene's geometry** when the background already has real
  structure — a "cozy reading nook" prompt over a room with a chair in it
  barely changed the room, because the depth map still encoded that
  chair's exact geometry. `suggest_controlnet_strength()` measures depth
  variance in the background region and suggests a lower value (0.45)
  only when there's real structure to fight against; a plain backdrop
  still defaults to 0.75, where the higher strength actually helps.

### Known limitations

- Rotoscope (`birefnet-portrait`) runs CPU-only, ~20s/photo — this
  machine's `onnxruntime-gpu` wants CUDA 13 libraries that aren't
  published as pip wheels yet (checked: `nvidia-cublas-cu13` on PyPI is
  a version-`0.0.1` placeholder).
- Multiple browser sessions can connect and queue concurrently — each `/ws`
  connection gets its own session id, and ComfyUI's events route back by
  `prompt_id → session` rather than to a single global "whoever connected
  last." ComfyUI itself still renders one graph at a time (a real GPU
  constraint), so concurrent sessions queue through its own `/prompt` queue;
  what the routing guarantees is that each session gets *its own* result.
  The CPU analysis stages serialize behind a lock for the same reason —
  the cached PyTorch model instances aren't safe for concurrent forward
  passes (reproduced: two simultaneous `/api/analyze` calls crashed inside
  MiDaS).
- The illumination estimate is classic CV (per-quadrant luminance,
  highlight color, contrast), not a learned model — a useful heuristic
  for prompt-grounding, not a physically accurate light probe.
- Generate Background / Relight always condition on the *original*
  uploaded photo's mask/depth, not the current live composite — so an
  object added via the region-draw tool stays in place (drawn on top)
  but isn't re-conditioned if you regenerate the background afterward,
  and can end up visually mismatched with the new scene. The UI surfaces
  an explicit warning when this would happen rather than silently
  producing a mismatched result; region-edit itself doesn't have this
  problem, since it conditions on the current composite directly.
- Node IDs in the ComfyUI workflow JSON files are hardcoded per exported
  template (e.g. `PHOTOSHOOT_POSITIVE_PROMPT_NODE = "7"`) — a workflow
  re-exported from the ComfyUI UI could silently shift those IDs and
  break the app. A production version would want a small schema mapping
  semantic node roles to IDs per workflow version. Partially mitigated:
  `tests/test_provenance.py` asserts every hardcoded ID still resolves to a
  node with the expected input, so a renumbered workflow fails in CI rather
  than at generation time — and the two model *names* in the provenance
  record are already read by `class_type` instead of by ID.

### Testing

Two complementary layers.

**1. Offline unit tests** — no ComfyUI, no GPU, no photos, ~1 second:

```
pip install -r requirements-test.txt
pytest
```

Covers the pure-logic parts of the pipeline (mask blob cleanup, illumination
estimation, resolution capping, the ControlNet-strength heuristic, contact
shadow geometry, cover-fit), the provenance extraction, and the multi-session
job routing in `web_server.py` — the last driven against a fake backend, so
the cross-session-leak cases can be checked without a GPU in the loop. Runs
in CI on Python 3.10 and 3.12 (`.github/workflows/tests.yml`).

**2. End-to-end verification** — needs the real stack running.

`verify_web_ui.py` drives a real Chromium via Playwright through the full
click-through path (upload → analyze → generate background → region-draw
object → relight → voice button → living-photo export → disclosure copy
→ PNG/JSON export → Spout send), since the canvas/layer JS in
`web/index.html` previously had no coverage beyond a Python websocket
test client:

```
pip install playwright && playwright install chromium
python verify_web_ui.py --image path\to\any\subject\photo.jpg
```

Needs ComfyUI and `web_server.py` already running. Screenshots and any
exported downloads land in `verify_out/<timestamp>/` (gitignored).

Three narrower scripts sit alongside it, same requirements, each taking its
own photos on the command line:

```
python verify_multi_session.py    --image-a one.jpg --image-b two.jpg
python verify_canvas_fit.py       --image portrait.jpg --square-image square.png
python verify_project_roundtrip.py --image portrait.jpg
```

`verify_multi_session.py` is the interesting one: two concurrent tabs firing
real overlapping generations, asserting neither receives the other's result.

They're named `verify_*` rather than `test_*` precisely because they *aren't*
collectable tests — they parse argv, drive live servers, and need photos the
repo doesn't ship. `pytest` means `tests/` only.

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
  `send_trigger.py` and `spout_viewer.py` without it)
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
python send_trigger.py --clip 1 1        # simulate a clip trigger
python send_trigger.py --prompt "a cat made of stained glass"
python send_trigger.py --video "a cat made of stained glass, slow pan"  # ComfyUI backend only
python send_trigger.py --resync           # pull live state from Resolume's REST API
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
  gets fit to it. `SpoutFrameBuffer.set_image()` (shared by both apps via
  `spout_output.py`) cover-fits and crops to the sender's aspect ratio
  rather than stretching, so non-square sources no longer get squashed —
  they get cropped to fill instead, same as any normal video source.
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
  onnxruntime wheels are published; re-conditioning Generate
  Background/Relight on the *current* composite instead of always the
  original upload (see Known limitations above).
- Per-layer Spout output — right now the photo booth sends one flattened
  composite; for an actual on-set mixed-reality setup, each layer group
  (background, subject, added objects) as its own named Spout source
  would let a projection-mapping tool place them independently on real
  projectors/LED panels instead of one flattened, already-composited
  frame.
- RAW format support / direct camera tethering (gPhoto2/PTP) for the
  photo booth's input side, instead of `<input type=file>` only.
