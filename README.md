# resolume-genai-bridge

Quick-and-dirty local demo connecting Resolume to generative AI, in both
directions:

- **Trigger → generate**: a Resolume clip trigger (or a manual/test call)
  fires a generation and streams the result back in live as a Spout
  source.
- **Resolume → prompt**: pull whatever's currently live in Resolume
  (active clip names, active effects) and turn it into a text prompt, so
  you can regenerate/reinterpret what the VJ actually built instead of
  typing a prompt by hand.

Generation is backend-agnostic — swap between local Stable Diffusion
(ComfyUI), Runway, or Kling with one flag.

```
                    +-------------------------------+
Resolume clip  ---> |  bridge.py                     |
  trigger (OSC)     |   - clip -> prompts.json       |
                     |   - OR resync -> Resolume      |---> backend.generate_image(prompt)
Resolume state <---- |     REST API -> description    |         |
  (REST API)         |     -> prompt                  |         v
                     +-------------------------------+   ComfyUI / Runway / Kling
                                    |                            |
                                    v                            |
                          Spout sender "ComfyBridge" <-----------+
                                    |
                                    v
                    Resolume Sources > Spout > ComfyBridge (live layer)
                       or  python spout_viewer.py  (no Resolume needed)
```

## Prerequisites

- Windows (Spout is Windows-only)
- A generation backend, at least one of:
  - [ComfyUI](https://github.com/comfyanonymous/ComfyUI) running locally
    with a checkpoint installed (default `127.0.0.1:8188`)
  - Runway API access (`RUNWAYML_API_SECRET` env var) — https://docs.dev.runwayml.com
  - Kling AI API access (`KLING_ACCESS_KEY` / `KLING_SECRET_KEY` env vars) — https://kling.ai/document-api
- Resolume Arena or Avenue (optional — everything works against
  `test_trigger.py` and `spout_viewer.py` without it)
- Python 3.10+ (SpoutGL ships prebuilt wheels for common CPython versions)

## Setup

```
pip install -r requirements.txt
pip install -r requirements-viewer.txt   # only needed for spout_viewer.py
```

If you're using the ComfyUI backend, edit `workflows/txt2img_api.json` and
set `"ckpt_name"` (node `4`) to a checkpoint you actually have installed.

Edit `prompts.json` to map Resolume clip slots (`L<layer>C<clip>`) to
prompts for the clip-trigger path. `_default` covers any untracked clip.
The resync path doesn't need this file — it builds its prompt from
Resolume's live state instead.

## Run

```
python bridge.py --backend comfy              # local Stable Diffusion via ComfyUI
python bridge.py --backend runway
python bridge.py --backend kling
```

This starts an OSC listener (`:9000` by default) and a Spout sender named
`ComfyBridge`.

**See it without Resolume:**

```
python spout_viewer.py
```

opens a window showing whatever the bridge is currently sending.

**In Resolume** (once you're ready):
1. **Preferences > OSC** — enable OSC output, host `127.0.0.1`, port
   `9000` (must match `bridge.py`'s `--osc-port`). Resolume broadcasts
   OSC for every clip connect/disconnect automatically once this is on.
2. **Preferences > Webserver** — enable it (default port `8080`) if you
   want the resync path to read live composition state.
3. Add a layer, **Sources > Spout > ComfyBridge**.
4. Trigger any clip to fire a generation from `prompts.json`.

## Testing without Resolume open

```
python test_trigger.py --clip 1 1        # simulate a clip trigger
python test_trigger.py --prompt "a cat made of stained glass"
python test_trigger.py --resync           # pull live state from Resolume's REST API
```

`--resync` needs Resolume actually running with the webserver enabled,
since it reads real composition state; the other two work with just
`bridge.py` and a backend running.

## How resync builds a prompt

`resolume_state.py` calls Resolume's REST API
(`GET /api/v1/composition`), walks the visible layers, and for each one:
finds the active (connected) clip and uses its name as a prompt fragment,
then adds a short descriptor for each non-bypassed effect on that clip or
layer (`kaleidoscope` → "kaleidoscopic, symmetrical fractal patterns",
etc. — see `EFFECT_DESCRIPTORS`). Fragments are joined and a fixed style
suffix is appended. This is a heuristic name/effect → adjective mapping,
not real image understanding — clip names that already read as
descriptions (which is how most VJs name their clips anyway) work best.

## Known limitations (this is a demo, not production)

- Polls REST endpoints on a timer instead of using websocket/streaming
  APIs — simpler, slightly higher latency.
- One generation in flight at a time; triggers that arrive while busy are
  dropped rather than queued.
- Fixed output canvas (`SPOUT_WIDTH`/`SPOUT_HEIGHT` in `bridge.py`,
  default 512x512) — anything a backend returns gets resized to fit.
- `EFFECT_DESCRIPTORS` in `resolume_state.py` is a small hand-picked
  table, not a mapping of Resolume's full effect library.
- No retry/backoff on backend errors, no auth on the bridge's own OSC
  listener, single-machine only.

## Where this goes next

- Websocket/streaming completion detection instead of polling, for lower
  latency.
- Queue triggers instead of dropping them while busy.
- img2video backends (Runway/Kling both support image-to-video) so a
  trigger produces motion, not just a still.
- Feed effect *parameter values* (not just names) into the resync prompt
  — e.g. a Colorize hue value as an actual color descriptor.
