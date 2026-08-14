# tests/

Fast, offline unit tests. No ComfyUI, no GPU, no network, no Playwright, no
model downloads, no photos you have to supply yourself:

```
pip install -r ../requirements-test.txt
pytest
```

That file is deliberately much smaller than `requirements.txt` — no rembg, no
controlnet_aux, no torch, no SpoutGL. The model wrappers are lazy-loaded and
the ComfyUI backend is faked, so the suite never reaches them, which is what
keeps a full run under a couple of seconds.

They cover the parts of the pipeline that are pure logic and were previously
verified only by eye during a live demo — mask cleanup, illumination
estimation, resolution capping, ControlNet-strength heuristics, provenance
extraction, cover-fit geometry, the Resolume prompt descriptors, the
`/api/analyze` request handling, `doctor.py`'s diagnosis logic, brand-kit
parsing and prompt composition, workflow role resolution, the generation
queue, and the multi-session job routing in `web_server.py` (against a fake
backend, so the routing is tested without a GPU in the loop).

`test_doctor.py` is the reason the checks in `doctor.py` take their facts as
arguments rather than discovering them: a diagnostic can only be tested
against a broken environment if it doesn't need to *be* in one.

`test_brand_enforcement.py` is a different shape from the rest and worth
reading as such: it drives the real websocket handlers with a recording
backend to assert a *security-ish* property rather than a computed value --
that the browser can only ever send `{brand_id, look_id, free text}`, and
that a request smuggling its own `negative_prompt` or `seed` is ignored. If
that stops holding, a modified client could drop a client's blocklist while
every other brand-kit test still passed.

`test_workflow_graph.py` is mostly about *refusing to guess*. It resolves
workflow node ids from graph structure, and the dangerous failure is not
"found nothing" (which raises) but "found the wrong node of the right kind" --
writing a prompt into the negative conditioning produces a plausible bad
image that reads as a model problem rather than a bug.

## Why this exists separately from the `verify_*.py` scripts

The repo root holds a second, complementary kind of check:

| | `tests/` | `verify_*.py` |
|---|---|---|
| Runs on a clean clone | yes | no |
| Needs ComfyUI + `web_server.py` up | no | yes |
| Needs subject photos (gitignored) | no | yes |
| Runtime | ~1s | minutes (real generations) |
| Proves | logic is correct | the whole stack works end to end |

Both matter. The end-to-end scripts are what actually caught the real-photo
bugs documented in the README, and nothing here replaces them. But they
can't run in CI and can't run for anyone who just cloned the repo, which
left every pure-logic function in the project with no automated coverage at
all.

Those scripts used to be named `test_*.py`, which meant a bare `pytest`
collected them, executed their module bodies at import, and failed trying to
launch Chromium against a server that wasn't running. Renaming them to
`verify_*` (matching `verify_web_ui.py`, which already used that convention)
makes the split explicit and makes `pytest` mean one unambiguous thing.
