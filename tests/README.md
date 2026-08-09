# tests/

Fast, offline unit tests. No ComfyUI, no GPU, no network, no Playwright, no
model downloads, no photos you have to supply yourself:

```
pip install pytest
pytest
```

They cover the parts of the pipeline that are pure logic and were previously
verified only by eye during a live demo — mask cleanup, illumination
estimation, resolution capping, ControlNet-strength heuristics, provenance
extraction, cover-fit geometry, and the multi-session job routing in
`web_server.py` (against a fake backend, so the routing is tested without a
GPU in the loop).

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
