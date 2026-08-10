"""
Automated end-to-end verification of the mixed-reality photo booth's browser
UI (web/index.html) -- the one layer of this app that had only ever been
exercised through a Python websocket test client, never through an actual
browser (see README's "Known limitations"). Drives a real Chromium via
Playwright through the full click-through path: connect -> upload/analyze ->
generate background -> add-object region edit -> relight -> voice-button
fallback -> living photo export -> disclosure copy -> PNG/JSON export ->
Spout send. Stops and reports at the first stage that doesn't reach its
expected DOM state, with a screenshot and any JS console errors, so a broken
UI is caught before a live demo instead of during one.

Requires ComfyUI and web_server.py already running -- this only drives the
browser, the same assumption send_trigger.py makes about bridge.py.

    pip install playwright
    playwright install chromium
    python verify_web_ui.py --image path\\to\\any\\subject\\photo.jpg
    python verify_web_ui.py --headed --image path\\to\\photo.jpg --slow-mo 150

The default --image lives in fallback_examples/, which is gitignored --
on a fresh clone you'll need to pass your own subject photo explicitly.

Screenshots and exported downloads land in verify_out/<timestamp>/.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import verify_common
from console_encoding import use_utf8_console

# Windows consoles default to cp1252, which raises on emoji in button text
# (e.g. the voice-input button's mic icon).
use_utf8_console()

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("Playwright not installed. Run:\n  pip install playwright\n  playwright install chromium")
    sys.exit(1)

DEFAULT_IMAGE = Path(__file__).parent / "fallback_examples" / "loft_interior_preview.png"
SCENE_PROMPT = "sunlit rooftop terrace at golden hour, city skyline"
OBJECT_LABEL = "a small wooden stool"
GEN_TIMEOUT_MS = 120_000  # SDXL background/region generation
ANALYZE_TIMEOUT_MS = 90_000  # CPU rotoscope/pose/depth, ~20-30s each


class Verifier:
    def __init__(self, page, out_dir: Path):
        self.page = page
        self.out_dir = out_dir
        self.results = []  # (name, ok, detail, elapsed_s)
        self.console_errors = []
        page.on("console", self._on_console)
        page.on("pageerror", lambda exc: self.console_errors.append(f"pageerror: {exc}"))

    def _on_console(self, msg):
        if msg.type == "error":
            self.console_errors.append(f"console.error: {msg.text}")

    def screenshot(self, name: str):
        path = self.out_dir / f"{name}.png"
        self.page.screenshot(path=str(path))
        return path

    def stage(self, name):
        """Context manager: times a stage, catches failures, screenshots either way."""
        return _Stage(self, name)


class _Stage:
    def __init__(self, verifier: Verifier, name: str):
        self.v = verifier
        self.name = name
        self.start = None

    def __enter__(self):
        self.start = time.time()
        print(f"[{len(self.v.results) + 1}] {self.name} ...", flush=True)
        return self.v.page

    def __exit__(self, exc_type, exc, tb):
        elapsed = time.time() - self.start
        slug = self.name.lower().replace(" ", "_").replace("/", "_")
        shot = self.v.screenshot(slug)
        if exc_type is not None:
            detail = f"{exc_type.__name__}: {exc}"
            self.v.results.append((self.name, False, detail, elapsed))
            print(f"    FAILED ({elapsed:.1f}s): {detail}")
            print(f"    screenshot: {shot}")
            return False  # propagate — later stages depend on this one's state, so run() stops the whole flow
        self.v.results.append((self.name, True, "ok", elapsed))
        print(f"    ok ({elapsed:.1f}s)")
        return False


def wait_text(page, selector, expected, timeout):
    page.wait_for_function(
        "([sel, exp]) => document.querySelector(sel) && document.querySelector(sel).textContent.trim() === exp",
        arg=[selector, expected],
        timeout=timeout,
    )


def wait_generation_done(page, timeout=GEN_TIMEOUT_MS):
    """Waits for a *newly triggered* ComfyUI generation to finish. Checking
    only for progressText === 'done' races the action that triggers the new
    job: the 'queued' websocket message that overwrites progressText hasn't
    necessarily arrived yet when this is called, so it can see the *previous*
    generation's leftover 'done' text and return immediately, before the new
    job has even started. Wait for the 'queued...' transition first (proof
    the new cycle began) and only then for 'done'."""
    page.wait_for_function(
        "() => document.getElementById('progressText').textContent !== 'done'",
        timeout=15_000,
    )
    page.wait_for_function(
        "() => document.getElementById('progressText').textContent === 'done'",
        timeout=timeout,
    )


def run(base_url: str, image_path: Path, headed: bool, slow_mo: int, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    ok_overall = True
    # Bound before the browser exists: the summary block at the bottom reads
    # v.results unconditionally, so a failure between launching Chromium and
    # constructing the Verifier used to surface as a confusing NameError
    # instead of the real error.
    v = None

    with sync_playwright() as pw:
        try:
            browser = pw.chromium.launch(headless=not headed, slow_mo=slow_mo)
        except Exception as exc:
            print(f"Could not launch Chromium ({exc}).\nRun: playwright install chromium")
            sys.exit(1)
        context = browser.new_context(accept_downloads=True)
        try:
            context.grant_permissions(["clipboard-read", "clipboard-write"], origin=base_url)
        except Exception as exc:
            print(f"(clipboard permission grant skipped: {exc})")
        page = context.new_page()
        v = Verifier(page, out_dir)

        # dialog handling is contextual (prompt() during region-draw needs an
        # answer; alert() elsewhere just needs dismissing) -- a mutable box
        # the per-stage code can set before triggering the dialog.
        dialog_box = {"answer": None, "seen": []}

        def on_dialog(dialog):
            dialog_box["seen"].append((dialog.type, dialog.message))
            if dialog.type == "prompt" and dialog_box["answer"] is not None:
                dialog.accept(dialog_box["answer"])
            else:
                dialog.accept()

        page.on("dialog", on_dialog)

        try:
            with v.stage("Connect and load") as p:
                p.goto(base_url, timeout=30_000)
                wait_text(p, "#connStatus", "connected", 15_000)

            with v.stage("Upload and analyze photo") as p:
                p.set_input_files("#fileInput", str(image_path))
                wait_text(p, "#footerStatus", "subject analyzed — ready to generate a background", ANALYZE_TIMEOUT_MS)
                assert p.is_visible("#stagesPanel"), "stages panel did not appear"
                for sel in ("#thumbCutout", "#thumbPose", "#thumbDepth"):
                    src = p.get_attribute(sel, "src")
                    assert src and src.startswith("data:image"), f"{sel} has no image data"
                illum = p.text_content("#illumText")
                assert illum and illum.strip(), "illumination text is empty"
                print(f"    illumination: {illum.strip()!r}")

            with v.stage("Generate background") as p:
                p.fill("#scenePrompt", SCENE_PROMPT)
                p.click("#btnGenerate")
                wait_generation_done(p)
                card = p.text_content("#provenanceCard")
                assert "type: background" in card, f"unexpected provenance: {card!r}"
                assert not p.eval_on_selector(
                    "#eventLog", "el => el.lastElementChild && el.lastElementChild.classList.contains('ev-error')"
                ), "last event log line was an error"
                print(f"    provenance: {card.splitlines()[0:2]}")

            with v.stage("Region draw: add object") as p:
                dialog_box["answer"] = OBJECT_LABEL
                p.click("#btnAddObject")
                box = p.eval_on_selector(
                    "#compositeCanvas",
                    "el => { const r = el.getBoundingClientRect(); return {x: r.x, y: r.y, w: r.width, h: r.height}; }",
                )
                cx, cy = box["x"] + box["w"] / 2, box["y"] + box["h"] / 2
                half = max(box["w"], box["h"]) * 0.22  # -> >=44% of canvas per side, safely above the 30% floor
                p.mouse.move(cx - half, cy - half)
                p.mouse.down()
                p.mouse.move(cx + half, cy + half, steps=10)
                p.mouse.up()
                assert any(t == "prompt" for t, _ in dialog_box["seen"]), "region-label prompt() never fired"
                wait_generation_done(p)
                card = p.text_content("#provenanceCard")
                assert "type: region" in card, f"unexpected provenance after region edit: {card!r}"
                dialog_box["answer"] = None

            with v.stage("Relight (time-of-day)") as p:
                p.eval_on_selector(
                    "#timeOfDaySlider",
                    "(el) => { el.value = '0'; el.dispatchEvent(new Event('input')); }",
                )
                assert p.text_content("#timeOfDayLabel").strip() == "Dawn"
                p.click("#btnRelight")
                wait_generation_done(p)
                card = p.text_content("#provenanceCard")
                assert "pre-dawn" in card or "dawn" in card.lower(), f"relight prompt missing time-of-day phrase: {card!r}"

            with v.stage("Voice button (best-effort, not gating)") as p:
                # Real behavior here depends on the OS/browser's microphone
                # permission plumbing (headless Chromium exposes
                # webkitSpeechRecognition but has no audio device, and how
                # long it takes to surface that as onerror vs. just hanging
                # in "listening" varies by environment) -- not something
                # worth blocking the whole suite over. Recorded as a note,
                # not asserted on; verify the mic/permission behavior by eye
                # in a real browser instead.
                dialog_box["seen"].clear()
                p.eval_on_selector("#footerStatus", "el => el.textContent = ''")
                p.click("#btnVoicePrompt")
                p.wait_for_timeout(2000)
                alerts = [msg for t, msg in dialog_box["seen"] if t == "alert"]
                footer = p.text_content("#footerStatus")
                btn_text = p.text_content("#btnVoicePrompt")
                print(f"    observed (informational only): alerts={alerts!r} footer={footer!r} button={btn_text!r}")

            with v.stage("Living photo export (.webm)") as p:
                with p.expect_download(timeout=15_000) as dl_info:
                    p.click("#btnExportLiving")
                download = dl_info.value
                dest = out_dir / "living_photo.webm"
                download.save_as(str(dest))
                assert dest.stat().st_size > 1000, "living photo export is suspiciously small"
                print(f"    saved: {dest} ({dest.stat().st_size} bytes)")

            with v.stage("Copy AI disclosure text") as p:
                p.click("#btnCopyDisclosure")
                p.wait_for_timeout(300)
                clip = p.evaluate("() => navigator.clipboard.readText().catch(() => null)")
                if clip:
                    assert "seed:" in clip.lower(), f"clipboard content missing expected fields: {clip!r}"
                    print(f"    clipboard: {clip.splitlines()[0]}")
                else:
                    print("    (clipboard read unavailable in this context — skipped content check)")

            with v.stage("Export flattened PNG") as p:
                with p.expect_download(timeout=10_000) as dl_info:
                    p.click("#btnExportPng")
                download = dl_info.value
                dest = out_dir / "composite.png"
                download.save_as(str(dest))
                assert dest.stat().st_size > 1000, "PNG export is suspiciously small"

            with v.stage("Export project JSON") as p:
                with p.expect_download(timeout=10_000) as dl_info:
                    p.click("#btnExportJson")
                download = dl_info.value
                dest = out_dir / "project.json"
                download.save_as(str(dest))
                data = json.loads(dest.read_text(encoding="utf-8"))
                assert data.get("provenanceLog"), "exported project has no provenanceLog"
                assert len(data["provenanceLog"]) >= 3, "expected background + region + relight generations logged"
                print(f"    provenanceLog entries: {len(data['provenanceLog'])}")

            with v.stage("Send to Spout") as p:
                errors_before = len(v.console_errors)
                p.click("#btnSendSpout")
                # The button no longer flips to a success label on click; it
                # waits for the server to confirm the frame reached a live
                # Spout sender. Assert only that it *resolved* one way or the
                # other -- staying on "Sending…" means the round trip broke,
                # which is a real failure. Whether Spout itself is running is
                # a property of the machine, not of the app, so an honest
                # failure label is reported rather than asserted on.
                p.wait_for_function(
                    "() => document.getElementById('btnSendSpout').textContent !== 'Sending…'",
                    timeout=10_000,
                )
                label = p.text_content("#btnSendSpout").strip()
                assert len(v.console_errors) == errors_before, "console error fired after Send to Spout"
                if "✓" in label:
                    print(f"    server confirmed the frame reached the PhotoBooth Spout sender ({label!r})")
                else:
                    print(f"    server reported the send did not reach a live Spout sender ({label!r}) — "
                          f"expected on a machine without SpoutGL; check the web_server log")

        except Exception as exc:
            print(f"\nStopped early: {exc}")
            ok_overall = False
        finally:
            context.close()
            browser.close()

    if v is None:
        print("\nThe browser session never started — nothing to summarise.")
        return False

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, ok, detail, elapsed in v.results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name} ({elapsed:.1f}s)" + ("" if ok else f" — {detail}"))
    ok_overall = ok_overall and all(ok for _, ok, _, _ in v.results)

    if v.console_errors:
        print(f"\n{len(v.console_errors)} browser console error(s) seen during the run:")
        for e in v.console_errors[:20]:
            print(f"  - {e}")
        ok_overall = False

    print(f"\nScreenshots + exported files: {out_dir}")
    print("RESULT:", "PASS" if ok_overall else "FAIL")
    return ok_overall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--image", default=str(DEFAULT_IMAGE),
                         help="subject photo to upload (default is in fallback_examples/, which is gitignored -- "
                              "pass your own path on a fresh clone)")
    parser.add_argument("--headed", action="store_true", help="show the browser window instead of running headless")
    parser.add_argument("--slow-mo", type=int, default=0, help="ms delay between actions, for watching it run")
    parser.add_argument("--out", default=None, help="output dir for screenshots/downloads (default: verify_out/<timestamp>)")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        if image_path == DEFAULT_IMAGE:
            print(f"Default test image not found: {image_path}\n"
                  f"fallback_examples/ is gitignored (not tracked in the repo), so a fresh clone won't have it.\n"
                  f"Pass any subject photo explicitly: python verify_web_ui.py --image path\\to\\photo.jpg")
        else:
            print(f"Image not found: {image_path}")
        sys.exit(1)

    out_dir = Path(args.out) if args.out else verify_common.OUT_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    verify_common.preflight(args.base_url)
    ok = run(args.base_url, image_path, args.headed, args.slow_mo, out_dir)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
