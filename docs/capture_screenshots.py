"""Regenerates the README's screenshots and demo GIF from the live app.

    python docs/capture_screenshots.py --image path/to/a/subject/photo.jpg

Driven through the real UI rather than mocked or hand-cropped, so the images
can't drift from what the app actually does -- a stale screenshot is a worse
lie than no screenshot. Needs ComfyUI and web_server.py already running, and
does a real generation, so expect it to take a couple of minutes.

Like the verify_* scripts, this is deliberately not a pytest test: it needs a
live stack and a photo the repo doesn't ship.
"""
import argparse
import io
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from PIL import Image
from playwright.sync_api import sync_playwright

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--image", required=True, help="a subject photo to run through the booth")
parser.add_argument("--server", default="http://127.0.0.1:8000/")
parser.add_argument("--brand", default="aurora")
parser.add_argument("--look", default="rooftop")
args = parser.parse_args()

SERVER = args.server
DOCS = Path(__file__).resolve().parent
PHOTO = Path(args.image)
if not PHOTO.is_file():
    raise SystemExit(f"no such photo: {PHOTO}")

frames = []


def grab(page, tag=""):
    frames.append(Image.open(io.BytesIO(page.screenshot())).convert("RGB"))
    if tag:
        print(f"  frame: {tag} ({len(frames)})")


def wait_filming(page, done, budget_s, label):
    """Waits for `done()` while collecting GIF frames, then subsamples them.

    The waits here are 20-60s of real model work; filming every second of that
    would make the wait most of the GIF's weight for none of its meaning.
    """
    start = time.time()
    shots = []
    while time.time() - start < budget_s:
        page.wait_for_timeout(1500)
        shots.append(Image.open(io.BytesIO(page.screenshot())).convert("RGB"))
        if done():
            break
    else:
        raise SystemExit(f"!! {label} did not finish within {budget_s}s")
    step = max(1, len(shots) // 5)
    frames.extend(shots[::step])
    print(f"  {label}: {time.time() - start:.0f}s, {len(shots)} frames -> every {step}")


def scroll_panels_to_top(page):
    """Puts every scrollable column back to the top, so a screenshot shows the
    panel headings rather than wherever the last interaction left them."""
    page.evaluate("document.querySelectorAll('*').forEach("
                  "e => { if (e.scrollHeight > e.clientHeight + 40) e.scrollTop = 0; })")
    page.wait_for_timeout(400)


with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1500, "height": 940}, device_scale_factor=1)
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: errors.append(f"console {m.type}: {m.text}")
            if m.type == "error" else None)

    page.goto(SERVER, wait_until="networkidle")
    page.wait_for_timeout(800)
    grab(page, "landing")

    # --- brand kit -----------------------------------------------------------
    page.select_option("#brandSelect", args.brand)
    page.wait_for_timeout(300)
    grab(page, "brand chosen")
    page.select_option("#lookSelect", args.look)
    page.wait_for_timeout(400)
    grab(page, "look chosen")

    # --- analyze -------------------------------------------------------------
    page.set_input_files("#fileInput", str(PHOTO))
    wait_filming(page, lambda: not page.is_disabled("#btnAddObject"), 180, "analysis")
    page.wait_for_timeout(800)
    grab(page, "analyzed")

    scroll_panels_to_top(page)
    page.screenshot(path=str(DOCS / "analysis.png"))
    print("wrote docs/analysis.png")
    page.screenshot(path=str(DOCS / "brand-kit.png"),
                    clip={"x": 1205, "y": 55, "width": 295, "height": 860})
    print("wrote docs/brand-kit.png")

    # --- logo ----------------------------------------------------------------
    page.click("#btnPlaceLogo")
    page.wait_for_timeout(600)
    grab(page, "logo placed")

    # --- generate ------------------------------------------------------------
    page.click("#btnGenerate")
    wait_filming(page,
                 lambda: (page.text_content("#progressText") or "").strip() == "done",
                 420, "generation")
    page.wait_for_timeout(1500)
    for _ in range(4):
        grab(page, "result")
        page.wait_for_timeout(400)

    scroll_panels_to_top(page)
    page.screenshot(path=str(DOCS / "photo-booth.png"))
    print("wrote docs/photo-booth.png")

    # The canvas on its own, guides off: this is the deliverable, and the
    # dashed clear-space rectangle is scaffolding that shouldn't be in the
    # picture of the output. (Placing a logo turns the guide on by default,
    # hence the toggle rather than a plain screenshot -- the full-window shot
    # above already shows the guides in context.)
    if "ON" in (page.text_content("#btnSafeArea") or ""):
        page.click("#btnSafeArea")
        page.wait_for_timeout(500)
    grab(page, "guides off")
    page.screenshot(path=str(DOCS / "result.png"),
                    clip=page.locator("#compositeCanvas").bounding_box())
    print("wrote docs/result.png")

    # Element screenshot rather than a viewport clip: it scrolls the card into
    # view first, which a clip against the current viewport does not.
    page.locator("#provenanceCard").screenshot(path=str(DOCS / "provenance.png"))
    print("wrote docs/provenance.png")

    print("errors:", errors or "none")
    browser.close()

# --- assemble the GIF --------------------------------------------------------
# Halved and palettised: a full-size GIF is tens of megabytes and GitHub will
# not inline it.
small = [f.resize((f.width // 2, f.height // 2), Image.LANCZOS) for f in frames]
small = [f.quantize(colors=128, method=Image.MEDIANCUT, dither=Image.FLOYDSTEINBERG)
         for f in small]
durations = [900] + [500] * (len(small) - 2) + [3000]
small[0].save(DOCS / "demo.gif", save_all=True, append_images=small[1:],
              duration=durations, loop=0, optimize=True)
print(f"wrote docs/demo.gif  ({len(small)} frames, "
      f"{(DOCS / 'demo.gif').stat().st_size / 1024 / 1024:.1f} MB)")
