"""
Proves the project .json export/import round-trip preserves the canvas-fit
feature's state (canvas dimensions, fit-offer UI) instead of silently
resetting to the square default on reload.

Needs ComfyUI + web_server.py already running, plus a NON-square subject
photo (the point is that a fitted canvas survives the round trip). The
default is gitignored, so pass your own on a fresh clone:

    python verify_project_roundtrip.py --image portrait.jpg
"""

import argparse

from playwright.sync_api import sync_playwright

import verify_common

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--base-url", default=verify_common.DEFAULT_BASE_URL)
parser.add_argument("--image", default=verify_common.DEFAULT_PORTRAIT_PHOTO,
                     help="a non-square subject photo")
args = parser.parse_args()

PORTRAIT = verify_common.resolve_image(args.image, "--image", "a non-square subject photo")
OUT = verify_common.out_dir()
verify_common.preflight(args.base_url)

with sync_playwright() as pw:
    browser = pw.chromium.launch(headless=True)
    page = browser.new_page()
    errors = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: errors.append(str(e)))

    page.goto(args.base_url)
    page.wait_for_function("() => document.getElementById('connStatus').textContent === 'connected'")

    page.set_input_files("#fileInput", PORTRAIT)
    page.wait_for_function(
        "() => document.getElementById('footerStatus').textContent.includes('ready to generate')",
        timeout=90_000,
    )
    page.click("#btnFitCanvas")  # fit to the portrait photo
    canvas_before = page.evaluate("() => [document.getElementById('compositeCanvas').width, document.getElementById('compositeCanvas').height]")
    print(f"canvas size before export: {canvas_before}")

    # export is disabled until at least one generation completes -- fire a
    # real one so there's something meaningful to round-trip
    print("generating a background (needed to enable export)...")
    page.fill("#scenePrompt", "plain grey studio backdrop, softbox lighting")
    page.click("#btnGenerate")
    page.wait_for_function("() => document.getElementById('progressText').textContent !== 'done'", timeout=15_000)
    page.wait_for_function("() => document.getElementById('progressText').textContent === 'done'", timeout=180_000)

    with page.expect_download(timeout=10_000) as dl_info:
        page.click("#btnExportJson")
    download = dl_info.value
    project_path = str(OUT / "roundtrip_project.json")
    download.save_as(project_path)
    print(f"exported to {project_path}")

    # fresh page/session, import the project, and check canvas dims + fit-offer state restore correctly
    page2 = browser.new_page()
    errors2 = []
    page2.on("console", lambda m: errors2.append(m.text) if m.type == "error" else None)
    page2.on("pageerror", lambda e: errors2.append(str(e)))
    page2.goto(args.base_url)
    page2.wait_for_function("() => document.getElementById('connStatus').textContent === 'connected'")

    page2.set_input_files("#importInput", project_path)
    page2.wait_for_function(
        "() => document.getElementById('footerStatus').textContent === 'project imported'",
        timeout=10_000,
    )
    canvas_after = page2.evaluate("() => [document.getElementById('compositeCanvas').width, document.getElementById('compositeCanvas').height]")
    fit_box_visible = page2.is_visible("#canvasFitBox")
    fit_btn_text = page2.text_content("#btnFitCanvas") if fit_box_visible else None
    print(f"canvas size after import: {canvas_after}")
    print(f"fit box visible after import: {fit_box_visible}")
    print(f"fit button text after import: {fit_btn_text!r}")

    assert canvas_after == canvas_before, f"canvas size did not survive export/import roundtrip: {canvas_before} -> {canvas_after}"
    assert fit_box_visible, "fit offer should still appear after import (photo is still non-square)"
    assert fit_btn_text and "square" in fit_btn_text.lower(), "button should reflect that canvas is currently fitted, not square"

    all_errors = errors + errors2
    print(f"console errors: {all_errors}")
    assert not all_errors, f"unexpected console errors: {all_errors}"

    print("\nROUNDTRIP TEST PASSED")
    browser.close()
