"""
Proves the canvas-fit offer (web/index.html) shows only for non-square
photos, resizes the canvas correctly when accepted, and toggles back.

Needs ComfyUI + web_server.py already running, and
test_images/pexels_6668809.jpg (portrait) + fallback_examples/loft_interior_preview.png
(square, for the negative case). Both gitignored -- swap in your own on a
fresh clone.
"""

from playwright.sync_api import sync_playwright

with sync_playwright() as pw:
    browser = pw.chromium.launch(headless=True)
    page = browser.new_page()
    errors = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: errors.append(str(e)))

    page.goto("http://127.0.0.1:8000")
    page.wait_for_function("() => document.getElementById('connStatus').textContent === 'connected'")

    # portrait photo -- should trigger the fit offer
    page.set_input_files("#fileInput", "test_images/pexels_6668809.jpg")
    page.wait_for_function(
        "() => document.getElementById('footerStatus').textContent.includes('ready to generate')",
        timeout=90_000,
    )

    box_visible = page.is_visible("#canvasFitBox")
    hint = page.text_content("#canvasFitHint")
    btn_text = page.text_content("#btnFitCanvas")
    canvas_size_before = page.evaluate("() => [document.getElementById('compositeCanvas').width, document.getElementById('compositeCanvas').height]")
    print(f"offer box visible: {box_visible}")
    print(f"hint: {hint!r}")
    print(f"button: {btn_text!r}")
    print(f"canvas size before fit: {canvas_size_before}")
    assert box_visible, "canvas-fit offer should appear for a non-square photo"
    assert "landscape" in hint or "portrait" in hint, f"hint should describe orientation: {hint!r}"

    page.screenshot(path="verify_out/canvas_fit_before.png")

    page.click("#btnFitCanvas")
    page.wait_for_timeout(300)
    canvas_size_after = page.evaluate("() => [document.getElementById('compositeCanvas').width, document.getElementById('compositeCanvas').height]")
    btn_text_after = page.text_content("#btnFitCanvas")
    print(f"canvas size after fit: {canvas_size_after}")
    print(f"button after: {btn_text_after!r}")
    assert canvas_size_after != canvas_size_before, "canvas should have resized"
    assert "square" in btn_text_after.lower(), "button should now offer to go back to square"

    page.screenshot(path="verify_out/canvas_fit_after.png")

    # square photo -- offer should NOT appear
    page2 = browser.new_page()
    page2.goto("http://127.0.0.1:8000")
    page2.wait_for_function("() => document.getElementById('connStatus').textContent === 'connected'")
    page2.set_input_files("#fileInput", "fallback_examples/loft_interior_preview.png")
    page2.wait_for_function(
        "() => document.getElementById('footerStatus').textContent.includes('ready to generate')",
        timeout=90_000,
    )
    square_box_visible = page2.is_visible("#canvasFitBox")
    print(f"offer box visible for square photo: {square_box_visible}")
    assert not square_box_visible, "canvas-fit offer should NOT appear for an already-square photo"

    print(f"\nconsole errors: {errors}")
    assert not errors, f"unexpected console errors: {errors}"
    print("\nALL CHECKS PASSED")

    browser.close()
