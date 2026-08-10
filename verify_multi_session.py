"""
Proves multi-session job routing actually works: two concurrent browser
sessions, different photos, different prompts, fired close together so
their jobs overlap in ComfyUI's own queue -- then verifies each session's
websocket only ever receives ITS OWN result, never the other session's.

Needs ComfyUI + web_server.py already running, plus two DIFFERENT subject
photos -- different, so that a result leaking between sessions is visible.
The defaults are gitignored, so pass your own on a fresh clone:

    python verify_multi_session.py --image-a one.jpg --image-b two.jpg
"""

import argparse

from playwright.sync_api import sync_playwright

import verify_common

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--base-url", default=verify_common.DEFAULT_BASE_URL)
parser.add_argument("--image-a", default=verify_common.DEFAULT_SECOND_PHOTO, help="subject photo for session A")
parser.add_argument("--image-b", default=verify_common.DEFAULT_SQUARE_PHOTO, help="subject photo for session B")
args = parser.parse_args()

IMAGE_A = verify_common.resolve_image(args.image_a, "--image-a", "a subject photo for session A")
IMAGE_B = verify_common.resolve_image(args.image_b, "--image-b", "a different subject photo for session B")
verify_common.preflight(args.base_url)

with sync_playwright() as pw:
    browser = pw.chromium.launch(headless=True)

    session_a = {"errors": [], "provenance": None, "done": False}
    session_b = {"errors": [], "provenance": None, "done": False}

    page_a = browser.new_page()
    page_b = browser.new_page()
    page_a.on("console", lambda m: session_a["errors"].append(m.text) if m.type == "error" else None)
    page_b.on("console", lambda m: session_b["errors"].append(m.text) if m.type == "error" else None)
    page_a.on("pageerror", lambda e: session_a["errors"].append(str(e)))
    page_b.on("pageerror", lambda e: session_b["errors"].append(str(e)))

    print("connecting both sessions...")
    page_a.goto(args.base_url)
    page_b.goto(args.base_url)
    page_a.wait_for_function("() => document.getElementById('connStatus').textContent === 'connected'")
    page_b.wait_for_function("() => document.getElementById('connStatus').textContent === 'connected'")

    print("uploading different photos to each session...")
    page_a.set_input_files("#fileInput", IMAGE_A)
    page_b.set_input_files("#fileInput", IMAGE_B)
    page_a.wait_for_function(
        "() => document.getElementById('footerStatus').textContent.includes('ready to generate')", timeout=90_000)
    page_b.wait_for_function(
        "() => document.getElementById('footerStatus').textContent.includes('ready to generate')", timeout=90_000)

    prompt_a = "session A: neon-lit cyberpunk alley, rain-slicked street, purple and cyan lighting"
    prompt_b = "session B: bright minimalist art gallery, white walls, natural skylight"

    print("firing both generations close together (not sequentially)...")
    page_a.fill("#scenePrompt", prompt_a)
    page_b.fill("#scenePrompt", prompt_b)
    page_a.click("#btnGenerate")
    page_b.click("#btnGenerate")  # fired immediately after A, while A is still queued/running

    print("waiting for both to complete...")
    for page, label in ((page_a, "A"), (page_b, "B")):
        page.wait_for_function("() => document.getElementById('progressText').textContent !== 'done'", timeout=15_000)
        page.wait_for_function("() => document.getElementById('progressText').textContent === 'done'", timeout=180_000)
        print(f"  session {label} done")

    session_a["provenance"] = page_a.text_content("#provenanceCard")
    session_b["provenance"] = page_b.text_content("#provenanceCard")

    print("\n--- session A provenance ---")
    print(session_a["provenance"])
    print("\n--- session B provenance ---")
    print(session_b["provenance"])

    a_has_a_prompt = prompt_a in session_a["provenance"]
    a_has_b_prompt = prompt_b in session_a["provenance"]
    b_has_b_prompt = prompt_b in session_b["provenance"]
    b_has_a_prompt = prompt_a in session_b["provenance"]

    print(f"\nsession A got its own prompt: {a_has_a_prompt} | got B's prompt (should be False): {a_has_b_prompt}")
    print(f"session B got its own prompt: {b_has_b_prompt} | got A's prompt (should be False): {b_has_a_prompt}")

    assert a_has_a_prompt and not a_has_b_prompt, "session A received the wrong result -- cross-session leak!"
    assert b_has_b_prompt and not b_has_a_prompt, "session B received the wrong result -- cross-session leak!"

    all_errors = session_a["errors"] + session_b["errors"]
    print(f"\nconsole errors: {all_errors}")
    assert not all_errors, f"unexpected console errors: {all_errors}"

    print("\nMULTI-SESSION ISOLATION TEST PASSED -- no cross-talk between concurrent sessions")
    browser.close()
