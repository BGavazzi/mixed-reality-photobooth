"""
Tests for POST /api/analyze's request handling.

The three CV models are stubbed out -- they're slow, they download weights,
and their wrappers are covered separately. What's under test here is
everything around them: what the endpoint accepts, what it rejects, how it
rejects it, the resolution cap, and the EXIF handling that decides whether
the models see the subject upright or on their side.

TestClient is used without `with`, deliberately: entering it as a context
manager runs the app's lifespan, which warms up rembg/MiDaS and opens a
websocket to ComfyUI. Neither belongs in a unit test, and skipping the
lifespan is what keeps this file runnable on a clean clone.
"""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import photoshoot_pipeline
import web_server


@pytest.fixture
def client(monkeypatch):
    """Returns (TestClient, seen) where `seen["image"]` is whatever image the
    pipeline was actually handed -- which is how the EXIF and downscaling
    assertions check what the *models* would have received, not just what
    the response reported."""
    seen = {}

    def fake_analyze(image):
        # cap_resolution is policy, not a model, so it stays real -- stubbing
        # it would hide the thing the downscale test is checking.
        image = photoshoot_pipeline.cap_resolution(image)
        seen["image"] = image
        mask = Image.new("L", image.size, 255)
        return {
            "image": image,
            "cutout": image.convert("RGBA"),
            "mask": mask,
            "pose": image.convert("RGBA"),
            "depth": image,
            "shadow": Image.new("RGBA", image.size),
            "illumination": photoshoot_pipeline.estimate_illumination(image, mask),
            "suggested_controlnet_strength": 0.75,
        }

    monkeypatch.setattr(web_server.photoshoot_pipeline, "analyze", fake_analyze)
    return TestClient(web_server.app), seen


def upload(data, filename="photo.png", content_type="image/png"):
    return {"file": (filename, io.BytesIO(data), content_type)}


def png_bytes(size=(64, 48), colour=(120, 90, 60)):
    buf = io.BytesIO()
    Image.new("RGB", size, colour).save(buf, format="PNG")
    return buf.getvalue()


def test_valid_upload_returns_every_layer(client):
    test_client, _ = client
    response = test_client.post("/api/analyze", files=upload(png_bytes()))

    assert response.status_code == 200
    body = response.json()
    for key in ("original", "cutout", "mask", "pose", "depth", "shadow"):
        assert body[key], f"{key} layer is missing from the response"
    assert (body["width"], body["height"]) == (64, 48)
    assert body["illumination"]["descriptor"]
    assert body["suggested_controlnet_strength"] == 0.75


def test_non_image_upload_is_a_400_not_a_500(client):
    """This used to raise inside Image.open and surface as a 500 with a
    traceback; the browser's only clue was "analysis failed"."""
    test_client, _ = client
    response = test_client.post("/api/analyze", files=upload(b"this is not a PNG"))

    assert response.status_code == 400
    assert "image" in response.json()["detail"].lower()


def test_empty_upload_is_rejected(client):
    test_client, _ = client
    assert test_client.post("/api/analyze", files=upload(b"")).status_code == 400


def test_oversized_upload_is_rejected_before_decoding(client, monkeypatch):
    """The size check has to run before Image.open: a decompression-bomb PNG
    can exhaust memory during the decode itself, so rejecting afterwards is
    too late to help."""
    test_client, seen = client
    monkeypatch.setattr(web_server, "MAX_UPLOAD_BYTES", 1024)

    response = test_client.post("/api/analyze", files=upload(png_bytes((400, 400))))

    assert response.status_code == 413
    assert "limit" in response.json()["detail"]
    assert "image" not in seen, "the pipeline must not have been reached"


def test_exif_orientation_is_applied_before_the_models_see_the_photo(client):
    """Phone cameras store a portrait shot as landscape pixels plus a
    rotation flag. The browser honours that flag when displaying the file,
    so without exif_transpose the user saw an upright photo whose pose,
    depth and cutout layers were all rotated 90 degrees."""
    test_client, seen = client

    exif = Image.Exif()
    exif[274] = 6  # Orientation tag: rotate 90 CW
    buf = io.BytesIO()
    Image.new("RGB", (100, 50), (200, 120, 60)).save(buf, format="JPEG", exif=exif)

    response = test_client.post(
        "/api/analyze", files=upload(buf.getvalue(), "phone.jpg", "image/jpeg"))

    assert response.status_code == 200
    assert seen["image"].size == (50, 100), "the pipeline should receive the upright orientation"
    assert (response.json()["width"], response.json()["height"]) == (50, 100)


def test_oversized_photos_are_downscaled_before_the_models_see_them(client):
    """The 22MP-phone-photo case from the README. The endpoint must report
    the *capped* dimensions, since the browser treats them as the canonical
    original for every later generation call."""
    test_client, seen = client
    cap = photoshoot_pipeline.MAX_INPUT_DIMENSION

    response = test_client.post("/api/analyze", files=upload(png_bytes((cap * 2, cap))))

    assert max(seen["image"].size) == cap
    assert max(response.json()["width"], response.json()["height"]) == cap


def test_index_route_serves_the_app_regardless_of_working_directory(tmp_path, monkeypatch):
    """The route used to serve the literal relative path "web/index.html",
    so the whole app 404'd when started from anywhere but the repo root."""
    monkeypatch.chdir(tmp_path)
    response = TestClient(web_server.app).get("/")

    assert response.status_code == 200
    assert "Mixed-Reality Photo Booth" in response.text
