"""
Unit tests for the pure-CV half of the photo booth pipeline.

Everything here runs without rembg, OpenPose, MiDaS, ComfyUI or a GPU:
these are the functions that decide *how* the models' output gets used
(which mask blobs survive, what lighting descriptor reaches the prompt, what
ControlNet strength gets suggested, what resolution the models even see),
and they were previously verified only by looking at the result in a browser.
"""

import numpy as np
import pytest
from PIL import Image, ImageDraw

import photoshoot_pipeline as pipeline


# --- cap_resolution ---------------------------------------------------------

def test_cap_resolution_leaves_small_images_untouched():
    image = Image.new("RGB", (800, 600))
    assert pipeline.cap_resolution(image) is image, "should not copy when no downscale is needed"


def test_cap_resolution_downscales_oversized_photos_preserving_aspect():
    # The 22MP phone photo from the README that drove ComfyUI into its tiled
    # VAE fallback and 41s/step sampling.
    capped = pipeline.cap_resolution(Image.new("RGB", (3875, 5812)))
    assert max(capped.size) == pipeline.MAX_INPUT_DIMENSION
    original_aspect = 3875 / 5812
    assert capped.width / capped.height == pytest.approx(original_aspect, abs=0.01)


def test_cap_resolution_does_not_mutate_the_caller_s_image():
    image = Image.new("RGB", (4000, 3000))
    pipeline.cap_resolution(image)
    assert image.size == (4000, 3000), "thumbnail() is in-place; the original must be copied first"


# --- _drop_tiny_disconnected_blobs ------------------------------------------

def _mask_with_blobs(*boxes):
    mask = Image.new("L", (200, 200), 0)
    draw = ImageDraw.Draw(mask)
    for box in boxes:
        draw.rectangle(box, fill=255)
    return mask


def test_single_blob_mask_is_returned_unchanged():
    mask = _mask_with_blobs((50, 50, 150, 150))
    assert pipeline._drop_tiny_disconnected_blobs(mask) is mask


def test_tiny_disconnected_speck_is_dropped():
    subject = (50, 50, 150, 150)          # 100x100 = 10,000 px
    speck = (five := 5, 5, five + 8, 13)  # ~9x9 = 81 px, well under 3% of 10,000
    cleaned = np.asarray(pipeline._drop_tiny_disconnected_blobs(_mask_with_blobs(subject, speck)))
    assert cleaned[100, 100] == 255, "the main subject must survive"
    assert cleaned[9, 9] == 0, "the disconnected speck should have been zeroed"


def test_large_disconnected_region_is_kept():
    """The threshold is deliberately not 'keep only the largest blob' -- a
    held bag, a dangling earring or a hair strand separated by anti-aliasing
    is a real part of the subject, and amputating it is a worse failure than
    keeping the occasional artifact."""
    subject = (50, 50, 150, 150)      # 10,000 px
    held_object = (10, 90, 40, 130)   # ~1,200 px = 12%, above the 3% threshold
    cleaned = np.asarray(pipeline._drop_tiny_disconnected_blobs(_mask_with_blobs(subject, held_object)))
    assert cleaned[110, 25] == 255, "a substantial disconnected region must be kept"


# --- estimate_illumination --------------------------------------------------

def test_illumination_detects_key_light_direction(lit_subject):
    image, mask = lit_subject
    result = pipeline.estimate_illumination(image, mask)
    assert result.direction == "upper-left"
    assert "upper-left" in result.descriptor
    assert result.softness in ("soft, diffuse", "hard, directional")


def test_illumination_reports_front_lighting_when_evenly_lit():
    image = Image.new("RGB", (128, 128), (20, 20, 20))
    ImageDraw.Draw(image).ellipse([30, 20, 100, 110], fill=(160, 160, 160))
    mask = Image.new("L", (128, 128), 0)
    ImageDraw.Draw(mask).ellipse([30, 20, 100, 110], fill=255)

    result = pipeline.estimate_illumination(image, mask)
    assert result.direction == "front", "a flat-lit subject has no dominant quadrant"
    assert result.descriptor.endswith("shadows")


@pytest.mark.parametrize("fill,expected", [
    ((220, 170, 120), "warm"),
    ((150, 150, 150), "neutral"),
    ((120, 170, 220), "cool"),
])
def test_illumination_reads_colour_temperature(fill, expected):
    image = Image.new("RGB", (128, 128), fill)
    mask = Image.new("L", (128, 128), 255)
    assert pipeline.estimate_illumination(image, mask).warmth == expected


def test_illumination_falls_back_to_the_whole_frame_when_the_mask_is_empty():
    """An empty mask means rembg found no subject at all. Analysing zero
    pixels would raise on ys.min(); the fallback keeps a failed rotoscope
    from taking the whole /api/analyze request down with it."""
    image = Image.new("RGB", (128, 128), (200, 100, 60))
    result = pipeline.estimate_illumination(image, Image.new("L", (128, 128), 0))
    assert result.warmth == "warm"
    assert isinstance(result.to_dict(), dict)


# --- suggest_controlnet_strength -------------------------------------------

def test_plain_backdrop_keeps_the_high_default_strength():
    flat_depth = Image.new("RGB", (128, 128), (128, 128, 128))
    subject_mask = Image.new("L", (128, 128), 0)
    ImageDraw.Draw(subject_mask).rectangle([40, 20, 90, 120], fill=255)

    strength = pipeline.suggest_controlnet_strength(flat_depth, subject_mask)
    assert strength == pipeline.PLAIN_BACKGROUND_STRENGTH


def test_structured_background_suggests_a_lower_strength():
    """The README's leather-armchair case: real depth variance behind the
    subject means a high ControlNet strength re-textures the existing room
    instead of replacing it."""
    depth = Image.new("RGB", (128, 128), (0, 0, 0))
    draw = ImageDraw.Draw(depth)
    for i in range(0, 128, 16):  # alternating near/far bands = high variance
        draw.rectangle([i, 0, i + 8, 128], fill=(255, 255, 255))
    subject_mask = Image.new("L", (128, 128), 0)
    ImageDraw.Draw(subject_mask).rectangle([50, 20, 78, 120], fill=255)

    strength = pipeline.suggest_controlnet_strength(depth, subject_mask)
    assert strength == pipeline.BUSY_BACKGROUND_STRENGTH
    assert strength < pipeline.PLAIN_BACKGROUND_STRENGTH


def test_fully_masked_frame_falls_back_to_the_default_strength():
    """No background pixels at all (a subject filling the frame) must not
    divide by zero or return a nonsense strength."""
    depth = Image.new("RGB", (128, 128), (100, 100, 100))
    assert pipeline.suggest_controlnet_strength(depth, Image.new("L", (128, 128), 255)) == \
        pipeline.PLAIN_BACKGROUND_STRENGTH


# --- generate_contact_shadow ------------------------------------------------

def test_no_detected_feet_yields_a_fully_transparent_layer():
    """A seated or cropped subject has no ankle keypoints. The layer must
    still exist and composite invisibly rather than painting a shadow under
    a foot that was never found."""
    layer = pipeline.generate_contact_shadow((256, 256), [])
    assert layer.size == (256, 256)
    assert np.asarray(layer)[..., 3].max() == 0


def test_shadow_is_drawn_below_each_detected_foot():
    layer = pipeline.generate_contact_shadow((256, 256), [(64.0, 100.0), (192.0, 100.0)])
    alpha = np.asarray(layer)[..., 3]

    assert alpha[:90, :].max() == 0, "nothing should be painted well above the feet"
    for x in (64, 192):
        column = alpha[95:120, x - 5:x + 5]
        assert column.max() > 0, f"expected a shadow under the foot at x={x}"
    assert alpha[104, 128] < alpha[104, 64], "the gap between the two feet should be lighter"
