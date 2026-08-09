"""
Unit tests for the Resolume resync path's prompt building.

These are the deterministic pixel-stat descriptors the README leans on to
claim resync is *reproducible* ("the same visual content always produces the
same descriptors, regardless of what the clip happens to be named"). That
claim is only worth making if the descriptors actually describe the image.
"""

from PIL import Image, ImageDraw

import resolume_state as rs


def solid(colour, size=(96, 96)):
    return Image.new("RGB", size, colour)


# --- describe_image ---------------------------------------------------------

def test_solid_colour_is_named_correctly():
    assert "red-dominant" in rs.describe_image(solid((255, 40, 40)))
    assert "blue-dominant" in rs.describe_image(solid((40, 60, 230)))
    assert "green-dominant" in rs.describe_image(solid((40, 220, 80)))


def test_two_vivid_colours_do_not_average_into_a_third():
    """The regression this function was rewritten for: averaging RGB before
    reading the hue turned a red/cyan VJ visual into "magenta-dominant,
    muted, desaturated colors" -- a colour present nowhere in the frame, and
    the exact opposite of its real saturation."""
    image = solid((255, 30, 30), (200, 200))
    ImageDraw.Draw(image).rectangle([100, 0, 200, 200], fill=(30, 220, 255))

    description = rs.describe_image(image)
    assert "magenta" not in description
    assert "vivid, saturated colors" in description
    assert "desaturated" not in description
    assert "red-dominant" in description or "cyan-dominant" in description


def test_vivid_accent_on_black_is_not_called_desaturated():
    """Dark frame, small neon element -- the staple of VJ content. A flat
    saturation mean let the black outvote the colour entirely."""
    image = solid((0, 0, 0), (200, 200))
    ImageDraw.Draw(image).rectangle([80, 80, 120, 120], fill=(0, 255, 60))

    description = rs.describe_image(image)
    assert "green-dominant" in description
    assert "vivid, saturated colors" in description
    assert "dark, moody" in description, "the frame overall is still dark"


def test_greyscale_content_is_reported_as_monochrome():
    for grey in ((128, 128, 128), (0, 0, 0), (240, 240, 240)):
        assert "near-monochrome" in rs.describe_image(solid(grey))


def test_brightness_bands():
    assert "bright, luminous" in rs.describe_image(solid((250, 250, 250)))
    assert "dark, moody" in rs.describe_image(solid((12, 12, 14)))


def test_high_contrast_and_busy_texture_are_detected():
    image = Image.new("RGB", (96, 96), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    for x in range(0, 96, 4):  # tight black/white stripes
        draw.rectangle([x, 0, x + 1, 96], fill=(255, 255, 255))

    description = rs.describe_image(image)
    assert "high contrast" in description
    assert "busy, detailed texture" in description


def test_descriptors_are_deterministic():
    """The reproducibility claim in the README, asserted rather than
    assumed: identical pixels must always yield an identical string."""
    image = solid((30, 180, 200))
    assert rs.describe_image(image) == rs.describe_image(image.copy())


def test_describe_image_accepts_non_rgb_modes():
    """Resolume thumbnails arrive as whatever the REST API encodes; a
    palette or grayscale PNG must not blow up the resync path."""
    for mode in ("L", "P", "RGBA"):
        assert rs.describe_image(Image.new(mode, (64, 64)))


# --- hue band naming --------------------------------------------------------

def test_hue_bands_cover_the_full_circle():
    """_hue_name walks HUE_NAMES in order and must return something for
    every degree, including the red band that wraps past 330."""
    assert {rs._hue_name(deg) for deg in range(0, 360)} <= set(name for _, name in rs.HUE_NAMES)
    assert rs._hue_name(0) == "red"
    assert rs._hue_name(355) == "red", "the wrap-around band must still be red"
    assert rs._hue_name(240) == "blue"


# --- describe_composition ---------------------------------------------------

def composition(*layers):
    return {"layers": list(layers)}


def layer(clip_name, connected="Connected", opacity=1.0, effects=(), clip_effects=()):
    return {
        "video": {"opacity": {"value": opacity}, "effects": list(effects)},
        "clips": [{
            "name": {"value": clip_name},
            "connected": {"value": connected},
            "video": {"effects": list(clip_effects)},
        }],
    }


def test_active_clip_name_and_effects_become_prompt_fragments():
    prompt = rs.describe_composition(
        composition(layer("neon jellyfish", clip_effects=[{"name": "Kaleidoscope"}])),
        use_thumbnails=False,
    )
    assert "neon jellyfish" in prompt
    assert rs.EFFECT_DESCRIPTORS["kaleidoscope"] in prompt


def test_unmapped_effects_fall_back_to_their_own_lowercased_name():
    prompt = rs.describe_composition(
        composition(layer("clip", clip_effects=[{"name": "Posterize"}, {"name": "Colorize"}])),
        use_thumbnails=False,
    )
    assert rs.EFFECT_DESCRIPTORS["posterize"] in prompt
    assert "colorize" in prompt


def test_bypassed_effects_are_ignored():
    prompt = rs.describe_composition(
        composition(layer("clip", clip_effects=[{"name": "Glitch", "bypassed": {"value": True}}])),
        use_thumbnails=False,
    )
    assert "glitch" not in prompt.lower()


def test_invisible_layers_are_skipped():
    prompt = rs.describe_composition(
        composition(layer("hidden clip", opacity=0.0), layer("visible clip")),
        use_thumbnails=False,
    )
    assert "hidden clip" not in prompt
    assert "visible clip" in prompt


def test_disconnected_clips_are_skipped():
    prompt = rs.describe_composition(
        composition(layer("idle clip", connected="Disconnected")), use_thumbnails=False)
    assert prompt == ""


def test_previewing_counts_as_active():
    prompt = rs.describe_composition(
        composition(layer("previewed clip", connected="Previewing")), use_thumbnails=False)
    assert "previewed clip" in prompt


def test_duplicate_fragments_are_collapsed():
    """Two layers running the same effect shouldn't repeat its descriptor."""
    prompt = rs.describe_composition(
        composition(layer("a", clip_effects=[{"name": "Mirror"}]),
                    layer("b", clip_effects=[{"name": "Mirror"}])),
        use_thumbnails=False,
    )
    assert prompt.count(rs.EFFECT_DESCRIPTORS["mirror"]) == 1


def test_malformed_composition_does_not_raise():
    """Resolume's REST payload shape varies by version; missing keys and
    non-dict parameter values must degrade to an empty prompt, not a crash
    that takes out the OSC handler."""
    for broken in ({}, {"layers": []}, {"layers": [{}]}, {"layers": [{"clips": [{}]}]}):
        assert rs.describe_composition(broken, use_thumbnails=False) == ""
