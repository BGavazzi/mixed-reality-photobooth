"""
The batch deliverable used to be `canvas.alpha_composite(cutout)` and nothing
else -- no mark, no shadow, no light matching. These tests exist to make that
regression loud, because it is invisible in every other way: the pipeline still
runs, the zip still arrives, and the frames just quietly look like a demo.

Two things are asserted harder than the rest, because they are brand
violations rather than taste:

  * a logo is never stretched, and
  * the subject's *geometry* is never altered by the grade.

Everything else here is a structural property, not a pixel value -- the grade
strength is meant to be tuned by eye.
"""

import numpy as np
import pytest
from PIL import Image

import finish


def subject(size=(64, 64), colour=(220, 180, 160)) -> Image.Image:
    """An opaque person-ish blob on transparency, with real skin-ish tone so a
    colour grade has something meaningful to move."""
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    arr = np.asarray(img).copy()
    arr[16:56, 20:44] = (*colour, 255)
    return Image.fromarray(arr, mode="RGBA")


def plate(size=(64, 64), colour=(90, 110, 180)) -> Image.Image:
    """A cool-lit background: the classic 'studio person dropped into a blue
    evening' mismatch this module exists to soften."""
    return Image.new("RGB", size, colour)


# --- the grade -------------------------------------------------------------

def test_the_subject_moves_toward_the_plate_colour():
    warm = subject(colour=(230, 170, 120))          # tungsten-lit person
    cool = plate(colour=(80, 110, 200))             # blue-hour background

    graded = np.asarray(finish.grade_subject(warm, cool), dtype=np.float32)
    before = np.asarray(warm, dtype=np.float32)

    opaque = before[..., 3] > 0
    blue_shift = graded[..., 2][opaque].mean() - before[..., 2][opaque].mean()
    red_shift = graded[..., 0][opaque].mean() - before[..., 0][opaque].mean()
    assert blue_shift > 0, "a blue plate should cool the subject"
    assert red_shift < 0, "and pull the tungsten warmth back"


def test_the_grade_is_partial_and_never_repaints_the_subject():
    """The one promise the compositing approach makes is that the guest is not
    regenerated. A full grey-world match would make them a different person's
    skin tone; this must stay recognisably a nudge."""
    warm = subject(colour=(230, 170, 120))
    graded = np.asarray(finish.grade_subject(warm, plate(colour=(20, 240, 20))), dtype=np.float32)
    before = np.asarray(warm, dtype=np.float32)

    opaque = before[..., 3] > 0
    drift = np.abs(graded[..., :3][opaque] - before[..., :3][opaque]).mean()
    assert drift > 1, "a violently green plate should have *some* effect"
    assert drift < 60, "but must not repaint the subject toward the plate's hue"


def test_the_matte_is_carried_through_untouched():
    """Softening an edge is a different operation with different failure
    modes; a grade that quietly eroded alpha would fringe every cutout."""
    person = subject()
    graded = finish.grade_subject(person, plate())
    assert np.array_equal(np.asarray(graded)[..., 3], np.asarray(person)[..., 3])


def test_zero_strength_returns_the_subject_unchanged():
    person = subject()
    assert finish.grade_subject(person, plate(), 0) is person


def test_a_fully_covered_plate_is_not_graded_against_itself():
    """A close-up filling the frame leaves no background pixels to measure.
    Measuring the subject against a copy of itself would be a no-op at best
    and a divide-by-nothing at worst."""
    full = Image.new("RGBA", (32, 32), (200, 150, 120, 255))
    assert finish.grade_subject(full, plate((32, 32))) is full


def test_an_empty_cutout_is_not_graded():
    empty = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    assert finish.grade_subject(empty, plate((32, 32))) is empty


# --- the mark --------------------------------------------------------------

def test_the_logo_keeps_its_own_aspect_ratio():
    """The most common brand violation there is. Callers cannot pass a height,
    so the only way to commit it is a bug in here."""
    canvas = Image.new("RGBA", (800, 400), (0, 0, 0, 255))
    logo = Image.new("RGBA", (300, 100), (255, 255, 255, 255))  # 3:1

    placement = finish.place_logo(canvas, logo)

    assert abs(placement.width / placement.height - 3.0) < 0.05


def test_the_logo_sits_inside_its_clear_space():
    canvas = Image.new("RGBA", (800, 400), (0, 0, 0, 255))
    logo = Image.new("RGBA", (200, 100), (255, 255, 255, 255))

    placement = finish.place_logo(canvas, logo, corner="bottom-right", clear_space_pct=5.0)

    assert placement.x + placement.width <= canvas.width - placement.margin + 1
    assert placement.y + placement.height <= canvas.height - placement.margin + 1


def test_clear_space_is_measured_against_the_short_side():
    """Using width for both puts a much bigger gap top and bottom of a tall
    frame than at its sides -- the browser measures the short side, and a
    batch frame has to match it."""
    tall = Image.new("RGBA", (400, 1200), (0, 0, 0, 255))
    logo = Image.new("RGBA", (100, 100), (255, 255, 255, 255))

    placement = finish.place_logo(tall, logo, clear_space_pct=5.0)

    assert placement.margin == pytest.approx(400 * 0.05, abs=1)


@pytest.mark.parametrize("corner", sorted(finish.LOGO_ANCHORS))
def test_every_corner_places_the_mark_somewhere_on_the_canvas(corner):
    canvas = Image.new("RGBA", (400, 400), (0, 0, 0, 255))
    logo = Image.new("RGBA", (100, 50), (255, 255, 255, 255))

    placement = finish.place_logo(canvas, logo, corner=corner)

    assert 0 <= placement.x and placement.x + placement.width <= canvas.width
    assert 0 <= placement.y and placement.y + placement.height <= canvas.height


def test_an_unknown_corner_falls_back_rather_than_dropping_the_mark():
    """Brand kits are hand-edited JSON. A typo'd corner should still put the
    logo on the frame -- an unbranded deliverable is the worse failure."""
    canvas = Image.new("RGBA", (400, 400), (0, 0, 0, 255))
    placement = finish.place_logo(canvas, Image.new("RGBA", (100, 50), (255,) * 4),
                                  corner="middle-of-nowhere")
    assert placement.width > 0


# --- the whole frame -------------------------------------------------------

def test_a_finished_frame_reports_each_layer_it_applied():
    """This record is what makes 'was the mark on this frame' answerable from
    the manifest, months later, without opening the picture."""
    frame, applied = finish.finish_frame(
        plate(), subject(),
        shadow=Image.new("RGBA", (64, 64), (0, 0, 0, 60)),
        logo=Image.new("RGBA", (40, 20), (255, 255, 255, 255)))

    assert applied["shadow"] is True
    assert applied["graded"] is True
    assert applied["logo"]["width"] > 0
    assert frame.mode == "RGB"


def test_branding_is_a_separate_step_so_each_crop_gets_its_own_mark():
    """A logo composited before a surface crop is a logo the crop can push off
    the picture -- observed on a 6x4 postcard, which arrived unbranded."""
    frame = Image.new("RGB", (400, 400), (0, 0, 0))
    branded, placement = finish.brand(frame, Image.new("RGBA", (100, 50), (255,) * 4),
                                      corner="top-left")

    assert placement["corner"] == "top-left"
    assert branded is not frame, "the source frame should not be mutated"


def test_a_surfaces_safe_margin_widens_the_brands_clear_space():
    """A story's bottom band sits under phone UI. Honouring only the kit's 4%
    put the mark underneath the progress bar on a surface that declares 8%."""
    logo = Image.new("RGBA", (100, 50), (255,) * 4)

    class Rules:
        min_width_pct, clear_space_pct, default_corner = 10.0, 4.0, "bottom-right"

    _, tight = finish.brand(Image.new("RGB", (1000, 1000)), logo, Rules())
    _, safe = finish.brand(Image.new("RGB", (1000, 1000)), logo, Rules(),
                           min_clear_space_pct=8.0)

    assert safe["margin"] > tight["margin"]


def test_a_brand_can_still_demand_more_room_than_the_surface_asks_for():
    """The margin is a floor, not an override -- a kit with generous clear
    space must not be tightened by a surface with a small safe area."""
    class Rules:
        min_width_pct, clear_space_pct, default_corner = 10.0, 12.0, "bottom-right"

    _, placement = finish.brand(Image.new("RGB", (1000, 1000)),
                                Image.new("RGBA", (100, 50), (255,) * 4), Rules(),
                                min_clear_space_pct=5.0)
    assert placement["margin"] == pytest.approx(120, abs=1)


def test_branding_without_a_logo_is_a_no_op():
    """Live surfaces and unbranded runs both take this path; it must not
    invent a placement record for a mark that was never drawn."""
    frame = Image.new("RGB", (400, 400), (0, 0, 0))
    result, placement = finish.brand(frame, None)
    assert result is frame and placement is None


def test_a_frame_with_no_brand_kit_reports_no_logo():
    _, applied = finish.finish_frame(plate(), subject())
    assert applied["logo"] is None
    assert applied["shadow"] is False


def test_the_shadow_goes_under_the_subject_not_over_their_shoes():
    """Order matters and is invisible until you look at a foot. A shadow drawn
    last paints a grey wash across the guest."""
    black_shadow = Image.new("RGBA", (64, 64), (0, 0, 0, 255))
    frame, _ = finish.finish_frame(plate(), subject(colour=(255, 255, 255)),
                                   shadow=black_shadow, grade_strength=0)

    assert frame.getpixel((32, 32)) == (255, 255, 255), "the subject was painted over"
    assert frame.getpixel((1, 1)) == (0, 0, 0), "the shadow should show where the subject isn't"


def test_a_mismatched_cutout_is_resized_rather_than_failing_the_frame():
    """Losing a whole batch item to a size change upstream is a worse answer
    than a resize."""
    frame, _ = finish.finish_frame(plate((64, 64)), subject((16, 16)))
    assert frame.size == (64, 64)


def test_a_mismatched_shadow_is_resized_too():
    frame, applied = finish.finish_frame(
        plate((64, 64)), subject(), shadow=Image.new("RGBA", (16, 16), (0, 0, 0, 60)))
    assert frame.size == (64, 64)
    assert applied["shadow"] is True
