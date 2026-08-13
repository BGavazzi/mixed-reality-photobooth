"""
The failure these prevent is silent and only visible on paper: a guest whose
head is fine on the wall and cropped off on the print, or whose feet float in
mid-air in the 9:16 version. Nothing errors. The frames just come out wrong,
in a way nobody checks until the client does.

So the assertions are about *placement*, not about pixels: cover rather than
stretch, and feet where the surface says feet go.
"""

import numpy as np
import pytest
from PIL import Image

import surfaces


def frame(size=(1000, 1000)) -> Image.Image:
    return Image.new("RGB", size, (128, 128, 128))


def footed_mask(size=(1000, 1000), head=0.2, feet=0.8) -> Image.Image:
    width, height = size
    arr = np.zeros((height, width), dtype=np.uint8)
    arr[int(head * height):int(feet * height), width // 3:2 * width // 3] = 255
    return Image.fromarray(arr, mode="L")


# --- the catalogue ---------------------------------------------------------

def test_an_unknown_surface_lists_the_real_ones():
    with pytest.raises(surfaces.SurfaceError) as exc:
        surfaces.get("instagram")
    assert "story_9x16" in str(exc.value)


def test_no_surfaces_requested_means_none_written():
    """Surfaces are opt-in: rendering six versions of every photo triples a
    fifty-photo zip for an operator who only wanted the frames."""
    assert surfaces.parse_list("") == []
    assert surfaces.parse_list(None) == []


def test_a_list_is_parsed_and_whitespace_forgiven():
    parsed = surfaces.parse_list(" story_9x16 , print_2x6 ")
    assert [s.id for s in parsed] == ["story_9x16", "print_2x6"]


def test_one_bad_id_fails_the_whole_list_rather_than_being_skipped():
    """Quietly dropping it would deliver a zip missing a surface the operator
    asked for, discovered after the event rather than before it."""
    with pytest.raises(surfaces.SurfaceError):
        surfaces.parse_list("story_9x16,typo")


def test_prints_declare_a_dpi_and_screens_do_not():
    """A print's physical size is a real number; a screen's is meaningless."""
    for surface in surfaces.SURFACES.values():
        if surface.kind == surfaces.PRINT:
            assert surface.dpi, f"{surface.id} goes on paper and has no dpi"
        else:
            assert surface.dpi is None, f"{surface.id} is a screen and should not claim a dpi"


def test_the_live_surfaces_carry_no_logo():
    """The mark belongs on the frame the guest keeps, not on the wall the
    guest is standing in front of -- where their body covers it."""
    for surface in surfaces.SURFACES.values():
        if surface.kind == surfaces.LIVE:
            assert surface.logo_corner is None


def test_the_default_live_surface_exists_and_is_live():
    surface = surfaces.get(surfaces.DEFAULT_LIVE_SURFACE)
    assert surface.kind == surfaces.LIVE


# --- reframing -------------------------------------------------------------

@pytest.mark.parametrize("surface_id", sorted(surfaces.SURFACES))
def test_every_surface_reframes_to_exactly_its_declared_size(surface_id):
    surface = surfaces.get(surface_id)
    assert surfaces.reframe(frame(), surface).size == (surface.width, surface.height)


def test_a_source_narrower_than_the_target_is_covered_not_stretched():
    """A person made 12% wider to fit a wall is a worse failure than a person
    with less headroom, and the one nobody notices until it is printed.

    The marker is small and sits at the centre of the crop window on purpose:
    a marker the crop clips is not evidence of anything, and a first attempt
    at this test measured a cropped square and called it a stretched one.
    """
    tall_source = Image.new("RGB", (400, 1200), (0, 0, 0))
    tall_source.paste((255, 0, 0), (190, 590, 210, 610))  # square, mid-frame

    result = surfaces.reframe(tall_source, surfaces.get("ultrawide_backdrop"))
    arr = np.asarray(result)
    rows = np.where(arr[..., 0].max(axis=1) > 200)[0]
    cols = np.where(arr[..., 0].max(axis=0) > 200)[0]

    assert rows.size and cols.size, "the marker was cropped out; the test proves nothing"
    assert abs(len(rows) - len(cols)) / max(len(rows), len(cols)) < 0.1


def test_the_subject_lands_on_the_surfaces_anchor():
    """The field that does the real work. A person centred in a square is
    standing in mid-air once that square becomes a 9:16 story."""
    surface = surfaces.get("story_9x16")
    mask = footed_mask(feet=0.8)
    source = frame()
    source.paste((255, 0, 0), (0, 795, 1000, 800))  # a red line at the footline

    result = surfaces.reframe(source, surface, mask=mask)
    arr = np.asarray(result)
    rows = np.where(arr[..., 0].max(axis=1) > 200)[0]

    assert rows.size, "the footline was cropped out entirely"
    landed = rows.mean() / surface.height
    assert abs(landed - surface.subject_anchor) < 0.06


def test_without_a_mask_it_centres_rather_than_guessing():
    """No mask is a legitimate state -- a frame with no subject in it. Centre
    is the right answer there, and it must not throw."""
    source = frame((1000, 1000))
    source.paste((255, 0, 0), (0, 495, 1000, 505))

    result = surfaces.reframe(source, surfaces.get("frame_4x5"))
    arr = np.asarray(result)
    rows = np.where(arr[..., 0].max(axis=1) > 200)[0]
    assert abs(rows.mean() / result.height - 0.5) < 0.05


def test_a_square_photo_still_reaches_the_story_anchor():
    """The ordinary booth case, and the one cover-fit alone cannot serve: a
    square scaled to 9:16 fills the height exactly, leaving no window to move.
    Without the bounded extra zoom the anchor is a field that does nothing."""
    surface = surfaces.get("story_9x16")
    source = frame((1000, 1000))
    source.paste((255, 0, 0), (0, 795, 1000, 800))

    result = surfaces.reframe(source, surface, mask=footed_mask(feet=0.8))
    rows = np.where(np.asarray(result)[..., 0].max(axis=1) > 200)[0]

    assert rows.size
    assert abs(rows.mean() / surface.height - surface.subject_anchor) < 0.03


def test_the_extra_zoom_is_bounded_rather_than_cropping_a_head_off():
    """A subject high in the frame would need an enormous zoom to bring their
    feet down to the anchor. Landing short is the correct answer; removing
    their head to hit a number is not."""
    surface = surfaces.get("story_9x16")
    high = footed_mask(head=0.02, feet=0.25)
    result = surfaces.reframe(frame(), surface, mask=high)

    assert result.size == (surface.width, surface.height)
    assert np.asarray(result).min() > 0, "a black band means the window ran off the image"


def test_the_crop_window_never_leaves_the_picture():
    """A subject standing at the very top of frame drives the anchor offset
    negative; clamping is what stops that becoming a black band."""
    surface = surfaces.get("print_6x4")
    high = footed_mask(head=0.02, feet=0.25)
    result = surfaces.reframe(frame(), surface, mask=high)

    assert result.size == (surface.width, surface.height)
    assert np.asarray(result).min() > 0, "a black band means the crop ran off the image"


def test_a_subject_too_tall_for_the_surface_keeps_their_head():
    """Found on a real postcard: a standing figure cover-fitted into 6x4 is
    nearly twice as tall as the frame, and obediently anchoring the feet
    delivered a print of two shoes and two knees with the face cropped off.
    When the body cannot fit, the head is the half worth keeping."""
    surface = surfaces.get("print_6x4")            # 1800x1200 landscape
    source = frame((1024, 1536))                   # a full-length portrait
    source.paste((255, 0, 0), (400, 300, 600, 340))  # a marker where the face is

    result = surfaces.reframe(source, surface, mask=footed_mask((1024, 1536), 0.2, 0.95))
    rows = np.where(np.asarray(result)[..., 0].max(axis=1) > 200)[0]

    assert rows.size, "the subject's head was cropped out of the deliverable"
    assert rows.mean() / surface.height < 0.5, "the head should be in the upper half"


def test_a_subject_that_fits_still_gets_their_feet_anchored():
    """The head rule must not swallow the ordinary case."""
    surface = surfaces.get("frame_4x5")
    source = frame((1000, 1000))
    source.paste((255, 0, 0), (0, 795, 1000, 800))

    result = surfaces.reframe(source, surface, mask=footed_mask(feet=0.8))
    rows = np.where(np.asarray(result)[..., 0].max(axis=1) > 200)[0]

    assert rows.size
    assert abs(rows.mean() / surface.height - surface.subject_anchor) < 0.06


def test_an_empty_mask_is_treated_as_no_mask():
    empty = Image.new("L", (1000, 1000), 0)
    assert surfaces.subject_footline(empty) is None
    assert surfaces.subject_footline(None) is None


# --- safe area -------------------------------------------------------------

def test_the_safe_box_is_inside_the_surface_on_every_side():
    for surface in surfaces.SURFACES.values():
        left, top, right, bottom = surfaces.safe_box(surface)
        assert 0 < left < right < surface.width
        assert 0 < top < bottom < surface.height


def test_a_story_reserves_more_margin_than_a_delivered_frame():
    """Phone UI eats the top and bottom bands; a 4:5 in a feed does not have
    a progress bar over it."""
    story = surfaces.get("story_9x16")
    delivered = surfaces.get("frame_4x5")
    assert story.safe_margin_pct > delivered.safe_margin_pct
