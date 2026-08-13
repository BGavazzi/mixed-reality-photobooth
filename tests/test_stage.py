"""
The claim under test is a single sentence: *the background is no longer at
infinity, and the subject is untouched*. Everything here is a way of checking
one half of that without trusting the other.

These are deliberately property-style rather than pixel-exact. The stage is a
prior, not a render -- asserting that row 412 is 0.61 would freeze a number
that exists to be tuned by eye, and the next person to improve the look would
have to break a test to do it. What must not change is the structure: a floor
that gets nearer downward, a backdrop that is not zero, and a subject that
comes out the far side identical.
"""

import numpy as np
import pytest
from PIL import Image

import stage


def standing_subject(size=(96, 128), head=0.2, feet=0.9) -> Image.Image:
    """A mask shaped like somebody standing: a vertical bar down the middle."""
    width, height = size
    mask = Image.new("L", size, 0)
    arr = np.asarray(mask).copy()
    arr[int(head * height):int(feet * height), width // 3:2 * width // 3] = 255
    return Image.fromarray(arr, mode="L")


def void_depth(size=(96, 128)) -> Image.Image:
    """What the pipeline used to hand ControlNet: a subject on pure black."""
    mask = standing_subject(size)
    arr = (np.asarray(mask, dtype=np.float32) / 255.0 * 200).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def as_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L"), dtype=np.float32) / 255.0


# --- the lookup ------------------------------------------------------------

def test_an_unknown_stage_names_the_valid_ones():
    """A brand kit with a typo'd stage must fail loudly. Silently falling back
    to the default would look exactly like the prior not working."""
    with pytest.raises(stage.StageError) as exc:
        stage.get("rooftop")  # a *look* id, the likeliest confusion
    assert "terrace" in str(exc.value)


def test_no_stage_is_the_default_rather_than_an_error():
    assert stage.get(None).id == stage.DEFAULT_STAGE
    assert stage.get("").id == stage.DEFAULT_STAGE


# --- the horizon -----------------------------------------------------------

def test_the_horizon_follows_the_subject_not_the_frame():
    """A seated guest and a standing one photographed from the same camera do
    not share an eye level. A fixed horizon would fight one of them."""
    tall = standing_subject(head=0.10, feet=0.95)
    short = standing_subject(head=0.55, feet=0.95)
    terrace = stage.get("terrace")

    assert stage.horizon_for(tall, terrace) < stage.horizon_for(short, terrace)


def test_an_empty_mask_still_produces_a_horizon():
    """Segmentation can return nothing -- an empty frame, a failed matte. That
    should still generate a usable stage rather than divide by a missing
    subject."""
    empty = Image.new("L", (96, 128), 0)
    assert 0.0 < stage.horizon_for(empty, stage.get("terrace")) < 1.0


# --- the depth map itself --------------------------------------------------

def test_the_subject_depth_survives_untouched():
    """The whole promise of the compositing approach is that the guest is not
    regenerated. If the stage leaked into the subject's own depth, the model
    would be reshaping them."""
    depth, mask = void_depth(), standing_subject()
    staged = as_array(stage.build_depth(depth, mask, "room"))
    original = as_array(depth)

    inside = np.asarray(mask, dtype=np.float32) / 255.0 > 0.99
    assert np.abs(staged[inside] - original[inside]).max() < 0.02


def test_the_background_stops_being_infinitely_far_away():
    """The bug in one assertion: with the old void, everything outside the
    subject was 0.0 -- 'infinitely distant' -- so no wall, floor or object
    could attach to anything."""
    depth, mask = void_depth(), standing_subject()
    outside = np.asarray(mask, dtype=np.float32) / 255.0 < 0.01

    before = as_array(depth)[outside]
    after = as_array(stage.build_depth(depth, mask, "terrace"))[outside]

    assert before.max() < 0.01, "fixture is not the void this test is about"
    assert after.min() > 0.02, "somewhere in the background is still at infinity"


def test_the_floor_recedes_upward():
    """A ground plane means near at the bottom edge, far at the horizon. Get
    the sign wrong and the model is told the floor is a ceiling."""
    staged = as_array(stage.build_depth(void_depth(), standing_subject(), "terrace"))
    column = staged[:, 2]  # a strip clear of the subject

    bottom = column[-8:].mean()
    middle = column[len(column) // 2 - 4:len(column) // 2 + 4].mean()
    assert bottom > middle, "the bottom of the frame should be the nearest floor"


def test_side_walls_pull_the_frame_edges_nearer_than_the_centre():
    """The cue that separates an interior from an open field."""
    staged = as_array(stage.build_depth(void_depth(), standing_subject(), "room"))
    row = staged[8]  # above the subject's head, so this is pure background

    assert row[0] > row[len(row) // 2], "an interior's side walls are closer than its back wall"


def test_landscape_has_no_side_walls():
    staged = as_array(stage.build_depth(void_depth(), standing_subject(), "landscape"))
    row = staged[8]
    assert abs(float(row[0]) - float(row[len(row) // 2])) < 0.05


def test_a_terrace_puts_distance_above_its_parapet():
    """The bug this fixes was visible on the GPU, not in a test: one flat plane
    above the horizon is a complete description of a wall and a useless one for
    a city behind a wall, so a 'skyline softly out of focus' prompt came back
    with a clean wall and an empty sky. The step at the parapet's top is what
    gives a skyline somewhere to be."""
    staged = as_array(stage.build_depth(void_depth(), standing_subject(), "terrace"))
    column = staged[:, 2]
    horizon = stage.horizon_for(standing_subject(), stage.get("terrace"))
    horizon_px = int(horizon * len(column))

    parapet = column[max(0, horizon_px - 8):horizon_px].mean()
    sky = column[:8].mean()
    assert parapet > sky + 0.05, "there is no depth step for a skyline to sit behind"


def test_an_interior_has_no_sky_band():
    """A room's back wall runs to the top of the frame. Punching a hole of sky
    through the ceiling would be a worse prior than the flat one it replaced."""
    staged = as_array(stage.build_depth(void_depth(), standing_subject(), "room"))
    column = staged[:, 2]
    horizon_px = int(stage.horizon_for(standing_subject(), stage.get("room")) * len(column))

    assert abs(column[:8].mean() - column[max(0, horizon_px - 8):horizon_px].mean()) < 0.05


def test_the_prior_is_smooth_enough_to_be_obeyed_rather_than_traced():
    """A high-frequency fake shows through in the output as banding or a hard
    seam. Neighbouring background rows should differ gently."""
    staged = as_array(stage.build_depth(void_depth(), standing_subject(), "terrace"))
    column = staged[:, 2]
    assert np.abs(np.diff(column)).max() < 0.06


def test_void_is_a_real_escape_hatch():
    """Kept so the previous behaviour is reproducible -- if the prior ever
    makes a particular look worse, an operator needs a way back that is not a
    code change."""
    depth, mask = void_depth(), standing_subject()
    staged = stage.build_depth(depth, mask, "void")
    assert np.array_equal(np.asarray(staged), np.asarray(depth.convert("L")))


def test_a_mask_at_a_different_size_is_matched_to_the_depth():
    """Depth and matte come from two different models; nothing guarantees they
    agree on resolution, and a broadcast error here would fail the item."""
    depth = void_depth((96, 128))
    mask = standing_subject((48, 64))
    assert stage.build_depth(depth, mask, "terrace").size == (96, 128)


# --- the measurement -------------------------------------------------------

def test_relief_separates_a_built_stage_from_the_void():
    """This is the number that goes in the manifest, so it has to actually
    distinguish the two cases it claims to."""
    depth, mask = void_depth(), standing_subject()

    assert stage.background_relief(depth, mask) < 0.05
    assert stage.background_relief(stage.build_depth(depth, mask, "terrace"), mask) > 0.1


def test_relief_of_a_full_frame_subject_is_zero_not_an_error():
    """A close-up portrait can fill the frame. No background is a legitimate
    measurement, not a crash."""
    full = Image.new("L", (96, 128), 255)
    assert stage.background_relief(void_depth(), full) == 0.0
