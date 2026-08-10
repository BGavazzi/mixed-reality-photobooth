"""
Tests for the shared Spout frame buffer.

No Spout involved: SpoutGL is Windows-only and needs a real GL context, so
`spout_sender_loop` isn't exercised here. What is exercised is the geometry
and buffer handling around it -- the part that decides whether a non-square
composite reaches an LED wall correctly proportioned or visibly squashed.
"""

import threading

import pytest
from PIL import Image

from spout_output import DEFAULT_PLACEHOLDER_RGBA, SpoutFrameBuffer, _cover_fit, spout_sender_loop


@pytest.mark.parametrize("source_size", [(1080, 1920), (1920, 1080), (768, 768), (100, 3000)])
def test_cover_fit_always_produces_exactly_the_sender_s_dimensions(source_size):
    assert _cover_fit(Image.new("RGB", source_size), 512, 512).size == (512, 512)


def test_cover_fit_crops_rather_than_stretching():
    """A naive resize() squashes a portrait photo into a square sender --
    the same body-proportion distortion that was fixed on the browser canvas.
    Filling by crop is what any normal video source does.

    Checked by construction: a portrait frame with a centred horizontal
    stripe must come back with that stripe still horizontal and still
    spanning the full width, which a stretch to square would not preserve
    in proportion.
    """
    source = Image.new("RGB", (400, 1200), (0, 0, 0))
    for y in range(575, 625):  # a 50px stripe across the vertical centre
        for x in range(400):
            source.putpixel((x, y), (255, 0, 0))

    fitted = _cover_fit(source, 200, 200)
    scale = max(200 / 400, 200 / 1200)  # 0.5, driven by the width
    expected_stripe_height = round(50 * scale)

    centre_column = [fitted.getpixel((100, y))[0] for y in range(200)]
    assert sum(1 for v in centre_column if v > 200) == pytest.approx(expected_stripe_height, abs=2)
    assert fitted.getpixel((0, 100))[0] > 200, "the stripe should still span the full width"


def test_frame_buffer_starts_on_the_placeholder_colour():
    buffer = SpoutFrameBuffer(64, 32)
    data = buffer.get_bytes()
    assert len(data) == 64 * 32 * 4
    assert data[:4] == bytes(DEFAULT_PLACEHOLDER_RGBA)


@pytest.mark.parametrize("image_size", [(64, 32), (128, 128), (33, 17)])
def test_set_image_always_yields_a_correctly_sized_rgba_buffer(image_size):
    """The Spout sender reads a fixed byte count every tick; a buffer of the
    wrong length is a garbled or crashing output, not a resize."""
    buffer = SpoutFrameBuffer(64, 32)
    buffer.set_image(Image.new("RGB", image_size, (200, 100, 50)))
    assert len(buffer.get_bytes()) == 64 * 32 * 4


def test_set_image_converts_modes_that_arrive_from_the_browser():
    """The photo booth sends a flattened RGBA composite; the bridge sends
    RGBA video frames; generate_image() returns RGBA. Palette and grayscale
    images still have to survive the trip."""
    buffer = SpoutFrameBuffer(16, 16)
    for mode in ("RGB", "RGBA", "L", "P"):
        buffer.set_image(Image.new(mode, (16, 16)))
        assert len(buffer.get_bytes()) == 16 * 16 * 4


def test_busy_lock_implements_the_one_generation_at_a_time_pattern():
    buffer = SpoutFrameBuffer(8, 8)
    assert buffer.busy.acquire(blocking=False) is True
    assert buffer.busy.acquire(blocking=False) is False, "a second trigger must be refused, not queued"
    buffer.busy.release()
    assert buffer.busy.acquire(blocking=False) is True


def test_sender_loop_reports_and_returns_when_spout_is_unavailable(monkeypatch, capsys):
    """On a machine without SpoutGL the sender thread used to die on the
    import with a traceback lost in startup noise, leaving the app looking
    healthy while publishing nothing. It must now say so and leave
    live_event clear, which is what stops the UI claiming a successful send.
    """
    import builtins
    real_import = builtins.__import__

    def no_spoutgl(name, *args, **kwargs):
        if name == "SpoutGL":
            raise ImportError("No module named 'SpoutGL'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_spoutgl)

    live = threading.Event()
    spout_sender_loop(SpoutFrameBuffer(8, 8), "TestSender", 15, threading.Event(), live)

    assert not live.is_set(), "nothing is publishing, so the caller must not be told otherwise"
    assert "DISABLED" in capsys.readouterr().out
