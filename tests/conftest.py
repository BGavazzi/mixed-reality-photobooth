import asyncio
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

# The project is a flat set of top-level modules rather than an installed
# package, so tests import them the same way the entry points do.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def run_queued(coro_factory):
    """Runs a coroutine with the server's generation queue started, then
    waits for everything it enqueued to be submitted.

    Generation submission moved behind a worker pool (job_queue.py), so a
    test that calls a handler and immediately asserts on the backend is
    asserting before the worker has run. This is the seam's cost, paid once
    here rather than as a sleep in every test.

    `coro_factory` is a zero-arg callable rather than a coroutine because the
    coroutine has to be created inside the loop that will run it.
    """
    import web_server

    async def main():
        queue = web_server.GENERATION_QUEUE
        await queue.start()
        try:
            result = await coro_factory()
            await queue.drain()
            return result
        finally:
            await queue.stop()

    return asyncio.run(main())


@pytest.fixture
def lit_subject():
    """A synthetic 'photo': a bright figure on a dark background, lit
    distinctly from the upper left.

    Synthetic on purpose. The illumination and mask heuristics are pure
    arithmetic over pixels, so a generated image with a *known* correct
    answer tests them far more precisely than a real photo whose true light
    direction is a matter of opinion -- and it keeps the suite runnable on a
    clean clone, with no gitignored test_images/ to supply.
    """
    w = h = 128
    image = Image.new("RGB", (w, h), (18, 18, 22))
    draw = ImageDraw.Draw(image)
    draw.ellipse([32, 16, 96, 112], fill=(150, 150, 150))
    # Brighten the upper-left of the figure: this is the signal
    # estimate_illumination() is meant to pick up.
    for y in range(h):
        for x in range(w):
            r, g, b = image.getpixel((x, y))
            if r > 100 and x < 64 and y < 64:
                image.putpixel((x, y), (240, 235, 220))
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).ellipse([32, 16, 96, 112], fill=255)
    return image, mask
