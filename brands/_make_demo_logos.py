"""
Generates the placeholder logo artwork for the two demo brand kits.

The kits ship with generated marks rather than borrowed ones on purpose:
`aurora` and `northwind` are invented brands, and a demo that quietly bundles
a real company's trademark is not a demo anyone can safely fork. Everything
here is drawn from primitives, so the PNGs in this repo have a visible origin.

Replacing them for real work is a file swap, not a code change: drop your
client's transparent PNG in beside their brand.json and point `logo.file` at
it. Run this only if you want the placeholders regenerated.

    python brands/_make_demo_logos.py
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BRANDS_DIR = Path(__file__).parent
WIDTH, HEIGHT = 900, 260


def _font(size: int):
    """Best available font, degrading to Pillow's bitmap default.

    The default font is deliberately the last resort and not the first
    choice: at logo size it renders as a low-resolution blur, which would
    make the min-width rule in the brand kit look like it was failing when it
    is actually working.
    """
    for name in ("segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def make_logo(path: Path, text: str, mark_color: str, text_color: str, style: str):
    image = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    if style == "arc":
        # Aurora: a rising arc, drawn as a stroked partial ellipse.
        draw.arc([20, 60, 230, 270], start=180, end=360, fill=mark_color, width=26)
        draw.ellipse([112, 96, 148, 132], fill=mark_color)
    else:
        # Northwind: a compass-needle triangle inside a ring.
        draw.ellipse([20, 40, 220, 240], outline=mark_color, width=16)
        draw.polygon([(120, 70), (168, 190), (120, 160), (72, 190)], fill=mark_color)

    font = _font(96)
    draw.text((260, HEIGHT // 2), text, font=font, fill=text_color, anchor="lm")

    # Trim the transparent margin the fixed canvas leaves around a shorter
    # wordmark. Without this the two marks have different amounts of invisible
    # padding, and the brand kit's min_width_pct -- which measures the *file*,
    # not the ink -- would mean something different for each of them.
    bbox = image.getbbox()
    if bbox:
        image = image.crop(bbox)

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    print(f"wrote {path} ({image.width}x{image.height})")


if __name__ == "__main__":
    make_logo(BRANDS_DIR / "aurora" / "logo.png", "AURORA", "#00B5AD", "#0B3C49", "arc")
    make_logo(BRANDS_DIR / "northwind" / "logo.png", "NORTHWIND", "#C2542D", "#2F4A3C", "compass")
