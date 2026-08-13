"""
Turning a generated plate into a frame somebody would hand to a client.

There were two compositors in this repo and only one of them was finished. The
browser canvas draws a contact shadow, lets the operator place the brand logo,
and shows the result. The *batch* path -- the one that produces the fifty-photo
set a client actually receives -- did this:

    canvas.alpha_composite(cutout)

Subject over background, nothing else. So the deliverable had no logo (the mark
lived only as a browser layer), no contact shadow (the pipeline computes one
and it was discarded), and no relationship between how the subject was lit and
how the scene was lit. The demo was branded and grounded; the product was not.

Three steps, in the order a retoucher would do them:

**Shadow first, under the subject.** The pipeline already generates a soft
ellipse under each detected foot -- deterministic, not AI, and the single
cheapest fix for "the person is floating". It has to go down before the subject
or it draws on top of their shoes.

**Grade the subject toward the plate.** A studio-lit person dropped into a
golden-hour terrace reads as a sticker no matter how good the matte is, because
the two halves of the picture disagree about what colour the light is. This
measures the plate's colour cast and pulls the subject a *fraction* of the way
toward it. A fraction on purpose: pull the whole way and you have regenerated
the person's skin tone, which is the one thing this tool promises never to do.

**Logo last, on top of everything.** Using the same geometry as the browser
(`logoGeometry` in web/index.html), so a frame from batch mode and a frame from
the interactive app place the mark identically. Aspect comes from the file and
never from a caller -- a stretched logo is the most common brand violation
there is, and the way to not commit it is to make it unrepresentable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

# How far the subject is allowed to move toward the plate's colour. Tuned by
# eye against real photographs: 0.35 removes the "cut out of a different
# picture" tell, while leaving skin recognisably the skin in the original
# photograph. Higher starts to look like the subject was regenerated -- which
# it must never be, and which at 1.0 it would visibly appear to have been.
DEFAULT_GRADE_STRENGTH = 0.35

# Per-channel gain is clamped as well, because a strongly coloured plate (a
# bioluminescent jungle, a red-lit bar) would otherwise tint a face green.
MAX_CHANNEL_GAIN = 1.35
MIN_CHANNEL_GAIN = 0.74

LOGO_ANCHORS = {
    "bottom-right": (1.0, 1.0), "bottom-left": (0.0, 1.0),
    "top-right": (1.0, 0.0), "top-left": (0.0, 0.0),
}


@dataclass(frozen=True)
class LogoPlacement:
    """Where the mark landed, in pixels. Returned rather than only drawn so it
    can go into the manifest -- "was the logo on this frame, and where" is a
    question a brand reviewer asks about a specific image, months later."""
    x: int
    y: int
    width: int
    height: int
    corner: str
    margin: int

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height,
                "corner": self.corner, "margin": self.margin}


def grade_subject(subject: Image.Image, plate: Image.Image,
                  strength: float = DEFAULT_GRADE_STRENGTH) -> Image.Image:
    """Nudges the subject's colour balance toward the plate's.

    Grey-world, restricted to the pixels that actually matter: the subject's
    own opaque pixels against the plate *outside* the subject. Comparing means
    over the whole frame would be measuring the subject against a copy of
    itself once it is composited.

    Returns a new RGBA image; alpha is carried through untouched, since
    softening a matte is a different operation with different failure modes.
    """
    if strength <= 0:
        return subject

    rgba = np.asarray(subject.convert("RGBA"), dtype=np.float32)
    alpha = rgba[..., 3:4] / 255.0
    opaque = alpha[..., 0] > 0.5
    if not opaque.any():
        return subject

    plate_arr = np.asarray(plate.convert("RGB").resize(subject.size), dtype=np.float32)
    background = ~opaque
    if not background.any():
        return subject

    subject_mean = rgba[..., :3][opaque].mean(axis=0)
    plate_mean = plate_arr[background].mean(axis=0)

    # Ratio of ratios: how the plate's channel balance differs from the
    # subject's, ignoring overall brightness. Matching absolute brightness
    # would blow out a subject shot against a dark plate; what reads as "wrong
    # light" is the *hue* of it.
    subject_norm = subject_mean / max(subject_mean.mean(), 1e-3)
    plate_norm = plate_mean / max(plate_mean.mean(), 1e-3)
    gain = np.clip(plate_norm / np.maximum(subject_norm, 1e-3),
                   MIN_CHANNEL_GAIN, MAX_CHANNEL_GAIN)
    gain = 1.0 + (gain - 1.0) * float(strength)

    graded = np.clip(rgba[..., :3] * gain, 0, 255)
    out = np.concatenate([graded, rgba[..., 3:4]], axis=-1).astype(np.uint8)
    return Image.fromarray(out, mode="RGBA")


def place_logo(canvas: Image.Image, logo: Image.Image, *, min_width_pct: float = 10.0,
               clear_space_pct: float = 4.0, corner: str = "bottom-right",
               scale: float = 1.0) -> LogoPlacement:
    """Draws the mark and reports where it went. Mutates `canvas` in place.

    The geometry is deliberately identical to web/index.html's `logoGeometry`:
    clear space measured against the *short* side (using width for both puts a
    much larger gap top and bottom of a tall frame than at its sides), and
    height derived from the artwork's own aspect so it cannot be stretched.
    """
    margin = min(canvas.width, canvas.height) * (clear_space_pct / 100.0)
    width = canvas.width * (min_width_pct / 100.0) * max(scale, 1.0)
    height = width * (logo.height / logo.width)
    ax, ay = LOGO_ANCHORS.get(corner, LOGO_ANCHORS["bottom-right"])
    x = margin + ax * (canvas.width - 2 * margin - width)
    y = margin + ay * (canvas.height - 2 * margin - height)

    resized = logo.convert("RGBA").resize(
        (max(1, round(width)), max(1, round(height))), Image.LANCZOS)
    canvas.alpha_composite(resized, (round(x), round(y)))
    return LogoPlacement(round(x), round(y), resized.width, resized.height, corner, round(margin))


def brand(frame: Image.Image, logo: Image.Image | None, logo_rules=None,
          corner: str | None = None, min_clear_space_pct: float = 0.0
          ) -> tuple[Image.Image, dict | None]:
    """Puts the mark on a finished frame, and says where it went.

    Separate from finish_frame() because the mark cannot be baked in once and
    reused. Every surface is a different crop of the same generation, and a
    logo composited before the crop is a logo the crop moves, shrinks, or --
    observed on a 6x4 postcard -- removes entirely. So the frame is finished
    unbranded, and each output gets its own placement in its own corner.

    `min_clear_space_pct` is the surface's own safe margin, and it *widens* the
    brand's clear space rather than replacing it. A kit's clear space is a
    minimum distance from the artwork; a surface's safe margin is a region
    where pixels survive but should not be relied on -- a story's bottom band
    sits under phone UI. Honouring only the kit put an 8%-margin story's logo
    at 4%, i.e. underneath the progress bar. Taking the larger satisfies both,
    and a kit can still demand more room than the surface asks for.
    """
    if logo is None:
        return frame, None
    corner = corner or getattr(logo_rules, "default_corner", "bottom-right")
    canvas = frame.convert("RGBA")
    placement = place_logo(
        canvas, logo,
        min_width_pct=getattr(logo_rules, "min_width_pct", 10.0),
        clear_space_pct=max(getattr(logo_rules, "clear_space_pct", 4.0),
                            min_clear_space_pct),
        corner=corner)
    return canvas.convert("RGB"), placement.to_dict()


def finish_frame(plate: Image.Image, cutout: Image.Image, *,
                 shadow: Image.Image | None = None,
                 logo: Image.Image | None = None,
                 logo_rules=None,
                 grade_strength: float = DEFAULT_GRADE_STRENGTH) -> tuple[Image.Image, dict]:
    """Plate + shadow + graded subject + logo, in that order.

    Returns the frame and a small record of what was applied, which goes into
    the run manifest: "this frame has the mark on it" should be checkable
    without opening the picture.
    """
    canvas = plate.convert("RGBA")
    subject = cutout.convert("RGBA")
    if subject.size != canvas.size:
        # The generated plate comes back at the subject photo's own resolution,
        # so a mismatch means something upstream changed. Resize rather than
        # fail a whole batch item, but say so.
        subject = subject.resize(canvas.size, Image.LANCZOS)

    applied = {"shadow": False, "graded": False, "logo": None}

    if shadow is not None:
        layer = shadow.convert("RGBA")
        if layer.size != canvas.size:
            layer = layer.resize(canvas.size, Image.LANCZOS)
        canvas.alpha_composite(layer)
        applied["shadow"] = True

    if grade_strength > 0:
        graded = grade_subject(subject, plate, grade_strength)
        applied["graded"] = graded is not subject
        subject = graded

    canvas.alpha_composite(subject)

    if logo is not None:
        rules = logo_rules
        placement = place_logo(
            canvas, logo,
            min_width_pct=getattr(rules, "min_width_pct", 10.0),
            clear_space_pct=getattr(rules, "clear_space_pct", 4.0),
            corner=getattr(rules, "default_corner", "bottom-right"))
        applied["logo"] = placement.to_dict()

    return canvas.convert("RGB"), applied
