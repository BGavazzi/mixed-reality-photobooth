"""
What the booth actually renders onto.

The app has had exactly one output shape, and it was an accident: whatever
aspect ratio the guest's photograph happened to be. Everything downstream
inherited it. The Spout sender -- the one that feeds a real LED wall or
projector through Resolume -- was hardcoded to **768x768**, a square, which is
not the shape of any wall, screen, projector or print in existence.

A booth is not one output. It is several, simultaneously:

  * a **backdrop** the guest is standing in front of, live, at the wall's own
    pixel dimensions;
  * a **frame they take home**, usually cropped for a phone;
  * a **story** version, taller again;
  * a **print**, whose aspect is decided by the paper in the machine and whose
    resolution is decided by the printer, not the model;
  * a **loop** on a screen by the door, showing the evening's best frames.

Those are five different crops of one generation. Producing them by hand, or by
letting each consumer crop the square themselves, is how a subject ends up with
their head out of frame on the print and centred on the wall.

So a surface is declared: its pixel size, how much of it is safe to put
anything in, where the mark goes, and -- the part that matters most -- where
the *subject* should sit within it. Reframing is then a function of the
generation and the surface, and the subject's position is preserved by
construction rather than by luck.

**Cover, never stretch.** Every surface fit scales to fill and crops the
excess. A person made 12% wider to fit a wall is a worse failure than a person
with less headroom, and it is the failure nobody notices until the photographs
are printed.

**Safe area is not decoration.** An LED wall is partly occluded by the human
standing in front of it; a print is trimmed; a phone crops a story under its
own UI. Each of those is a region where content survives but should not be
relied on, which is exactly what a margin is for.
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

LIVE = "live"            # driven continuously; the guest is standing in front of it
DELIVERED = "delivered"  # a file the guest or client receives
PRINT = "print"          # goes on paper, so DPI is a real number and not a guess


@dataclass(frozen=True)
class Surface:
    """One place a frame can end up.

    `subject_anchor` is the fraction of the surface height where the subject's
    *feet* should land. It is the field that does the real work: a person
    centred in a square is standing in mid-air once that square becomes a
    9:16 story, and no amount of cover-fitting fixes it.
    """
    id: str
    label: str
    width: int
    height: int
    kind: str = DELIVERED
    safe_margin_pct: float = 4.0
    logo_corner: str | None = "bottom-right"
    subject_anchor: float = 0.94
    dpi: int | None = None
    notes: str = ""

    @property
    def aspect(self) -> float:
        return self.width / self.height

    def to_dict(self) -> dict:
        return {"id": self.id, "label": self.label, "width": self.width,
                "height": self.height, "aspect": round(self.aspect, 4), "kind": self.kind,
                "safe_margin_pct": self.safe_margin_pct, "logo_corner": self.logo_corner,
                "subject_anchor": self.subject_anchor, "dpi": self.dpi, "notes": self.notes}


SURFACES: dict[str, Surface] = {
    "led_backdrop": Surface(
        "led_backdrop", "LED wall / projector behind the subject",
        1920, 1080, kind=LIVE, safe_margin_pct=6.0, logo_corner=None,
        subject_anchor=1.0,
        notes="No logo: the mark belongs on the frame the guest keeps, not on the "
              "wall the guest is standing in front of, where their body would cover it."),
    "ultrawide_backdrop": Surface(
        "ultrawide_backdrop", "Ultrawide LED wall (2x1080p)",
        3840, 1080, kind=LIVE, safe_margin_pct=6.0, logo_corner=None,
        subject_anchor=1.0,
        notes="The shape that makes the square Spout sender obviously wrong."),
    "frame_4x5": Surface(
        "frame_4x5", "Delivered frame (4:5, portrait feed)",
        1080, 1350, kind=DELIVERED, safe_margin_pct=4.0),
    "story_9x16": Surface(
        "story_9x16", "Story / reel (9:16)",
        1080, 1920, kind=DELIVERED, safe_margin_pct=8.0, subject_anchor=0.88,
        notes="Larger margin and a higher subject: phone UI eats the top and bottom "
              "bands, and a guest's feet at the very edge sit under a progress bar."),
    "print_2x6": Surface(
        "print_2x6", "Photo-booth strip, 2x6 inch at 300dpi",
        600, 1800, kind=PRINT, safe_margin_pct=7.0, dpi=300, subject_anchor=0.90,
        notes="Trimmed on cut, so the margin is bleed rather than taste."),
    "print_6x4": Surface(
        "print_6x4", "Postcard print, 6x4 inch at 300dpi",
        1800, 1200, kind=PRINT, safe_margin_pct=5.0, dpi=300, subject_anchor=0.98),
    "loop_16x9": Surface(
        "loop_16x9", "Event screen loop (16:9)",
        1920, 1080, kind=DELIVERED, safe_margin_pct=5.0, subject_anchor=1.0),
}

DEFAULT_LIVE_SURFACE = "led_backdrop"


class SurfaceError(ValueError):
    """An unknown surface id, raised with the valid ones listed."""


def get(surface_id: str) -> Surface:
    try:
        return SURFACES[surface_id]
    except KeyError:
        raise SurfaceError(
            f"unknown surface {surface_id!r}; expected one of: "
            + ", ".join(sorted(SURFACES))) from None


def parse_list(raw: str | None) -> list[Surface]:
    """Comma-separated ids from a form field or a CLI flag. Empty means none --
    surfaces are opt-in, because writing six renders of every photo triples the
    size of a fifty-photo zip for an operator who only wanted the frames."""
    if not raw or not raw.strip():
        return []
    return [get(part.strip()) for part in raw.split(",") if part.strip()]


def options() -> list[dict]:
    return [s.to_dict() for s in SURFACES.values()]


def subject_span(mask: Image.Image | None) -> tuple[float, float] | None:
    """Head and feet as fractions of image height, or None if there is no
    subject to find."""
    if mask is None:
        return None
    import numpy as np
    arr = np.asarray(mask.convert("L"), dtype=np.float32)
    rows = np.where(arr.max(axis=1) > 127)[0]
    if rows.size == 0:
        return None
    height = arr.shape[0]
    return float(rows[0]) / height, float(rows[-1]) / height


def subject_footline(mask: Image.Image | None) -> float | None:
    """Where the subject's feet are, as a fraction of image height.

    None when there is no mask or nothing in it -- in which case reframing
    falls back to a centre crop, which is the right answer for a frame with no
    subject in it.
    """
    span = subject_span(mask)
    return None if span is None else span[1]


# How much further than cover-fit the image may be scaled in order to honour
# the subject anchor.
#
# Cover-fit alone often leaves *no* vertical slack to place the subject with.
# A square photograph fitted to a 9:16 story is the ordinary case: it scales to
# exactly the story's height, the crop window can only sit at zero, and the
# anchor -- the field this whole module is built around -- silently does
# nothing on the surface that needs it most.
#
# Scaling a little past cover buys that slack back, at the cost of cropping
# more of the frame. Bounded, and modestly: this trades away headroom, and an
# unbounded version would happily zoom until it removed a guest's head to get
# their feet onto a line. 1.18 is enough to move a subject ~15% of the frame
# height, which covers the square-into-story case and stops well short of
# reframing anybody's portrait for them.
MAX_ANCHOR_ZOOM = 1.18

# Where the top of a subject's head sits when the subject is too tall to fit
# the surface at all, as a fraction of the surface height.
#
# This is the case that anchoring feet gets catastrophically wrong. A standing
# full-length portrait cover-fitted into a 6x4 landscape postcard is nearly
# twice as tall as the frame; obediently placing the feet at 0.98 produced a
# postcard of a pair of shoes and a pair of knees, with the guest's face
# cropped off entirely. It was doing exactly what it was told, and what it was
# told was wrong: an anchor for feet only means anything if there is a body
# above them still in frame.
#
# So when the subject cannot fit, the head wins instead. A 6x4 of a standing
# figure becomes a waist-up portrait, which is what a person would have
# cropped, and the deliverable has a face in it.
HEADROOM = 0.10


def reframe(image: Image.Image, surface: Surface, *, mask: Image.Image | None = None) -> Image.Image:
    """Fits a finished frame to a surface: cover-scale, then crop.

    The crop is *placed*, not centred. Where a mask is available the vertical
    offset is chosen so the subject's feet land on the surface's
    `subject_anchor`; without one it falls back to centre. This is the whole
    reason a surface knows where feet go -- a centre crop of a full-length
    portrait into a 6x4 postcard cuts a person off at the knees and looks
    deliberate, which is worse than looking broken.

    Best-effort, and bounded on both sides: the window is never allowed off the
    picture (a black band is worse than a badly placed subject), and the extra
    zoom that buys placement room is capped at MAX_ANCHOR_ZOOM. A subject can
    therefore land short of their anchor; they can never land outside the frame.

    Where the subject is too tall for the surface to hold at all, the head is
    kept and the feet are given up -- see HEADROOM.
    """
    target_w, target_h = surface.width, surface.height
    cover = max(target_w / image.width, target_h / image.height)
    span = subject_span(mask)

    if span is None:
        scaled = image.resize((max(1, round(image.width * cover)),
                               max(1, round(image.height * cover))), Image.LANCZOS)
        return _crop(scaled, target_w, target_h,
                     round((scaled.height - target_h) / 2))

    head, feet = span
    scale = cover

    fits = (feet - head) * image.height * cover <= target_h
    if fits:
        # The window can only be moved if the scaled image is taller than the
        # target. Ask for exactly enough height that the requested offset fits,
        # then take the smaller of that and the cap.
        wanted_top = feet * image.height * cover - surface.subject_anchor * target_h
        if wanted_top < 0:
            # Feet are too high in the source: the image has to grow so there
            # is something above the window to crop away.
            needed = (surface.subject_anchor * target_h) / max(feet * image.height, 1e-3)
            scale = min(max(cover, needed), cover * MAX_ANCHOR_ZOOM)

    scaled_w, scaled_h = max(1, round(image.width * scale)), max(1, round(image.height * scale))
    scaled = image.resize((scaled_w, scaled_h), Image.LANCZOS)

    if fits:
        top = round(feet * scaled_h - surface.subject_anchor * target_h)
    else:
        top = round(head * scaled_h - HEADROOM * target_h)
    return _crop(scaled, target_w, target_h, top)


def _crop(scaled: Image.Image, target_w: int, target_h: int, top: int) -> Image.Image:
    """Takes the window, clamped onto the picture. Horizontal always centres:
    a booth subject stands in the middle of frame."""
    left = round((scaled.width - target_w) / 2)
    top = max(0, min(top, scaled.height - target_h))
    left = max(0, min(left, scaled.width - target_w))
    return scaled.crop((left, top, left + target_w, top + target_h))


def safe_box(surface: Surface) -> tuple[int, int, int, int]:
    """The rectangle inside the margin, in surface pixels. Used for the safe-area
    overlay and for logo placement checks."""
    margin = round(min(surface.width, surface.height) * (surface.safe_margin_pct / 100.0))
    return margin, margin, surface.width - margin, surface.height - margin
