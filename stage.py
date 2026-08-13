"""
Giving the generator somewhere to put a room.

The depth map handed to ControlNet looked like this: a bright silhouette of the
subject on pure black. In an inverse-depth map black means *infinitely far*, so
the model was being told, every single time, that everything which is not this
person is at infinite distance.

That single fact explains almost every complaint about the output. A model that
believes the background is infinitely far away cannot put a wall beside the
subject, a table in front of them, or a tree behind their shoulder -- there is
no geometry for any of it to attach to. The only thing it can legally paint is
a distant backdrop, which is exactly what it did: haze, sky, fog, a far
skyline. Every result was a person standing *in front of* a place rather than
*inside* one, and the composite read as a sticker no matter how good the plate
was.

So before generating, this synthesises the geometry a room would have:

**A ground plane.** Nearest at the bottom edge of the frame, receding to the
horizon. This is the single most valuable addition -- it is what lets the model
put the subject's feet on something, and what stops the "white studio floor
ends in a hard seam" failure.

**A horizon**, derived from the subject rather than fixed. A standing figure
tells you where the camera was: eye level in the picture is the horizon in the
world. Placing it at a constant fraction of the frame would fight the photo
whenever a guest is short, tall, or sitting.

**A backdrop at a chosen distance** rather than at infinity, so a wall can
exist. Whether that reads as a parapet, a treeline or a cyclorama is the
prompt's business; this only insists that there is *something there*.

**Optional side walls** -- in a real room the side walls are nearer to the
camera at the frame edges than in the middle, which is the cue that turns an
open field into an interior.

**What this is not.** It is not a 3D reconstruction and it does not know what
is actually behind the guest -- the studio wall two metres back is gone either
way. It is a *prior*: a plausible stage, deliberately smooth and low-detail, so
the model treats it as a constraint on where surfaces are rather than as an
edge map to trace. A high-frequency fake would show through in the output.

The subject's own measured depth is always preserved exactly. The stage is
built behind them, never over them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter

# Depth convention throughout: 1.0 = nearest, 0.0 = farthest, matching
# photoshoot_pipeline.estimate_depth ("white = near, black = far"). Every value
# below is in that space, so they can be read as "how close is this surface".


@dataclass(frozen=True)
class Stage:
    """One kind of space to build behind the subject.

    The numbers are deliberately few. This is a prior, and a prior with twenty
    knobs is a modelling project -- these are the four decisions that visibly
    change the result.
    """
    id: str
    label: str
    floor_near: float = 0.62      # depth at the bottom edge of the frame
    floor_far: float = 0.18       # depth where the floor reaches the horizon
    backdrop: float = 0.10        # depth above the horizon; 0 would be infinity again
    side_walls: float = 0.0       # >0 pulls the frame edges nearer, reading as an interior
    horizon_bias: float = 0.12    # where eye level sits between head (0) and feet (1)
    blur: float = 24.0            # smoothing; a sharp prior gets traced instead of obeyed

    def to_dict(self) -> dict:
        return {"id": self.id, "label": self.label, "floor_near": self.floor_near,
                "floor_far": self.floor_far, "backdrop": self.backdrop,
                "side_walls": self.side_walls}


# A small closed set, like the brand looks they pair with. Adding one is a
# line here; letting each look invent its own numbers would mean nobody could
# say why two looks composite differently.
STAGES: dict[str, Stage] = {
    "terrace": Stage(
        "terrace", "Open terrace — floor, low wall, distance beyond",
        floor_near=0.66, floor_far=0.20, backdrop=0.12),
    "room": Stage(
        "room", "Interior — floor, near back wall, side walls",
        floor_near=0.70, floor_far=0.34, backdrop=0.30, side_walls=0.22, blur=18.0),
    "landscape": Stage(
        "landscape", "Open ground running to a far horizon",
        floor_near=0.60, floor_far=0.10, backdrop=0.05, horizon_bias=0.18),
    "studio": Stage(
        "studio", "Seamless cyclorama — floor curving into the backdrop",
        floor_near=0.68, floor_far=0.42, backdrop=0.40, blur=34.0),
    "void": Stage(
        "void", "No stage: the old behaviour, background left at infinity",
        floor_near=0.0, floor_far=0.0, backdrop=0.0, blur=0.0),
}

DEFAULT_STAGE = "terrace"


class StageError(ValueError):
    """An unknown stage id. A distinct type so a brand kit naming a stage that
    does not exist fails at load with the file named, rather than silently
    falling back and looking like the prior is not working."""


def get(stage_id: str | None) -> Stage:
    if not stage_id:
        return STAGES[DEFAULT_STAGE]
    try:
        return STAGES[stage_id]
    except KeyError:
        raise StageError(
            f"unknown stage {stage_id!r}; expected one of: " + ", ".join(sorted(STAGES))) from None


def options() -> list[dict]:
    return [s.to_dict() for s in STAGES.values()]


def subject_bounds(mask: Image.Image) -> tuple[float, float] | None:
    """Top and bottom of the subject as fractions of frame height, or None if
    the mask is empty. Used to place the horizon: a standing figure tells you
    where the camera was."""
    arr = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    rows = np.where(arr.max(axis=1) > 0.5)[0]
    if rows.size == 0:
        return None
    height = arr.shape[0]
    return float(rows[0]) / height, float(rows[-1]) / height


def horizon_for(mask: Image.Image, stage: Stage) -> float:
    """Eye level, as a fraction of frame height.

    Derived from the subject rather than fixed at the middle of the frame,
    because the camera height is what a horizon actually encodes -- and a booth
    photographs children, seated guests and people at very different distances.
    Falls back to the upper third only when there is no subject to read.
    """
    bounds = subject_bounds(mask)
    if bounds is None:
        return 0.45
    head, feet = bounds
    return head + (feet - head) * stage.horizon_bias


def build_depth(depth: Image.Image, mask: Image.Image, stage: Stage | str | None = None,
                *, horizon: float | None = None) -> Image.Image:
    """Returns a new depth map with a stage built behind the subject.

    The subject's measured depth is copied through untouched. Everything the
    mask does not cover is replaced -- not blended with -- because what is
    there now is the void this whole module exists to remove.
    """
    stage = stage if isinstance(stage, Stage) else get(stage)
    depth_l = depth.convert("L")
    mask_l = mask.convert("L").resize(depth_l.size, Image.LANCZOS)

    subject = np.asarray(depth_l, dtype=np.float32) / 255.0
    alpha = np.asarray(mask_l, dtype=np.float32) / 255.0
    height, width = subject.shape

    if stage.id == "void":
        return depth_l

    y_horizon = (horizon if horizon is not None else horizon_for(mask_l, stage))
    y_horizon = float(np.clip(y_horizon, 0.02, 0.98))

    ys = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]

    # Floor: linear from the horizon down to the bottom edge. Linear rather
    # than perspective-correct on purpose -- a true 1/z ramp crushes almost the
    # whole floor into the last few rows, and the model reads that as a cliff.
    floor_t = np.clip((ys - y_horizon) / max(1e-3, 1.0 - y_horizon), 0.0, 1.0)
    floor = stage.floor_far + (stage.floor_near - stage.floor_far) * floor_t

    background = np.where(ys >= y_horizon, floor, np.float32(stage.backdrop))
    background = np.repeat(background, width, axis=1)

    if stage.side_walls > 0:
        # Side walls are nearer to the camera at the frame edges. A squared
        # profile keeps the centre flat so the backdrop stays a backdrop and
        # only the margins turn into walls.
        xs = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
        background = background + stage.side_walls * (xs ** 2)

    background = np.clip(background, 0.0, 1.0)

    if stage.blur > 0:
        smoothed = Image.fromarray((background * 255).astype(np.uint8), mode="L")
        smoothed = smoothed.filter(ImageFilter.GaussianBlur(radius=stage.blur))
        background = np.asarray(smoothed, dtype=np.float32) / 255.0

    # The subject wins wherever the matte says so. Composited with the mask as
    # a weight rather than a hard threshold so the one-pixel edge does not
    # become a depth cliff -- ControlNet treats a hard depth step as a real
    # occluding edge and will happily draw a seam along it.
    combined = alpha * subject + (1.0 - alpha) * background
    return Image.fromarray((np.clip(combined, 0, 1) * 255).astype(np.uint8), mode="L")


def background_relief(depth: Image.Image, mask: Image.Image) -> float:
    """How much depth structure exists outside the subject, 0..1.

    Reported in provenance so "the prior is working" is a number rather than an
    impression: the void scores ~0 by construction, so a manifest full of
    zeroes says the stage never got built.

    The subject is excluded with a *dilated* mask, not the matte itself.
    Downsampling to 128px smears the subject's own bright silhouette a couple
    of pixels outward, and those pixels are the highest-contrast ones in the
    frame -- measured naively, the void scored 0.10 purely from its own edge,
    which is a third of the way to a real stage. A margin of error that large
    in the number that certifies the fix would be worse than not reporting it.
    """
    arr = np.asarray(depth.convert("L").resize((128, 128)), dtype=np.float32) / 255.0
    shrunk = mask.convert("L").resize((128, 128))
    alpha = np.asarray(shrunk.filter(ImageFilter.MaxFilter(5)), dtype=np.float32) / 255.0
    background = arr[alpha < 0.5]
    if background.size == 0:
        return 0.0
    return float(np.clip(background.std() * 3.0, 0.0, 1.0))
