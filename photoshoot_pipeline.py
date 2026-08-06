"""
Virtual-photoshoot analysis pipeline: takes a real photo of a model and
extracts everything needed to regenerate the environment around them while
leaving the subject's actual pixels untouched.

    rotoscope   -> segment_subject()   (rembg / U2Net)
    pose        -> estimate_pose()     (OpenPose/DWPose via controlnet_aux)
    depth       -> estimate_depth()    (MiDaS via controlnet_aux)
    lighting    -> estimate_illumination()  (plain CV, no model — mirrors the
                   descriptor-extraction approach in resolume_state.py)

Each stage is independent and returns a plain PIL Image (or dict, for
illumination) so the web layer can show them as separate inspectable/
layerable results rather than one opaque black-box call.

Models are lazy-loaded and cached at module scope — the CV stages run on
CPU (a few seconds each) so they don't fight the GPU with ComfyUI's SDXL
generation, which runs concurrently.
"""

import colorsys
from dataclasses import dataclass, asdict

import numpy as np
from PIL import Image

_rembg_session = None
_pose_detector = None
_depth_detector = None


def _get_rembg_session():
    global _rembg_session
    if _rembg_session is None:
        from rembg import new_session
        # birefnet-portrait gives noticeably cleaner hair/fringe edges than
        # u2net (worth it for a photoshoot tool where the subject cutout is
        # the whole point) at the cost of ~20s/image on CPU — this machine's
        # onnxruntime-gpu build wants CUDA 13 libs that aren't published yet,
        # so this runs CPU-only until that's available.
        _rembg_session = new_session("birefnet-portrait")
    return _rembg_session


def _get_pose_detector():
    global _pose_detector
    if _pose_detector is None:
        from controlnet_aux import OpenposeDetector
        _pose_detector = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
    return _pose_detector


def _get_depth_detector():
    global _depth_detector
    if _depth_detector is None:
        from controlnet_aux import MidasDetector
        _depth_detector = MidasDetector.from_pretrained("lllyasviel/Annotators")
    return _depth_detector


# --- rotoscope ---------------------------------------------------------

def segment_subject(image: Image.Image) -> tuple[Image.Image, Image.Image]:
    """Returns (cutout_rgba, mask_l). mask is 255=subject, 0=background."""
    from rembg import remove

    cutout = remove(image.convert("RGB"), session=_get_rembg_session())
    mask = cutout.split()[-1]  # alpha channel is already the subject mask, hard-edged

    # rembg's raw alpha edge is a hard cutoff, which shows as a visible seam
    # once composited over a different background. Soften just the cutout's
    # own alpha for display — `mask` stays hard-edged since that's what
    # drives the server-side inpaint boundary, where a crisp mask is normal.
    from PIL import ImageFilter
    feathered_alpha = mask.filter(ImageFilter.GaussianBlur(radius=1.5))
    r, g, b, _ = cutout.split()
    cutout = Image.merge("RGBA", (r, g, b, feathered_alpha))

    return cutout, mask


# --- pose ---------------------------------------------------------------

def estimate_pose(image: Image.Image) -> Image.Image:
    """Returns an RGBA skeleton overlay — black background pixels are made
    transparent so this can sit as a layer above other content."""
    detector = _get_pose_detector()
    pose_rgb = detector(image.convert("RGB"), include_hand=True, include_face=True).convert("RGB")

    arr = np.array(pose_rgb)
    alpha = (arr.max(axis=-1) > 10).astype(np.uint8) * 255
    rgba = np.dstack([arr, alpha])
    return Image.fromarray(rgba, mode="RGBA")


# --- depth ----------------------------------------------------------------

def estimate_depth(image: Image.Image) -> Image.Image:
    """Returns a grayscale depth map (white = near, black = far)."""
    detector = _get_depth_detector()
    return detector(image.convert("RGB")).convert("RGB")


# --- illumination (plain CV, no model) -------------------------------------

@dataclass
class Illumination:
    direction: str        # e.g. "upper-left"
    warmth: str            # "warm" | "neutral" | "cool"
    softness: str           # "soft, diffuse" | "hard, directional"
    contrast: float
    descriptor: str        # ready-to-use text for a generation prompt

    def to_dict(self):
        return asdict(self)


def estimate_illumination(image: Image.Image, mask: Image.Image) -> Illumination:
    """Estimates key-light direction, color temperature, and softness by
    analyzing luminance across the subject region only — same technique
    resolume_state.py uses for effect/hue descriptors, applied to lighting
    instead of composition."""
    rgb = image.convert("RGB").resize((128, 128))
    m = mask.resize((128, 128))
    rgb_arr = np.asarray(rgb).astype(np.float32)
    mask_arr = np.asarray(m).astype(np.float32) / 255.0

    luminance = (0.299 * rgb_arr[..., 0] + 0.587 * rgb_arr[..., 1] + 0.114 * rgb_arr[..., 2])
    subject_px = mask_arr > 0.5
    if not subject_px.any():
        subject_px = np.ones_like(mask_arr, dtype=bool)

    ys, xs = np.nonzero(subject_px)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    ymid, xmid = (y0 + y1) // 2, (x0 + x1) // 2

    def quadrant_mean(y_lo, y_hi, x_lo, x_hi):
        region = subject_px[y_lo:y_hi, x_lo:x_hi]
        lum = luminance[y_lo:y_hi, x_lo:x_hi]
        if not region.any():
            return 0.0
        return float(lum[region].mean())

    quadrants = {
        "upper-left": quadrant_mean(y0, ymid, x0, xmid),
        "upper-right": quadrant_mean(y0, ymid, xmid, x1),
        "lower-left": quadrant_mean(ymid, y1, x0, xmid),
        "lower-right": quadrant_mean(ymid, y1, xmid, x1),
    }
    brightest, dimmest = max(quadrants, key=quadrants.get), min(quadrants, key=quadrants.get)
    spread = quadrants[brightest] - quadrants[dimmest]
    direction = brightest if spread > 8 else "front"  # flat-lit -> no strong direction

    subject_lum = luminance[subject_px]
    contrast = float(subject_lum.std())
    softness = "hard, directional" if contrast > 45 else "soft, diffuse"

    r_mean = rgb_arr[..., 0][subject_px].mean()
    b_mean = rgb_arr[..., 2][subject_px].mean()
    warm_bias = r_mean - b_mean
    warmth = "warm" if warm_bias > 6 else ("cool" if warm_bias < -6 else "neutral")

    if direction == "front":
        direction_phrase = "even front lighting"
    else:
        direction_phrase = f"key light from the {direction}"

    descriptor = f"{warmth} {direction_phrase}, {softness} shadows"

    return Illumination(
        direction=direction,
        warmth=warmth,
        softness=softness,
        contrast=round(contrast, 1),
        descriptor=descriptor,
    )


# --- convenience: run everything at once -----------------------------------

def analyze(image: Image.Image) -> dict:
    """Runs all four stages and returns a dict of results, for the web
    layer to hand straight to the client as separate layers."""
    cutout, mask = segment_subject(image)
    pose = estimate_pose(image)
    depth = estimate_depth(image)
    illumination = estimate_illumination(image, mask)
    return {
        "cutout": cutout,
        "mask": mask,
        "pose": pose,
        "depth": depth,
        "illumination": illumination,
    }
