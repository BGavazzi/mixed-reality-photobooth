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

import cv2
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

_MIN_COMPONENT_AREA_RATIO = 0.03  # relative to the largest blob's area


def _drop_tiny_disconnected_blobs(mask: Image.Image) -> Image.Image:
    """Zeroes out alpha blobs that are disconnected from the main subject
    AND tiny relative to it. rembg occasionally misclassifies a small
    disconnected patch of a busy background (e.g. a sliver of a chair leg)
    as subject — a real floating artifact, not a body part — since
    alpha-channel thresholding has no notion of connectivity.

    Deliberately NOT "keep only the largest blob": a subject can
    legitimately have smaller disconnected regions that are still part of
    them — a dangling earring, a held bag/phone, a strand of hair separated
    by anti-aliasing — and discarding those would silently cut off a real
    part of the subject. The area-ratio threshold is a heuristic, not a
    guarantee: it trades "might keep a very small real artifact" against
    "won't ever amputate a normal-sized accessory," which is the safer
    failure mode for a photoshoot tool."""
    binary = (np.asarray(mask) > 127).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 2:  # just background (0) + the one subject blob
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_area = areas.max()
    keep_labels = {1 + i for i, area in enumerate(areas) if area >= largest_area * _MIN_COMPONENT_AREA_RATIO}
    keep = np.isin(labels, list(keep_labels))
    cleaned = np.where(keep, np.asarray(mask), 0).astype(np.uint8)
    return Image.fromarray(cleaned, mode="L")


def segment_subject(image: Image.Image) -> tuple[Image.Image, Image.Image]:
    """Returns (cutout_rgba, mask_l). mask is 255=subject, 0=background."""
    from rembg import remove

    cutout = remove(image.convert("RGB"), session=_get_rembg_session())
    mask = _drop_tiny_disconnected_blobs(cutout.split()[-1])  # alpha channel is the subject mask, hard-edged

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


# --- ground-contact shadow (deterministic, no AI) ---------------------------

# COCO-18/OpenPose body keypoint order used by controlnet_aux's Body model.
_ANKLE_KEYPOINT_INDICES = (10, 13)  # RAnkle, LAnkle


def detect_foot_points(image: Image.Image) -> list[tuple[float, float]]:
    """Returns pixel-space (x, y) of any detected ankle keypoints. Pulled
    from the same OpenPose model estimate_pose() uses, but read before it
    gets rendered to a skeleton image — detect_poses() hands back the raw
    (normalized 0-1) keypoint coordinates directly."""
    detector = _get_pose_detector()
    np_img = np.array(image.convert("RGB"))
    poses = detector.detect_poses(np_img, include_hand=False, include_face=False)
    if not poses:
        return []

    body = poses[0].body
    w, h = image.size
    points = []
    for idx in _ANKLE_KEYPOINT_INDICES:
        if idx < len(body.keypoints) and body.keypoints[idx] is not None:
            kp = body.keypoints[idx]
            points.append((kp.x * w, kp.y * h))
    return points


def generate_contact_shadow(image_size: tuple[int, int], foot_points: list[tuple[float, float]],
                             opacity: int = 110) -> Image.Image:
    """Classic photo-compositing trick, deliberately not AI: a soft dark
    ellipse under each detected foot. Fast, deterministic, and always
    correctly grounds the subject regardless of how good (or not) the
    generated background turns out — a useful contrast to "let the model
    figure it out." Returns an RGBA layer, transparent except the shadow(s)."""
    from PIL import ImageDraw, ImageFilter

    w, h = image_size
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    if not foot_points:
        return layer

    draw = ImageDraw.Draw(layer)
    ellipse_w = w * 0.09
    ellipse_h = ellipse_w * 0.30
    foot_offset_y = h * 0.015  # ankle keypoint sits a bit above the actual sole

    for x, y in foot_points:
        gy = y + foot_offset_y
        draw.ellipse(
            [x - ellipse_w / 2, gy - ellipse_h / 2, x + ellipse_w / 2, gy + ellipse_h / 2],
            fill=(0, 0, 0, opacity),
        )

    return layer.filter(ImageFilter.GaussianBlur(radius=max(2.0, ellipse_h * 0.5)))


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


# --- background complexity -> suggested ControlNet strength (plain CV) -----

# Depth ControlNet conditions the new background on the *original* scene's
# geometry. A plain backdrop has near-flat depth, so a high strength barely
# constrains the model and the prompt is free to invent a whole new
# environment. A background with real structure (furniture, a doorway, a
# wall at an angle) has real depth variance -- at the same high strength,
# the model ends up re-texturing that existing geometry instead of
# replacing it, so a "reading nook" prompt over a room with a chair in it
# comes back looking like the same room with a chair in it. Observed
# directly: a photo with a leather armchair in frame barely changed under a
# "cozy reading nook, bookshelves" prompt at strength 0.75.
BUSY_BACKGROUND_DEPTH_STD = 25.0
PLAIN_BACKGROUND_STRENGTH = 0.75
BUSY_BACKGROUND_STRENGTH = 0.45


def suggest_controlnet_strength(depth: Image.Image, mask: Image.Image) -> float:
    """Heuristic, not a guarantee: measures depth variance in the
    background region (inverse of the subject mask) and suggests a lower
    ControlNet strength when that background has real 3D structure to
    fight against, rather than always defaulting to the same value
    regardless of what's actually behind the subject."""
    depth_arr = np.asarray(depth.convert("L").resize((128, 128))).astype(np.float32)
    mask_arr = np.asarray(mask.resize((128, 128))).astype(np.float32) / 255.0
    background_px = depth_arr[mask_arr < 0.5]
    if background_px.size == 0:
        return PLAIN_BACKGROUND_STRENGTH
    variance = float(background_px.std())
    return BUSY_BACKGROUND_STRENGTH if variance > BUSY_BACKGROUND_DEPTH_STD else PLAIN_BACKGROUND_STRENGTH


# --- convenience: run everything at once -----------------------------------

# workflows/photoshoot_bg_api.json has no resize node -- VAEEncode runs at
# the subject photo's *native* resolution. A realistic camera/phone photo
# (e.g. 3875x5812, 22MP) pushed straight into that drove ComfyUI into a VAE
# out-of-memory fallback ("retrying with tiled VAE encoding") and dropped
# sampling from ~1.2s/step to ~41s/step -- observed directly: interrupted
# after 374s at step 7/30, on track for ~20 minutes with zero error surfaced
# to the user. 1536px keeps CPU stages fast and generation in the normal
# 30-45s range while still being well above SDXL's native working resolution.
MAX_INPUT_DIMENSION = 1536


def cap_resolution(image: Image.Image) -> Image.Image:
    if max(image.size) <= MAX_INPUT_DIMENSION:
        return image
    resized = image.copy()
    resized.thumbnail((MAX_INPUT_DIMENSION, MAX_INPUT_DIMENSION), Image.LANCZOS)
    return resized


def analyze(image: Image.Image) -> dict:
    """Runs all stages and returns a dict of results, for the web layer to
    hand straight to the client as separate layers. `image` is downscaled
    first if oversized -- callers should treat the returned `image` as the
    new canonical original (e.g. what gets sent back to the browser and
    reused for later generation calls), not the caller's original object."""
    image = cap_resolution(image)
    cutout, mask = segment_subject(image)
    pose = estimate_pose(image)
    depth = estimate_depth(image)
    illumination = estimate_illumination(image, mask)
    foot_points = detect_foot_points(image)
    shadow = generate_contact_shadow(image.size, foot_points)
    suggested_controlnet_strength = suggest_controlnet_strength(depth, mask)
    return {
        "image": image,
        "cutout": cutout,
        "mask": mask,
        "suggested_controlnet_strength": suggested_controlnet_strength,
        "pose": pose,
        "depth": depth,
        "illumination": illumination,
        "shadow": shadow,
    }
