"""
Read Resolume's live composition state over its own REST API and turn
whatever's currently on screen into a text prompt — so a generation
backend can produce a still that matches what the VJ actually built,
instead of a canned prompt from prompts.json.

Two sources feed the prompt, and both matter:
  - Names and effects (clip name, active effect names) — readable, but
    only as good as how the VJ happened to name things.
  - The clip's actual thumbnail image (dominant color, brightness,
    contrast, texture) — pulled from Resolume's own REST API and
    analyzed directly, so the same visual content produces the same
    descriptors regardless of naming. This is what makes regeneration
    reproducible rather than just a vibes-based guess from a label.

Requires Preferences > Webserver enabled in Resolume (default port 8080).
Reference: https://resolume.com/support/en/restapi
"""

import colorsys
import io

import numpy as np
import requests
from PIL import Image

DEFAULT_RESOLUME_URL = "http://127.0.0.1:8080/api/v1"

# Resolume effect names vary by version/effect pack, and this is a
# heuristic mapping, not an exhaustive one. Anything not listed here just
# falls back to its own (lowercased) name, which is often already a
# reasonable prompt fragment (e.g. "Colorize", "Posterize").
EFFECT_DESCRIPTORS = {
    "kaleidoscope": "kaleidoscopic, symmetrical fractal patterns",
    "mirror": "mirrored, symmetrical",
    "strobe": "strobing, flickering",
    "feedback": "recursive feedback trails, echo",
    "rgbdelay": "chromatic aberration, RGB glitch",
    "glitch": "digital glitch, datamosh",
    "posterize": "posterized, flat color bands",
    "blur": "soft focus, motion blur",
    "pixelate": "pixelated, low-res blocks",
    "noise": "grainy noise, film grain",
    "invert": "inverted colors, negative",
    "edgedetect": "edge-detected, wireframe outline",
    "trails": "long exposure trails",
    "zoom": "zooming, warping perspective",
}

ACTIVE_CONNECT_STATES = {"Connected", "Previewing"}

# (upper hue bound in degrees, name) — walked in order, first match wins.
HUE_NAMES = [
    (15, "red"), (45, "orange"), (70, "yellow"), (170, "green"),
    (200, "cyan"), (250, "blue"), (290, "purple"), (330, "magenta"), (361, "red"),
]


def _value(param, default=None):
    if not isinstance(param, dict):
        return default
    return param.get("value", default)


def _hue_name(hue_deg: float) -> str:
    for threshold, name in HUE_NAMES:
        if hue_deg < threshold:
            return name
    return "red"


def fetch_composition(resolume_url: str = DEFAULT_RESOLUME_URL) -> dict:
    resp = requests.get(f"{resolume_url}/composition", timeout=5)
    resp.raise_for_status()
    return resp.json()


def fetch_clip_thumbnail(resolume_url: str, layer_index: int, clip_index: int, refresh: bool = True):
    """layer_index / clip_index are 1-based, matching Resolume's REST API."""
    base = f"{resolume_url}/composition/layers/{layer_index}/clips/{clip_index}"
    if refresh:
        try:
            requests.post(f"{base}/thumbnail/update", timeout=5)
        except requests.RequestException:
            pass  # best-effort; fall back to whatever thumbnail already exists
    resp = requests.get(f"{base}/thumbnail", timeout=5)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def describe_image(image: Image.Image) -> str:
    """Deterministic low-level visual read of an image: dominant hue,
    saturation, brightness, contrast, and edge density, turned into
    prompt-friendly adjectives. No ML, just pixel stats — same input
    image always produces the same descriptors."""
    small = image.resize((48, 48))
    arr = np.asarray(small, dtype=np.float32) / 255.0

    mean_r, mean_g, mean_b = arr[..., 0].mean(), arr[..., 1].mean(), arr[..., 2].mean()
    hue, sat, val = colorsys.rgb_to_hsv(float(mean_r), float(mean_g), float(mean_b))

    fragments = []
    if sat < 0.12:
        fragments.append("near-monochrome")
    else:
        fragments.append(f"{_hue_name(hue * 360)}-dominant")

    if sat >= 0.55:
        fragments.append("vivid, saturated colors")
    elif sat < 0.2:
        fragments.append("muted, desaturated colors")

    if val >= 0.65:
        fragments.append("bright, luminous")
    elif val < 0.3:
        fragments.append("dark, moody")

    gray = arr.mean(axis=2)
    contrast = float(gray.std())
    if contrast >= 0.22:
        fragments.append("high contrast")
    elif contrast < 0.08:
        fragments.append("flat, low contrast")

    edge_density = float(np.abs(np.diff(gray, axis=0)).mean() + np.abs(np.diff(gray, axis=1)).mean())
    if edge_density >= 0.05:
        fragments.append("busy, detailed texture")
    elif edge_density < 0.015:
        fragments.append("smooth, minimal")

    return ", ".join(fragments)


def describe_composition(composition: dict, resolume_url: str = None, use_thumbnails: bool = True) -> str:
    """Active clip names + active (non-bypassed) effect descriptors +
    (if resolume_url given) visual descriptors pulled from each active
    clip's actual thumbnail, front-to-back across visible layers,
    de-duplicated."""
    fragments = []

    for layer_idx, layer in enumerate(composition.get("layers", []), start=1):
        layer_opacity = _value((layer.get("video") or {}).get("opacity"), 1.0)
        if layer_opacity is not None and layer_opacity <= 0.02:
            continue  # layer effectively invisible

        active_clip = None
        active_clip_idx = None
        for clip_idx, clip in enumerate(layer.get("clips", []), start=1):
            if _value(clip.get("connected")) in ACTIVE_CONNECT_STATES:
                active_clip = clip
                active_clip_idx = clip_idx
                break
        if active_clip is None:
            continue

        clip_name = (_value(active_clip.get("name"), "") or "").strip()
        if clip_name:
            fragments.append(clip_name)

        effects = list((active_clip.get("video") or {}).get("effects", []))
        effects += list((layer.get("video") or {}).get("effects", []))
        for effect in effects:
            if _value(effect.get("bypassed"), False):
                continue
            name = (effect.get("name") or effect.get("displayName") or "").strip()
            key = name.lower().replace(" ", "")
            fragments.append(EFFECT_DESCRIPTORS.get(key, name.lower()))

        if use_thumbnails and resolume_url:
            try:
                thumb = fetch_clip_thumbnail(resolume_url, layer_idx, active_clip_idx)
                fragments.append(describe_image(thumb))
            except Exception as exc:
                print(f"[resolume] thumbnail fetch failed for layer {layer_idx} clip {active_clip_idx}: {exc}")

    # de-dupe while preserving order
    seen = set()
    unique = []
    for fragment in fragments:
        if fragment and fragment not in seen:
            seen.add(fragment)
            unique.append(fragment)
    return ", ".join(unique)


def compose_prompt(resolume_url: str = DEFAULT_RESOLUME_URL, style_suffix: str = "", use_thumbnails: bool = True) -> str:
    composition = fetch_composition(resolume_url)
    description = describe_composition(composition, resolume_url, use_thumbnails)
    if not description:
        return style_suffix
    return f"{description}, {style_suffix}" if style_suffix else description
