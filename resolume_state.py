"""
Read Resolume's live composition state over its own REST API and turn
whatever's currently on screen into a text prompt — so a generation
backend can produce a still that matches what the VJ actually built,
instead of a canned prompt from prompts.json.

Requires Preferences > Webserver enabled in Resolume (default port 8080).
Reference: https://resolume.com/support/en/restapi
"""

import requests

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


def _value(param, default=None):
    if not isinstance(param, dict):
        return default
    return param.get("value", default)


def fetch_composition(resolume_url: str = DEFAULT_RESOLUME_URL) -> dict:
    resp = requests.get(f"{resolume_url}/composition", timeout=5)
    resp.raise_for_status()
    return resp.json()


def describe_composition(composition: dict) -> str:
    """Active clip names + active (non-bypassed) effect descriptors,
    front-to-back across visible layers, de-duplicated."""
    fragments = []

    for layer in composition.get("layers", []):
        layer_opacity = _value((layer.get("video") or {}).get("opacity"), 1.0)
        if layer_opacity is not None and layer_opacity <= 0.02:
            continue  # layer effectively invisible

        active_clip = None
        for clip in layer.get("clips", []):
            if _value(clip.get("connected")) in ACTIVE_CONNECT_STATES:
                active_clip = clip
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

    # de-dupe while preserving order
    seen = set()
    unique = []
    for fragment in fragments:
        if fragment and fragment not in seen:
            seen.add(fragment)
            unique.append(fragment)
    return ", ".join(unique)


def compose_prompt(resolume_url: str = DEFAULT_RESOLUME_URL, style_suffix: str = "") -> str:
    description = describe_composition(fetch_composition(resolume_url))
    if not description:
        return style_suffix
    return f"{description}, {style_suffix}" if style_suffix else description
