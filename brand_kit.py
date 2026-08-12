"""
Brand kits: turning a free-text prompt box into something a client can sign off on.

The photo booth's original generate path took whatever the operator typed and
sent it straight to SDXL. That is fine for a personal toy and unusable for
branded work, for three reasons this module exists to fix:

  1. Nothing was repeatable. The seed came from uuid4() on every call
     (backends/comfy.py), so two hundred guests at an activation produced two
     hundred visually unrelated images. A brand wants one shoot, not two
     hundred.
  2. Nothing was enforced. There was no way to say "always include this
     lighting language" or "never generate these subjects", and the negative
     prompt was welded shut inside the workflow JSON where nobody outside
     this repo could review it.
  3. Nothing was recorded. The provenance card named the model and seed but
     had no idea which brand, which approved look, or which revision of the
     logo asset produced the frame -- the fields a brand-safety reviewer
     actually asks about.

A brand kit is a directory holding a `brand.json` and its logo artwork. The
operator picks a brand and one of its *approved looks*; the prompt is then
composed rather than typed. Free text is still allowed, but it can only ever
be added to what the kit mandates -- it cannot remove or override it.

Composition deliberately happens on the server (see compose() call sites in
web_server.py) rather than in the browser. A locked negative prompt that the
client assembles is a locked negative prompt that a modified client can drop,
which would make the whole guarantee decorative.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

BRANDS_DIR = Path(__file__).parent / "brands"

# ComfyUI's KSampler takes a 64-bit seed, but staying inside 32 bits keeps the
# number short enough to read aloud off a disclosure card and matches what the
# random path already produced (uuid4().bytes[:4]).
SEED_MODULUS = 2 ** 32

SEED_POLICIES = ("locked", "random")

# Where a logo may sit by default. Values are (x, y) anchors in 0..1 canvas
# space; the browser turns them into pixel offsets once it knows the canvas
# size, which it does and this module does not.
LOGO_CORNERS = {
    "bottom-right": (1.0, 1.0),
    "bottom-left": (0.0, 1.0),
    "top-right": (1.0, 0.0),
    "top-left": (0.0, 0.0),
}


class BrandKitError(ValueError):
    """A brand pack is malformed. Raised with the offending file named, since
    these are hand-edited JSON files and the usual author is not a
    programmer."""


@dataclass(frozen=True)
class Look:
    """One approved scene. The operator picks from these instead of typing."""
    id: str
    label: str
    prompt: str


@dataclass(frozen=True)
class LogoRules:
    """The parts of a brand book that can actually be checked automatically.

    Real guidelines run to dozens of pages; these three are the ones a booth
    operator breaks in practice -- shrinking the mark until it is unreadable,
    letting the composite crowd it, and stretching it to fit a gap.
    """
    file: str
    min_width_pct: float = 10.0      # of canvas width; below this the mark stops being legible
    clear_space_pct: float = 4.0     # margin from every canvas edge, as % of the short side
    default_corner: str = "bottom-right"
    locked_aspect: bool = True       # a stretched logo is the single most common brand violation


@dataclass(frozen=True)
class BrandKit:
    id: str
    name: str
    version: str
    directory: Path
    positive_suffix: str = ""
    negative_suffix: str = ""
    palette: tuple[dict, ...] = ()
    looks: tuple[Look, ...] = ()
    seed_policy: str = "locked"
    logo: LogoRules | None = None
    notes: str = ""

    def look(self, look_id: str) -> Look | None:
        for entry in self.looks:
            if entry.id == look_id:
                return entry
        return None

    @property
    def logo_path(self) -> Path | None:
        return self.directory / self.logo.file if self.logo else None

    def to_dict(self) -> dict:
        """The shape the browser consumes. Deliberately not the same as the
        on-disk shape: paths are replaced by the URL that serves them, and
        anything the UI has no use for is left out."""
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "palette": list(self.palette),
            "looks": [{"id": l.id, "label": l.label, "prompt": l.prompt} for l in self.looks],
            "seed_policy": self.seed_policy,
            "notes": self.notes,
            "positive_suffix": self.positive_suffix,
            "negative_suffix": self.negative_suffix,
            "logo": None if not self.logo else {
                "url": f"/api/brands/{self.id}/logo",
                "min_width_pct": self.logo.min_width_pct,
                "clear_space_pct": self.logo.clear_space_pct,
                "default_corner": self.logo.default_corner,
                "locked_aspect": self.logo.locked_aspect,
            },
        }


# --- loading ------------------------------------------------------------------

def _require(data: dict, key: str, source: Path, expected: type = str):
    if key not in data:
        raise BrandKitError(f"{source}: missing required field {key!r}")
    value = data[key]
    if not isinstance(value, expected) or (expected is str and not value.strip()):
        raise BrandKitError(f"{source}: field {key!r} must be a non-empty {expected.__name__}")
    return value


def parse_brand(data: dict, directory: Path) -> BrandKit:
    """Builds a BrandKit from already-decoded JSON.

    Split out from load_brand() so the validation rules can be tested without
    a directory of fixture files on disk -- the same reasoning behind
    doctor.py's checks taking their facts as arguments.
    """
    source = directory / "brand.json"

    brand_id = _require(data, "id", source)
    if not brand_id.replace("_", "").replace("-", "").isalnum():
        # This lands in a URL path (/api/brands/<id>/logo) and in a filename,
        # so it gets the same treatment as any other untrusted path segment.
        raise BrandKitError(f"{source}: id {brand_id!r} must be alphanumeric with - or _ only")

    prompt = data.get("prompt", {})
    if not isinstance(prompt, dict):
        raise BrandKitError(f"{source}: 'prompt' must be an object")

    looks = []
    seen_ids = set()
    for raw in data.get("looks", []):
        if not isinstance(raw, dict):
            raise BrandKitError(f"{source}: every entry in 'looks' must be an object")
        look = Look(
            id=_require(raw, "id", source),
            label=_require(raw, "label", source),
            prompt=_require(raw, "prompt", source),
        )
        if look.id in seen_ids:
            # Duplicate ids would silently shadow each other in look(), and
            # the seed is derived from the id -- two different looks sharing
            # one would then also share an image.
            raise BrandKitError(f"{source}: duplicate look id {look.id!r}")
        seen_ids.add(look.id)
        looks.append(look)

    if not looks:
        raise BrandKitError(f"{source}: a brand kit with no approved looks would leave "
                            f"the operator with nothing to pick -- add at least one 'looks' entry")

    seed_policy = data.get("seed_policy", "locked")
    if seed_policy not in SEED_POLICIES:
        raise BrandKitError(f"{source}: seed_policy must be one of {SEED_POLICIES}, got {seed_policy!r}")

    logo = None
    raw_logo = data.get("logo")
    if raw_logo is not None:
        if not isinstance(raw_logo, dict):
            raise BrandKitError(f"{source}: 'logo' must be an object")
        corner = raw_logo.get("default_corner", "bottom-right")
        if corner not in LOGO_CORNERS:
            raise BrandKitError(f"{source}: logo.default_corner must be one of "
                                f"{sorted(LOGO_CORNERS)}, got {corner!r}")
        logo = LogoRules(
            file=_require(raw_logo, "file", source),
            min_width_pct=float(raw_logo.get("min_width_pct", 10.0)),
            clear_space_pct=float(raw_logo.get("clear_space_pct", 4.0)),
            default_corner=corner,
            locked_aspect=bool(raw_logo.get("locked_aspect", True)),
        )

    return BrandKit(
        id=brand_id,
        name=_require(data, "name", source),
        version=str(data.get("version", "unversioned")),
        directory=directory,
        positive_suffix=prompt.get("positive_suffix", "").strip(),
        negative_suffix=prompt.get("negative_suffix", "").strip(),
        palette=tuple(data.get("palette", [])),
        looks=tuple(looks),
        seed_policy=seed_policy,
        logo=logo,
        notes=data.get("notes", "").strip(),
    )


def load_brand(directory: Path) -> BrandKit:
    source = directory / "brand.json"
    try:
        with open(source, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise BrandKitError(f"{source}: not valid JSON ({exc})") from exc
    if not isinstance(data, dict):
        raise BrandKitError(f"{source}: top level must be an object")

    brand = parse_brand(data, directory)
    if brand.logo and not brand.logo_path.exists():
        raise BrandKitError(f"{source}: logo file {brand.logo.file!r} does not exist in {directory}")
    return brand


def load_brands(directory: Path = BRANDS_DIR) -> dict[str, BrandKit]:
    """Loads every brand pack under `directory`.

    A broken pack is skipped with a printed reason rather than taking the
    server down: one bad JSON file should cost you that one brand, not the
    whole booth in the middle of an event.
    """
    brands: dict[str, BrandKit] = {}
    if not directory.is_dir():
        return brands
    for child in sorted(directory.iterdir()):
        if not child.is_dir() or not (child / "brand.json").exists():
            continue
        try:
            brand = load_brand(child)
        except BrandKitError as exc:
            print(f"[brands] skipping {child.name}: {exc}")
            continue
        if brand.id in brands:
            print(f"[brands] skipping {child.name}: id {brand.id!r} already used by "
                  f"{brands[brand.id].directory.name}")
            continue
        brands[brand.id] = brand
    return brands


# --- composition ----------------------------------------------------------------

@dataclass
class ComposedPrompt:
    """The result of applying a brand kit to an operator's choices.

    Carries the pieces separately as well as joined, because the UI shows the
    operator which words came from the kit (locked) and which came from them
    (free) -- a preview that just showed the final string would make the
    guarantee invisible, which is most of the point of having one.
    """
    positive: str
    negative: str
    seed: int | None
    brand_id: str | None = None
    brand_name: str | None = None
    brand_version: str | None = None
    look_id: str | None = None
    look_label: str | None = None
    free_text: str = ""
    locked_positive: str = ""
    locked_negative: str = ""

    def to_provenance(self) -> dict:
        """The brand-side fields that get stapled onto every generation record."""
        return {
            "brand": self.brand_name,
            "brand_id": self.brand_id,
            "brand_version": self.brand_version,
            "look": self.look_label,
            "look_id": self.look_id,
            "negative_prompt": self.negative,
            "operator_text": self.free_text,
        }


def _join(*parts: str) -> str:
    """Comma-joins prompt fragments, dropping empties and collapsing the
    doubled commas they would otherwise leave behind."""
    return ", ".join(part.strip().strip(",").strip() for part in parts if part and part.strip())


def locked_seed(brand_id: str, look_id: str) -> int:
    """A stable seed for a (brand, look) pair.

    Derived rather than stored so a brand pack does not have to carry a magic
    number per look, and hashed rather than incremented so two looks added in
    the same edit do not land on adjacent seeds -- neighbouring seeds are not
    meaningfully more similar than distant ones in SDXL, but they *look* like
    a mistake on a disclosure card.
    """
    digest = hashlib.sha256(f"{brand_id}/{look_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % SEED_MODULUS


def compose(
    brand: BrandKit | None,
    look_id: str | None,
    free_text: str,
    base_negative: str = "",
) -> ComposedPrompt:
    """Turns (brand, approved look, whatever the operator typed) into the
    exact strings handed to the sampler.

    With no brand this is a pass-through, so the unbranded path behaves
    exactly as it did before brand kits existed -- free text in, baked
    workflow negative out, random seed.

    Ordering is not arbitrary: SDXL weights leading tokens more heavily, so
    the approved look leads, the operator's addition follows, and the brand's
    always-on styling closes. That way free text colours the scene without
    being able to displace the look it was supposed to be decorating.
    """
    free_text = (free_text or "").strip()

    if brand is None:
        return ComposedPrompt(
            positive=free_text,
            negative=base_negative,
            seed=None,
            free_text=free_text,
        )

    look = brand.look(look_id) if look_id else None
    if look_id and look is None:
        # Not silently ignored: falling back to "no look" would hand the
        # client a generic image while the disclosure card still named the
        # brand, which is precisely the mismatch this module exists to stop.
        raise BrandKitError(f"brand {brand.id!r} has no approved look {look_id!r}")

    locked_positive = brand.positive_suffix
    locked_negative = _join(base_negative, brand.negative_suffix)

    return ComposedPrompt(
        positive=_join(look.prompt if look else "", free_text, locked_positive),
        negative=locked_negative,
        seed=(locked_seed(brand.id, look.id)
              if brand.seed_policy == "locked" and look is not None else None),
        brand_id=brand.id,
        brand_name=brand.name,
        brand_version=brand.version,
        look_id=look.id if look else None,
        look_label=look.label if look else None,
        free_text=free_text,
        locked_positive=locked_positive,
        locked_negative=locked_negative,
    )
