"""
Tests for brand_kit.py -- the rules that make a generation signable off.

Three things here are load-bearing and everything else is detail:

  * the kit's negative prompt survives every path a prompt can take, because
    that string is the entire brand-safety guarantee;
  * free text can only ever be *added* to what the kit mandates, never
    substituted for it;
  * a locked seed is a pure function of (brand, look), because "the same look
    renders the same way for two hundred guests" is the promise a locked seed
    is making.

The parsing tests take decoded dicts rather than fixture files for the same
reason doctor.py's checks take their facts as arguments: a malformed kit can
then be described in one line, next to the assertion about it.
"""

import json

import pytest

import brand_kit
from brand_kit import BrandKitError

BASE_NEGATIVE = "blurry, low quality, watermark"


def minimal(**overrides) -> dict:
    data = {
        "id": "acme",
        "name": "Acme Corp",
        "version": "2026.01",
        "prompt": {
            "positive_suffix": "clean editorial finish",
            "negative_suffix": "competitor logos, alcohol",
        },
        "looks": [{"id": "studio", "label": "Studio", "prompt": "seamless white studio backdrop"}],
    }
    data.update(overrides)
    return data


def kit(**overrides) -> brand_kit.BrandKit:
    return brand_kit.parse_brand(minimal(**overrides), brand_kit.BRANDS_DIR / "acme")


# --- parsing ------------------------------------------------------------------

def test_a_minimal_kit_parses():
    brand = kit()
    assert brand.id == "acme"
    assert brand.seed_policy == "locked"          # the default, and the point of the feature
    assert [l.id for l in brand.looks] == ["studio"]


@pytest.mark.parametrize("missing", ["id", "name"])
def test_missing_required_fields_name_the_field_and_the_file(missing):
    data = minimal()
    del data[missing]
    with pytest.raises(BrandKitError) as excinfo:
        brand_kit.parse_brand(data, brand_kit.BRANDS_DIR / "acme")
    assert missing in str(excinfo.value)
    assert "brand.json" in str(excinfo.value), "the error should say which file to go and fix"


def test_a_kit_with_no_looks_is_rejected():
    """The operator picks from `looks`; an empty list leaves a brand selected
    and nothing to select under it."""
    with pytest.raises(BrandKitError):
        kit(looks=[])


def test_duplicate_look_ids_are_rejected():
    """They would shadow each other in look() -- and since the locked seed is
    derived from the look id, they would also silently share one image."""
    with pytest.raises(BrandKitError):
        kit(looks=[{"id": "a", "label": "One", "prompt": "x"},
                   {"id": "a", "label": "Two", "prompt": "y"}])


def test_a_brand_id_that_could_escape_a_url_path_is_rejected():
    """It reaches /api/brands/<id>/logo, so it gets the same scrutiny as any
    other untrusted path segment even though the route only uses it as a
    dict key."""
    for bad in ("../../etc", "acme/../../secret", "with space", "semi;colon"):
        with pytest.raises(BrandKitError):
            kit(id=bad)


def test_unknown_seed_policy_is_rejected_rather_than_defaulted():
    """Silently falling back to random would turn a kit that asked for
    reproducibility into one that doesn't have it, with no signal."""
    with pytest.raises(BrandKitError):
        kit(seed_policy="sometimes")


def test_an_invalid_logo_corner_is_rejected():
    with pytest.raises(BrandKitError):
        kit(logo={"file": "logo.png", "default_corner": "middle"})


def test_logo_rules_default_to_something_usable():
    brand = kit(logo={"file": "logo.png"})
    assert brand.logo.min_width_pct > 0
    assert brand.logo.locked_aspect is True, "a stretched logo is the violation this defends against"
    assert brand.logo.default_corner in brand_kit.LOGO_CORNERS


def test_a_kit_with_no_logo_block_is_valid():
    """Not every brand puts a mark on the frame."""
    assert kit().logo is None


# --- composition ----------------------------------------------------------------

def test_no_brand_is_a_pass_through():
    """The unbranded path has to behave exactly as it did before brand kits
    existed, or adding the feature changed an app that was already working."""
    composed = brand_kit.compose(None, None, "a rooftop at dusk", base_negative=BASE_NEGATIVE)
    assert composed.positive == "a rooftop at dusk"
    assert composed.negative == BASE_NEGATIVE
    assert composed.seed is None
    assert composed.brand_id is None


def test_the_kits_negative_is_appended_to_the_workflows_own():
    """A brand author writes what *their* client must never appear next to.
    They should not have to re-type the generic quality guards to keep them."""
    composed = brand_kit.compose(kit(), "studio", "", base_negative=BASE_NEGATIVE)
    assert composed.negative.startswith(BASE_NEGATIVE)
    assert "competitor logos" in composed.negative


def test_free_text_cannot_displace_the_locked_parts():
    composed = brand_kit.compose(kit(), "studio", "neon cyberpunk alley", base_negative=BASE_NEGATIVE)
    assert "seamless white studio backdrop" in composed.positive, "the approved look must survive"
    assert "clean editorial finish" in composed.positive, "the kit's styling must survive"
    assert "neon cyberpunk alley" in composed.positive, "the operator's text is still added"


def test_prompt_order_puts_the_approved_look_first():
    """SDXL weights leading tokens most heavily. The look leads so that free
    text colours the scene instead of replacing it."""
    composed = brand_kit.compose(kit(), "studio", "with a red chair", base_negative="")
    assert composed.positive.index("seamless white studio backdrop") \
        < composed.positive.index("with a red chair") \
        < composed.positive.index("clean editorial finish")


def test_an_unknown_look_is_an_error_not_a_silent_fallback():
    """Falling back to no look would produce a generic image while the
    disclosure card still named the brand -- the exact mismatch a brand kit
    exists to prevent."""
    with pytest.raises(BrandKitError):
        brand_kit.compose(kit(), "does-not-exist", "", base_negative="")


def test_empty_free_text_leaves_no_dangling_commas():
    composed = brand_kit.compose(kit(), "studio", "   ", base_negative=BASE_NEGATIVE)
    assert ", ," not in composed.positive
    assert not composed.positive.strip().endswith(",")


def test_a_kit_with_no_suffixes_still_composes():
    brand = kit(prompt={})
    composed = brand_kit.compose(brand, "studio", "extra", base_negative=BASE_NEGATIVE)
    assert composed.positive == "seamless white studio backdrop, extra"
    assert composed.negative == BASE_NEGATIVE


# --- seeds ------------------------------------------------------------------------

def test_a_locked_seed_is_stable_across_calls():
    """The whole promise: two hundred guests, one look, one consistent
    campaign."""
    first = brand_kit.compose(kit(), "studio", "guest one", base_negative="")
    second = brand_kit.compose(kit(), "studio", "guest two", base_negative="")
    assert first.seed == second.seed is not None


def test_different_looks_get_different_seeds():
    brand = kit(looks=[{"id": "studio", "label": "S", "prompt": "a"},
                       {"id": "rooftop", "label": "R", "prompt": "b"}])
    assert (brand_kit.compose(brand, "studio", "", base_negative="").seed
            != brand_kit.compose(brand, "rooftop", "", base_negative="").seed)


def test_different_brands_get_different_seeds_for_the_same_look_id():
    """'studio' is an obvious id for any brand to use; two clients sharing it
    must not share an image."""
    assert brand_kit.locked_seed("acme", "studio") != brand_kit.locked_seed("globex", "studio")


def test_a_locked_seed_fits_the_range_comfyui_accepts():
    for look in ("studio", "rooftop", "a-very-long-look-identifier"):
        seed = brand_kit.locked_seed("acme", look)
        assert 0 <= seed < 2 ** 32


def test_random_seed_policy_leaves_the_seed_to_the_backend():
    """None means 'don't pin it', which is what makes the backend draw a fresh
    one -- the pre-brand-kit behaviour, still available per kit."""
    assert brand_kit.compose(kit(seed_policy="random"), "studio", "", base_negative="").seed is None


# --- loading from disk --------------------------------------------------------------

def test_load_brands_reads_the_kits_shipped_with_the_repo():
    brands = brand_kit.load_brands()
    assert brands, "the repo ships demo kits; loading none means the loader or the packs broke"
    for brand in brands.values():
        assert brand.looks
        if brand.logo:
            assert brand.logo_path.exists(), f"{brand.id} names a logo file that isn't there"


def test_a_broken_pack_is_skipped_without_taking_the_others_down(tmp_path, capsys):
    """One bad JSON file should cost you that one brand, not the whole booth
    in the middle of an event."""
    (tmp_path / "good").mkdir()
    (tmp_path / "good" / "brand.json").write_text(json.dumps(minimal()), encoding="utf-8")
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "brand.json").write_text("{ not json", encoding="utf-8")

    brands = brand_kit.load_brands(tmp_path)

    assert list(brands) == ["acme"]
    assert "broken" in capsys.readouterr().out, "the skip should say which pack and why"


def test_a_missing_logo_file_is_caught_at_load_time(tmp_path):
    """Better here than as a 404 in the browser halfway through an event."""
    (tmp_path / "acme").mkdir()
    (tmp_path / "acme" / "brand.json").write_text(
        json.dumps(minimal(logo={"file": "nope.png"})), encoding="utf-8")
    with pytest.raises(BrandKitError):
        brand_kit.load_brand(tmp_path / "acme")


def test_load_brands_on_a_missing_directory_returns_empty(tmp_path):
    """Running without any brand kits is a supported configuration, not an
    error -- it is the app's original behaviour."""
    assert brand_kit.load_brands(tmp_path / "not-there") == {}


# --- the shape handed to the browser --------------------------------------------------

def test_to_dict_replaces_the_logo_path_with_a_url():
    brand = kit(logo={"file": "logo.png"})
    payload = brand.to_dict()
    assert payload["logo"]["url"] == "/api/brands/acme/logo"
    assert "file" not in payload["logo"], "no filesystem paths should reach the browser"


def test_to_dict_is_json_serialisable():
    """It goes out over /api/config, so a dataclass leaking into it would be a
    500 at page load rather than a type error anywhere useful."""
    json.dumps(kit(logo={"file": "logo.png"}).to_dict())


def test_to_provenance_records_what_a_reviewer_asks_about():
    composed = brand_kit.compose(kit(), "studio", "with a red chair", base_negative=BASE_NEGATIVE)
    record = composed.to_provenance()
    assert record["brand"] == "Acme Corp"
    assert record["brand_version"] == "2026.01"
    assert record["look"] == "Studio"
    assert record["operator_text"] == "with a red chair"
    assert "competitor logos" in record["negative_prompt"], \
        "the disclosure has to say what was actively excluded, not just what was asked for"
