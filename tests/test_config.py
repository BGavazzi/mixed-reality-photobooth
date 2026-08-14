"""
Tests for the settings layer.

The bar here is not "does it parse an integer". It is that a *typo* cannot
silently change behaviour -- especially not in the direction that keeps
strangers' photographs on a disk. Every test below is a mistake somebody could
plausibly make in a shell.
"""

import pytest

import config
import obs


@pytest.fixture(autouse=True)
def clean_registry(monkeypatch):
    monkeypatch.setattr(config, "REGISTRY", {})
    yield


# --- reading and validating ----------------------------------------------------

def test_an_unset_variable_uses_the_default_and_says_so(monkeypatch):
    monkeypatch.delenv("SOME_KNOB", raising=False)

    assert config.env_int("SOME_KNOB", 7) == 7
    assert config.REGISTRY["SOME_KNOB"].source == "default"
    assert config.REGISTRY["SOME_KNOB"].overridden is False


def test_an_empty_variable_is_treated_as_unset(monkeypatch):
    """`FOO=` in a compose file or an exported-but-blank shell variable means
    "I did not set this", not "set this to the empty string" -- which as an int
    would have been a crash at import."""
    monkeypatch.setenv("SOME_KNOB", "")

    assert config.env_int("SOME_KNOB", 7) == 7
    assert config.REGISTRY["SOME_KNOB"].source == "default"


def test_a_non_numeric_value_names_the_variable(monkeypatch):
    """The old behaviour was a bare ValueError from int() with no mention of
    which of fifteen variables was wrong."""
    monkeypatch.setenv("SOME_KNOB", "four")

    with pytest.raises(config.ConfigError) as err:
        config.env_int("SOME_KNOB", 4)
    assert "SOME_KNOB" in str(err.value)
    assert "four" in str(err.value)


def test_a_value_outside_its_range_is_refused(monkeypatch):
    monkeypatch.setenv("SOME_KNOB", "0")

    with pytest.raises(config.ConfigError) as err:
        config.env_int("SOME_KNOB", 2, minimum=1)
    assert "minimum" in str(err.value)


def test_a_negative_retention_window_cannot_be_set_by_accident(monkeypatch):
    """The one that matters. A negative window reads as "retention off" to the
    sweep, so `BATCH_RETAIN_DAYS=-1` -- a plausible way to type "unset" --
    silently stopped deleting photographs while looking configured. Switching
    retention off is a decision and has to be spelled 0."""
    monkeypatch.setenv("BATCH_RETAIN_DAYS", "-1")

    with pytest.raises(config.ConfigError):
        config.env_float("BATCH_RETAIN_DAYS", 7.0, minimum=0)

    monkeypatch.setenv("BATCH_RETAIN_DAYS", "0")
    assert config.env_float("BATCH_RETAIN_DAYS", 7.0, minimum=0) == 0


@pytest.mark.parametrize("raw,expected", [
    ("1", True), ("true", True), ("TRUE", True), ("yes", True), ("on", True),
    ("0", False), ("false", False), ("no", False), ("off", False),
])
def test_booleans_accept_the_spellings_people_actually_type(monkeypatch, raw, expected):
    monkeypatch.setenv("SOME_FLAG", raw)
    assert config.env_bool("SOME_FLAG", not expected) is expected


def test_a_bool_that_is_neither_is_an_error_not_a_false(monkeypatch):
    """`BATCH_KEEP_INTERMEDIATES=True` is Python's own repr and an easy thing
    to paste. The old membership test read it as *off*, which is the direction
    that throws away someone's data while looking like it was configured."""
    monkeypatch.setenv("SOME_FLAG", "True")
    assert config.env_bool("SOME_FLAG", False) is True

    monkeypatch.setenv("SOME_FLAG", "sure")
    with pytest.raises(config.ConfigError):
        config.env_bool("SOME_FLAG", False)


# --- reporting -----------------------------------------------------------------

def test_the_registry_reports_where_each_value_came_from(monkeypatch):
    monkeypatch.setenv("FROM_ENV", "3")
    config.env_int("FROM_ENV", 1)
    config.env_int("FROM_DEFAULT", 1)
    config.set_flag("FROM_FLAG", True)

    sources = {s["name"]: s["source"] for s in config.describe()}
    assert sources == {"FROM_ENV": "env", "FROM_DEFAULT": "default", "FROM_FLAG": "flag"}


def test_a_cli_flag_overrides_what_the_banner_reports(monkeypatch):
    """Otherwise the banner confidently prints the environment's value while
    the process runs on the flag's -- the worst kind of diagnostic, the kind
    that is wrong."""
    monkeypatch.setenv("BOOTH_REQUIRE_CONSENT", "0")
    config.env_bool("BOOTH_REQUIRE_CONSENT", False)
    config.set_flag("BOOTH_REQUIRE_CONSENT", True)

    reported = {s["name"]: s["value"] for s in config.describe()}
    assert reported["BOOTH_REQUIRE_CONSENT"] is True


def test_the_banner_marks_overrides(monkeypatch):
    monkeypatch.setenv("FROM_ENV", "3")
    config.env_int("FROM_ENV", 1)
    config.env_int("FROM_DEFAULT", 1)

    banner = config.banner()
    assert "* FROM_ENV" in banner
    assert "* FROM_DEFAULT" not in banner, "an untouched default must not look configured"


def test_settings_are_sorted_so_two_booths_can_be_diffed():
    for name in ("ZZZ", "AAA", "MMM"):
        config.env_int(name, 1)
    names = [s["name"] for s in config.describe()]
    assert names == sorted(names)


# --- version -------------------------------------------------------------------

def test_version_info_always_answers(monkeypatch):
    """Including when there is no git and no build variable -- a Docker image
    has neither. An unknown build is a worse answer than a version, not a
    reason to have no answer."""
    monkeypatch.delenv("BOOTH_BUILD_SHA", raising=False)
    monkeypatch.setattr(config.subprocess, "run",
                        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("no git")))

    info = config.version_info()
    assert info["version"] == config.VERSION
    assert info["build"] == "unknown"


def test_an_injected_build_sha_wins(monkeypatch):
    """How a container gets one, since the image has no .git directory."""
    monkeypatch.setenv("BOOTH_BUILD_SHA", "deadbee")
    assert config.version_info()["build"] == "deadbee"
