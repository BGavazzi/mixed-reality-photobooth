"""
Tests for the three promises the booth makes about the people it photographs:
we record why we were allowed to, we keep as little as possible, and we do not
keep it forever.

These are the tests most worth having teeth, because every failure here is
silent. A retention sweep that quietly does nothing looks exactly like one that
works, right up until someone asks what is on the disk.
"""

import io
import json
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import batch
import consent
import web_server
from tests.test_batch import InlineQueue, png_bytes   # the fixtures' machinery


@pytest.fixture(autouse=True)
def clean(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "BATCH_ROOT", tmp_path / "batch_runs")
    monkeypatch.setattr(batch, "KEEP_INTERMEDIATES", False)
    monkeypatch.setattr(batch, "DEFAULT_RETAIN_DAYS", 7)
    batch.RUNS.clear()
    web_server.JOBS.clear()
    yield
    batch.RUNS.clear()


# --- consent -------------------------------------------------------------------

def test_by_default_a_run_that_declares_nothing_is_allowed_and_says_so():
    """The gate is opt-in (see consent.py). What it must not do when it is off
    is leave the field blank: "nobody recorded this" and "this manifest predates
    the field" have to read differently to whoever opens it later."""
    record = consent.parse("", "")

    assert record.basis == consent.NOT_RECORDED
    assert record.recorded is False
    assert record.to_dict()["description"], \
        "an absent record still needs to explain itself in the manifest"


def test_with_enforcement_on_an_empty_declaration_is_refused_with_the_options():
    with pytest.raises(consent.ConsentError) as err:
        consent.parse("", "", required=True)
    assert "guest_verbal" in str(err.value), "the error should name what to pass"


def test_a_half_filled_declaration_is_refused_even_with_the_gate_off():
    """The one case worth refusing in both modes: a basis with nobody's name on
    it looks like a record and cannot be followed up, and a name with no basis
    records nothing at all."""
    with pytest.raises(consent.ConsentError):
        consent.parse("guest_verbal", "", required=False)
    with pytest.raises(consent.ConsentError):
        consent.parse("", "Bernardo", required=False)


def test_the_module_default_is_what_the_flag_says():
    """`required=None` reads consent.REQUIRED at call time, so flipping the
    flag at startup changes every caller rather than the ones that remembered
    to pass it."""
    original = consent.REQUIRED
    try:
        consent.REQUIRED = True
        with pytest.raises(consent.ConsentError):
            consent.parse("", "")
        consent.REQUIRED = False
        assert consent.parse("", "").basis == consent.NOT_RECORDED
    finally:
        consent.REQUIRED = original


def test_an_invented_basis_is_refused():
    """A closed set, not free text: "consent: yes" records that somebody typed
    something, which answers nothing when a reviewer asks what people were
    told."""
    with pytest.raises(consent.ConsentError):
        consent.parse("sure_why_not", "Bernardo")


def test_consent_must_name_a_person_not_just_a_basis():
    """A record with nobody's name on it cannot be followed up when a guest
    later asks for their photographs to be deleted."""
    with pytest.raises(consent.ConsentError) as err:
        consent.parse("guest_verbal", "   ")
    assert "who recorded" in str(err.value)


def test_a_valid_record_carries_the_description_not_just_the_key():
    record = consent.parse("event_notice", "Bernardo", note="wristband desk")

    payload = record.to_dict()
    assert payload["basis"] == "event_notice"
    assert "notice" in payload["description"].lower(), \
        "the manifest should be readable without this codebase in hand"
    assert payload["recorded_by"] == "Bernardo"
    assert payload["recorded_at"] > 0


def test_the_options_the_ui_offers_are_the_ones_the_server_accepts():
    """Served from one place so the two lists cannot drift apart."""
    for option in consent.options():
        assert consent.parse(option["id"], "someone").basis == option["id"]


# --- retention -------------------------------------------------------------------

def test_a_run_past_its_retention_window_is_swept():
    run = batch.create_run(["a.png"], None, None, None, retain_days=1)
    run.path_for("output", run.items[0]).write_bytes(png_bytes())
    directory = run.directory

    removed = batch.sweep_expired(now=time.time() + 2 * 86400)

    assert run.run_id in removed
    assert not directory.exists()
    assert run.run_id not in batch.RUNS


def test_a_run_inside_its_window_is_left_alone():
    run = batch.create_run(["a.png"], None, None, None, retain_days=7)
    assert batch.sweep_expired(now=time.time() + 86400) == []
    assert run.directory.exists()


def test_an_orphaned_directory_from_a_crashed_run_is_still_swept():
    """The case that matters most. A server that died mid-evening leaves a
    folder of photographs that nothing in the app remembers -- so nothing
    would ever delete it, and it is the oldest data on the disk."""
    orphan = batch.BATCH_ROOT / "orphan_run"
    orphan.mkdir(parents=True)
    (orphan / "input").mkdir()
    (orphan / "input" / "someone.orig").write_bytes(png_bytes())
    import os
    old = time.time() - 30 * 86400
    os.utime(orphan, (old, old))

    removed = batch.sweep_expired(retain_days=7)

    assert "orphan_run" in removed
    assert not orphan.exists()


def test_an_orphan_with_an_unreadable_manifest_is_still_aged_out():
    """Missing or corrupt paperwork is not a reason to keep someone's face
    forever -- it is a symptom of the interrupted run that needs sweeping."""
    orphan = batch.BATCH_ROOT / "bad_manifest"
    orphan.mkdir(parents=True)
    (orphan / "manifest.json").write_text("{not json", encoding="utf-8")
    import os
    old = time.time() - 30 * 86400
    os.utime(orphan, (old, old))

    assert "bad_manifest" in batch.sweep_expired(retain_days=7)


def test_the_manifests_own_timestamp_wins_over_the_directory_mtime():
    """A directory touched by a later zip download should not have its clock
    reset -- retention runs from when the photos were taken."""
    orphan = batch.BATCH_ROOT / "with_manifest"
    orphan.mkdir(parents=True)
    (orphan / "manifest.json").write_text(
        json.dumps({"created_at": time.time() - 30 * 86400}), encoding="utf-8")

    assert "with_manifest" in batch.sweep_expired(retain_days=7)


def test_retention_can_be_switched_off_but_only_deliberately():
    batch.create_run(["a.png"], None, None, None, retain_days=1)
    assert batch.sweep_expired(now=time.time() + 999 * 86400, retain_days=0) == []


def test_a_run_reports_its_own_expiry_so_the_operator_can_see_it():
    run = batch.create_run(["a.png"], None, None, None, retain_days=2)
    payload = run.to_dict()
    assert payload["expires_at"] == pytest.approx(payload["created_at"] + 2 * 86400)


# --- minimisation ------------------------------------------------------------------

def test_the_original_photograph_is_deleted_once_a_cutout_exists():
    """Nothing downstream reads it again: the compositor needs the cutout, not
    the photograph."""
    run = batch.create_run(["guest.jpg"], None, None, None)
    item = run.items[0]
    original = run.path_for("input", item, ".orig")
    original.write_bytes(png_bytes())
    run.path_for("cutout", item).write_bytes(png_bytes())

    run.drop_original(item)

    assert not original.exists()
    assert run.path_for("cutout", item).exists(), "the cutout is still needed"


def test_keeping_intermediates_is_possible_but_must_be_asked_for(monkeypatch):
    monkeypatch.setattr(batch, "KEEP_INTERMEDIATES", True)
    run = batch.create_run(["guest.jpg"], None, None, None)
    item = run.items[0]
    original = run.path_for("input", item, ".orig")
    original.write_bytes(png_bytes())

    run.drop_original(item)

    assert original.exists()


def test_dropping_an_original_twice_is_harmless():
    """It runs from a worker thread on a path that can be retried."""
    run = batch.create_run(["guest.jpg"], None, None, None)
    run.drop_original(run.items[0])
    run.drop_original(run.items[0])


def test_a_zip_of_a_finished_run_contains_no_original_photographs():
    """The end-to-end version of the promise: what leaves the building is
    composited frames and a manifest, not the source photographs."""
    import zipfile

    run = batch.create_run(["guest.jpg"], None, None, None,
                           consent=consent.parse("guest_verbal", "op").to_dict())
    item = run.items[0]
    run.path_for("input", item, ".orig").write_bytes(png_bytes())
    run.path_for("cutout", item).write_bytes(png_bytes())
    run.path_for("output", item).write_bytes(png_bytes())
    run.drop_original(item)
    run.write_manifest()

    with zipfile.ZipFile(batch.zip_run(run)) as archive:
        names = archive.namelist()

    assert not any("input" in n or ".orig" in n for n in names)
    assert not any("cutout" in n for n in names)
    assert sum(n.endswith(".png") for n in names) == 1


# --- what the manifest records -------------------------------------------------------

def test_the_manifest_records_consent_and_retention():
    """The two questions a reviewer asks first, and that no amount of seed
    provenance answers."""
    run = batch.create_run(
        ["a.png"], "aurora", "rooftop", "Rooftop Social",
        consent=consent.parse("guest_signed", "Bernardo", "waiver at door").to_dict(),
        retain_days=3)
    run.write_manifest()

    manifest = json.loads((run.directory / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["consent"]["basis"] == "guest_signed"
    assert manifest["consent"]["recorded_by"] == "Bernardo"
    assert manifest["retention"]["retain_days"] == 3
    assert manifest["retention"]["originals_kept"] is False


# --- the HTTP surface -----------------------------------------------------------------

@pytest.fixture
def client(monkeypatch):
    def fake_analyze(image):
        return {"image": image, "cutout": image.convert("RGBA"), "mask": image.convert("L"),
                "pose": image, "depth": image, "shadow": image.convert("RGBA"),
                "illumination": None, "suggested_controlnet_strength": 0.5}

    class FakeBackend:
        def queue_background_generation(self, *args, **kwargs):
            return kwargs.get("prompt_id") or "p1", kwargs.get("seed") or 1

    monkeypatch.setattr(web_server.photoshoot_pipeline, "analyze", fake_analyze)
    monkeypatch.setattr(web_server, "backend", FakeBackend())
    monkeypatch.setattr(web_server, "GENERATION_QUEUE", InlineQueue())
    return TestClient(web_server.app)


def post_batch(client, **fields):
    files = [("files", ("photo.png", png_bytes(), "image/png"))]
    data = {"prompt": "a rooftop", "consent_basis": "guest_verbal",
            "consent_by": "Bernardo", **fields}
    return client.post("/api/batch", files=files, data=data)


def test_the_api_refuses_a_half_filled_consent_form(client):
    resp = post_batch(client, consent_basis="")

    assert resp.status_code == 400
    assert "consent" in resp.json()["detail"].lower()
    assert batch.RUNS == {}, "nothing should have been written to disk"


def test_the_api_accepts_a_batch_with_no_consent_declaration(client):
    """The default. The run is not blocked, and the manifest says plainly that
    nothing was declared rather than implying something was."""
    resp = post_batch(client, consent_basis="", consent_by="")

    assert resp.status_code == 202
    assert resp.json()["consent"]["basis"] == consent.NOT_RECORDED


def test_the_api_refuses_an_empty_declaration_when_enforcement_is_on(client, monkeypatch):
    monkeypatch.setattr(consent, "REQUIRED", True)

    resp = post_batch(client, consent_basis="", consent_by="")

    assert resp.status_code == 400
    assert batch.RUNS == {}, "nothing should have been written to disk"


def test_consent_is_checked_before_any_photograph_is_written(client, tmp_path):
    """Order matters: a run about to be rejected should not first spend thirty
    seconds putting fifty strangers' photographs on disk."""
    post_batch(client, consent_by="")

    root = batch.BATCH_ROOT
    assert not root.exists() or not any(root.iterdir())


def test_an_accepted_batch_carries_its_consent_into_the_response(client):
    body = post_batch(client).json()

    assert body["consent"]["basis"] == "guest_verbal"
    assert body["consent"]["recorded_by"] == "Bernardo"
    assert body["expires_at"] > body["created_at"]


def test_config_advertises_the_consent_bases_and_retention(client):
    config = client.get("/api/config").json()

    assert {o["id"] for o in config["consent_bases"]} == set(consent.BASES)
    assert consent.NOT_RECORDED not in {o["id"] for o in config["consent_bases"]}, \
        "the absence of a basis is not a basis an operator can pick"
    assert config["consent_required"] is consent.REQUIRED, \
        "whether the form blocks is the server's call, not the page's guess"
    assert config["retention"]["retain_days"] == batch.DEFAULT_RETAIN_DAYS
