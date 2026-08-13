"""
Tests for the operator attention queue.

The mechanism is a dict. The value is entirely in the criteria, so that is what
these test: that a dead dependency reads as *one* problem rather than fifty,
that an alert closes itself when its cause goes away, and that things an
operator cannot act on never reach the queue at all.

An alert nobody can act on is worse than no alert, because it teaches people to
ignore the ones that matter -- so "this must NOT raise an item" is as much a
requirement here as "this must".
"""

import pytest
from fastapi.testclient import TestClient

import attention
import batch
import web_server
from tests.test_batch import InlineQueue, png_bytes


@pytest.fixture(autouse=True)
def clean(tmp_path, monkeypatch):
    attention.clear()
    monkeypatch.setattr(batch, "BATCH_ROOT", tmp_path / "batch_runs")
    batch.RUNS.clear()
    web_server.JOBS.clear()
    yield
    attention.clear()


# --- deduplication ---------------------------------------------------------------

def test_fifty_identical_failures_are_one_item_with_a_count():
    """A fifty-photo batch against a dead ComfyUI is one problem. Fifty lines
    would bury everything else in the panel, which is how a queue stops being
    read."""
    for i in range(50):
        attention.raise_item(attention.BATCH_ITEM_FAILED,
                             "batch run abc: photo(s) failed to generate",
                             detail=f"item {i} timed out")

    items = attention.open_items()
    assert len(items) == 1
    assert items[0].count == 50
    assert "item 49" in items[0].detail, "the newest detail is the one they'll investigate"


def test_different_problems_stay_separate():
    attention.raise_item(attention.DEPENDENCY_DOWN, "ComfyUI is not responding")
    attention.raise_item(attention.SUBJECT_NOT_FOUND, "no subject found in a photo")

    assert len(attention.open_items()) == 2


def test_a_resolved_item_does_not_absorb_a_new_occurrence():
    """Otherwise a problem that recurs after being handled would silently
    increment a closed row and never be seen again."""
    first = attention.raise_item(attention.DEPENDENCY_DOWN, "ComfyUI is not responding")
    attention.resolve(first.id)

    second = attention.raise_item(attention.DEPENDENCY_DOWN, "ComfyUI is not responding")

    assert second.id != first.id
    assert len(attention.open_items()) == 1


# --- severity and ordering ---------------------------------------------------------

def test_the_worst_thing_is_listed_first():
    attention.raise_item(attention.BATCH_ITEM_FAILED, "one frame failed")
    attention.raise_item(attention.DEPENDENCY_DOWN, "ComfyUI is not responding")
    attention.raise_item(attention.SUBJECT_NOT_FOUND, "no subject found")

    kinds = [i.kind for i in attention.open_items()]
    assert kinds[0] == attention.DEPENDENCY_DOWN
    assert kinds[-1] == attention.BATCH_ITEM_FAILED


def test_a_dead_dependency_outranks_a_single_bad_frame():
    """One is "the booth is down", the other is "this guest was unlucky"."""
    assert attention.SEVERITY[attention.DEPENDENCY_DOWN] == "high"
    assert attention.SEVERITY[attention.BATCH_ITEM_FAILED] == "low"


# --- resolution -------------------------------------------------------------------

def test_resolving_records_who_did_it():
    item = attention.raise_item(attention.GENERATION_FAILED, "a generation failed")

    resolved = attention.resolve(item.id, by="bernardo")

    assert resolved.resolved_by == "bernardo"
    assert attention.open_items() == []


def test_resolving_something_twice_reports_that_it_was_already_handled():
    item = attention.raise_item(attention.GENERATION_FAILED, "a generation failed")
    attention.resolve(item.id)

    assert attention.resolve(item.id) is None


def test_the_app_closes_a_dependency_alert_when_the_dependency_recovers():
    """An alert that outlives its cause is how an operator learns to distrust
    the panel entirely."""
    attention.raise_item(attention.DEPENDENCY_DOWN, "ComfyUI is not responding")
    attention.raise_item(attention.GENERATION_FAILED, "a generation failed")

    closed = attention.resolve_kind(attention.DEPENDENCY_DOWN, by="recovered")

    assert closed == 1
    kinds = [i.kind for i in attention.open_items()]
    assert kinds == [attention.GENERATION_FAILED], \
        "recovery should not sweep away unrelated problems"


# --- bounds ------------------------------------------------------------------------

def test_the_queue_is_bounded(monkeypatch):
    """An unbounded list in a long-running booth is a memory leak whose growth
    rate is the app's error rate -- the worst possible time to run out of RAM."""
    monkeypatch.setattr(attention, "MAX_OPEN", 10)

    for i in range(40):
        attention.raise_item(attention.GENERATION_FAILED, f"failure number {i}")

    assert len(attention.ITEMS) <= 10


def test_resolved_items_are_dropped_before_open_ones(monkeypatch):
    monkeypatch.setattr(attention, "MAX_OPEN", 5)
    keep = attention.raise_item(attention.DEPENDENCY_DOWN, "important, still open")
    for i in range(4):
        attention.resolve(attention.raise_item(attention.GENERATION_FAILED, f"old {i}").id)

    for i in range(5):
        attention.raise_item(attention.GENERATION_FAILED, f"new {i}")

    assert keep.id in attention.ITEMS, "an open high-severity item should survive the trim"


# --- what must NOT reach the queue ----------------------------------------------------

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


def test_a_guests_bad_upload_does_not_page_an_operator(client):
    """There is nothing for a human to do about a .txt file: the uploader is
    already being told. A queue that fills with noise is a queue nobody reads.
    """
    files = [("files", ("notaphoto.txt", b"this is not an image", "image/png"))]
    client.post("/api/batch", files=files,
                data={"prompt": "x", "consent_basis": "guest_verbal",
                      "consent_by": "op"})

    assert attention.open_items() == []


def test_a_rejected_consent_does_not_page_an_operator(client):
    """The operator is the one being told, in the response."""
    client.post("/api/batch",
                files=[("files", ("a.png", png_bytes(), "image/png"))],
                data={"prompt": "x"})

    assert attention.open_items() == []


# --- the HTTP surface -------------------------------------------------------------------

def test_the_endpoint_reports_what_needs_a_person(client):
    attention.raise_item(attention.DEPENDENCY_DOWN, "ComfyUI is not responding",
                         detail="connection refused")

    body = client.get("/api/attention").json()

    assert body["open"] == 1
    assert body["highest_severity"] == "high"
    assert body["items"][0]["summary"] == "ComfyUI is not responding"


def test_an_empty_queue_reports_cleanly(client):
    body = client.get("/api/attention").json()
    assert body == {"open": 0, "highest_severity": None, "items": []}


def test_an_item_can_be_resolved_over_http(client):
    item = attention.raise_item(attention.GENERATION_FAILED, "a generation failed")

    resp = client.post(f"/api/attention/{item.id}/resolve", data={"by": "bernardo"})

    assert resp.status_code == 200
    assert resp.json()["resolved_by"] == "bernardo"
    assert client.get("/api/attention").json()["open"] == 0


def test_resolving_an_unknown_item_is_a_404(client):
    assert client.post("/api/attention/9999/resolve").status_code == 404
