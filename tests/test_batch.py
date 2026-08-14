"""
Tests for batch mode.

Two halves worth distinguishing. `batch.py` is pure bookkeeping over a
directory -- statuses, filenames, manifests, zipping -- and is tested directly.
The server half is tested through the real FastAPI routes with a faked
pipeline and backend, because the interesting behaviour is the routing
decision: a batch job's result must go to the run directory rather than to a
websocket session, and that choice lives in the relay, not in batch.py.

The failure cases carry most of the weight here. A batch is a long unattended
operation, so "one bad file among fifty" and "the run was deleted while a
frame was still rendering" are normal events, not edge cases.
"""

import io
import json
import zipfile

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import batch
import job_queue
import web_server


@pytest.fixture(autouse=True)
def clean_runs(tmp_path, monkeypatch):
    """Every run's files land in a tmp dir, and the registry starts empty --
    otherwise one test's half-finished run is visible to the next."""
    monkeypatch.setattr(batch, "BATCH_ROOT", tmp_path / "batch_runs")
    batch.RUNS.clear()
    web_server.JOBS.clear()
    yield
    batch.RUNS.clear()
    web_server.JOBS.clear()


def png_bytes(size=(64, 64), colour=(120, 120, 120)):
    buf = io.BytesIO()
    Image.new("RGB", size, colour).save(buf, format="PNG")
    return buf.getvalue()


# --- item bookkeeping ---------------------------------------------------------

def test_item_stems_sort_in_upload_order_and_survive_awkward_filenames():
    """The index prefix is load-bearing twice over: it keeps output in the
    order the operator supplied, and it disambiguates two cameras that both
    produced DSC_0001.jpg."""
    run = batch.create_run(["DSC_0001.jpg", "a photo (2).png", "DSC_0001.jpg"], None, None, None)
    stems = [i.stem for i in run.items]

    assert stems == sorted(stems), "stems must sort into upload order"
    assert len(set(stems)) == 3, "same-named files must not collide"
    for stem in stems:
        assert all(c.isalnum() or c in "-_" for c in stem), f"{stem} is not filesystem-safe"


def test_a_run_reports_its_own_progress():
    run = batch.create_run(["a.png", "b.png", "c.png"], "aurora", "coastline", "Coastal Morning")
    run.set_status(run.items[0], batch.DONE)
    run.set_status(run.items[1], batch.FAILED, "not an image")

    payload = run.to_dict()
    assert payload["counts"] == {"pending": 1, "analyzing": 0, "generating": 0, "done": 1, "failed": 1}
    assert payload["finished"] is False
    assert payload["look_label"] == "Coastal Morning"

    run.set_status(run.items[2], batch.DONE)
    assert run.to_dict()["finished"] is True, "failed items still count as finished"


def test_the_manifest_records_every_seed_and_survives_an_interrupted_run():
    """This is the file a client is handed with the frames, so it has to be
    written as the run goes rather than only at the end."""
    run = batch.create_run(["a.png", "b.png"], "aurora", "coastline", "Coastal Morning")
    run.items[0].provenance = {"seed": 4242, "prompt": "a coastline"}
    run.set_status(run.items[0], batch.DONE)
    run.write_manifest()

    manifest = json.loads((run.directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["brand_id"] == "aurora"
    assert manifest["items"][0]["provenance"]["seed"] == 4242
    assert manifest["items"][1]["status"] == batch.PENDING, \
        "an unfinished item should still appear, so the record is honest about what's missing"


# --- compositing ---------------------------------------------------------------

def test_the_subject_is_composited_back_unchanged(tmp_path):
    """The one invariant the whole app rests on: those pixels are a real
    photograph of a real person and must survive the round trip."""
    cutout = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    for x in range(8, 24):
        for y in range(8, 24):
            cutout.putpixel((x, y), (10, 200, 30, 255))
    path = tmp_path / "cutout.png"
    cutout.save(path)

    # Grading off: this test is about the subject's *geometry* surviving the
    # round trip. What a colour grade may and may not do to those pixels is
    # tests/test_finish.py's question, and asserting both here would mean
    # neither could be changed without breaking the other.
    result, applied = batch.composite_subject_over(
        Image.new("RGB", (32, 32), (200, 0, 0)), path, grade_strength=0)

    assert result.getpixel((16, 16)) == (10, 200, 30), "subject pixel was altered"
    assert result.getpixel((0, 0)) == (200, 0, 0), "background should show where the subject isn't"
    assert applied["logo"] is None, "no brand kit was supplied, so no mark should appear"


def test_a_size_mismatch_is_resized_rather_than_failing_the_item(tmp_path):
    """The workflow has no resize node so sizes normally match; if that ever
    changes, losing the whole frame would be a worse answer than a resize."""
    path = tmp_path / "cutout.png"
    Image.new("RGBA", (16, 16), (0, 255, 0, 255)).save(path)

    result, _ = batch.composite_subject_over(Image.new("RGB", (64, 64), (0, 0, 0)), path)

    assert result.size == (64, 64)


# --- zipping --------------------------------------------------------------------

def test_the_zip_contains_the_frames_and_the_manifest():
    run = batch.create_run(["a.png"], "aurora", "coastline", "Coastal Morning")
    run.path_for("output", run.items[0]).write_bytes(png_bytes())
    run.write_manifest()

    with zipfile.ZipFile(batch.zip_run(run)) as archive:
        names = archive.namelist()

    assert any(n.endswith("manifest.json") for n in names)
    assert any(n.endswith(".png") for n in names)
    assert all(n.startswith(f"{run.run_id}/") for n in names), \
        "entries should unpack into their own folder, not over the user's cwd"


def test_zipping_a_run_with_nothing_finished_still_produces_a_valid_archive():
    run = batch.create_run(["a.png"], None, None, None)
    with zipfile.ZipFile(batch.zip_run(run)) as archive:
        assert archive.testzip() is None


def test_deleting_a_run_removes_its_files():
    """Batch output includes photographs of people; a booth left running for a
    week should not quietly accumulate them."""
    run = batch.create_run(["a.png"], None, None, None)
    run.path_for("output", run.items[0]).write_bytes(png_bytes())
    directory = run.directory

    assert batch.delete_run(run.run_id) is True
    assert not directory.exists()
    assert batch.delete_run(run.run_id) is False, "deleting twice should report not-found"


# --- the HTTP surface -------------------------------------------------------------

class InlineQueue:
    """Stands in for the worker pool, running each job in the caller's loop.

    Not a shortcut: the real queue is started by the app's lifespan, which
    these tests skip on purpose (it warms up models and dials ComfyUI). What
    is under test here is the *routing* -- accept, reject, and where a
    finished frame goes -- and the pool itself already has its own suite in
    test_job_queue.py. Running inline also removes the only source of timing
    flakiness from these tests.
    """

    def __init__(self):
        self.jobs = []
        self.full = False

    async def submit(self, job):
        if self.full:
            raise job_queue.QueueFullError("the generation queue is full (test)")
        self.jobs.append(job)
        try:
            prompt_id, seed = job.submit(job.job_id)
        except Exception as exc:
            await web_server._on_job_failed(job, exc)
            raise
        await web_server._on_job_accepted(job, prompt_id, seed)
        return len(self.jobs)

    async def start(self): ...
    async def stop(self): ...
    async def drain(self): ...

    def stats(self):
        return {"waiting": 0, "running": 0, "submitted": len(self.jobs),
                "failed": 0, "workers": 1, "max_depth": 64}


@pytest.fixture
def client(monkeypatch):
    """The real routes, with the CV pipeline and ComfyUI faked out.

    lifespan is skipped (TestClient is used without its context manager) so
    the model warmup and the ComfyUI relay never start -- this is exercising
    request handling, not startup.
    """
    submitted = []
    queue = InlineQueue()
    monkeypatch.setattr(web_server, "GENERATION_QUEUE", queue)

    def fake_analyze(image):
        return {
            "image": image, "cutout": image.convert("RGBA"), "mask": image.convert("L"),
            "pose": image, "depth": image, "shadow": image.convert("RGBA"),
            "illumination": None, "suggested_controlnet_strength": 0.5,
        }

    class FakeBackend:
        POSITIONAL = ("image", "mask", "depth", "prompt", "controlnet_strength", "denoise")

        def queue_background_generation(self, *args, **kwargs):
            # Named here so a test can assert on the positive prompt, which
            # the real backend takes positionally.
            submitted.append({**dict(zip(self.POSITIONAL, args)), **kwargs})
            return f"prompt-{len(submitted)}", kwargs.get("seed") or 1

        def upload_image(self, *args, **kwargs):
            return "uploaded.png"

    monkeypatch.setattr(web_server.photoshoot_pipeline, "analyze", fake_analyze)
    monkeypatch.setattr(web_server, "backend", FakeBackend())
    test_client = TestClient(web_server.app)
    test_client.submitted = submitted
    test_client.queue = queue
    return test_client


def post_batch(client, count=2, **fields):
    files = [("files", (f"photo{i}.png", png_bytes(), "image/png")) for i in range(count)]
    # Consent is optional at the endpoint by default (see consent.py, and
    # tests/test_privacy.py where both modes are tested). Declared here anyway,
    # honestly -- these are synthetic images, which is exactly what
    # `internal_test` means -- so that these tests stay about batching.
    data = {"prompt": "a rooftop", "consent_basis": "internal_test",
            "consent_by": "test suite", **fields}
    return client.post("/api/batch", files=files, data=data)


def test_a_batch_is_accepted_and_reports_a_run_id(client):
    resp = post_batch(client, count=3)

    assert resp.status_code == 202, "the work isn't done yet, so 202 not 200"
    body = resp.json()
    assert body["total"] == 3
    assert body["run_id"] in batch.RUNS


def test_an_unreadable_file_fails_its_own_item_not_the_run(client):
    """One bad file among fifty is a normal event in a batch, not a reason to
    reject the other forty-nine."""
    files = [
        ("files", ("good.png", png_bytes(), "image/png")),
        ("files", ("notaphoto.txt", b"this is not an image at all", "image/png")),
    ]
    resp = client.post("/api/batch", files=files,
                       data={"prompt": "x", "consent_basis": "internal_test",
                             "consent_by": "test suite"})

    body = resp.json()
    assert body["counts"]["failed"] == 1
    assert body["queued"] == 1
    failed = next(i for i in body["items"] if i["status"] == "failed")
    assert "not a readable image" in failed["error"]


def test_too_many_photos_is_refused_with_the_limit(client):
    monkey_limit = web_server.MAX_BATCH_FILES
    files = [("files", (f"p{i}.png", png_bytes(), "image/png")) for i in range(monkey_limit + 1)]
    resp = client.post("/api/batch", files=files, data={"prompt": "x"})

    assert resp.status_code == 413
    assert str(monkey_limit) in resp.json()["detail"]


def test_an_empty_prompt_and_no_look_is_refused(client):
    """Same rule as the interactive path: there has to be something to
    generate, and the message should name both ways of providing it."""
    resp = post_batch(client, prompt="   ")
    assert resp.status_code == 400
    assert "look" in resp.json()["detail"]


def test_an_unknown_brand_is_refused(client):
    resp = post_batch(client, brand_id="not-installed")
    assert resp.status_code == 400


def test_status_and_download_404_for_an_unknown_run(client):
    assert client.get("/api/batch/nope").status_code == 404
    assert client.get("/api/batch/nope/download").status_code == 404
    assert client.delete("/api/batch/nope").status_code == 404


def test_downloading_before_anything_finished_is_a_409_not_an_empty_zip(client):
    run_id = post_batch(client).json()["run_id"]
    resp = client.get(f"/api/batch/{run_id}/download")

    assert resp.status_code == 409, "an empty zip would look like a successful empty shoot"


def test_a_finished_run_downloads_as_a_zip(client):
    run_id = post_batch(client, count=1).json()["run_id"]
    run = batch.RUNS[run_id]
    run.path_for("output", run.items[0]).write_bytes(png_bytes())
    run.set_status(run.items[0], batch.DONE)
    run.write_manifest()

    resp = client.get(f"/api/batch/{run_id}/download")

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(resp.content)) as archive:
        assert archive.testzip() is None


def test_a_full_queue_fails_the_photos_it_cannot_take_and_keeps_the_rest(client):
    """Backpressure has to arrive as a per-photo result, not a 500. Fifty
    photos against a bounded queue is the normal way batch mode meets its own
    limit, and the operator needs to know *which* frames didn't make it."""
    client.queue.full = True
    body = post_batch(client, count=2).json()

    assert body["queued"] == 0
    assert body["counts"]["failed"] == 2
    assert "full" in body["items"][0]["error"]
    assert body["run_id"] in batch.RUNS, "the run should survive so its failures are inspectable"


def test_the_brand_kit_is_enforced_on_the_batch_path_too(client):
    """The interactive path composes server-side so a client cannot drop a
    locked negative; batch mode has to go through the same gate, or the
    unattended path becomes the way around it."""
    brand = next(iter(web_server.BRANDS.values()))
    look = brand.looks[0]

    post_batch(client, count=1, brand_id=brand.id, look_id=look.id)

    call = client.submitted[0]
    assert brand.negative_suffix.split(",")[0].strip() in call["negative_prompt"]
    assert brand.positive_suffix.split(",")[0].strip() in call["prompt"]
    assert call["seed"] == web_server.brand_kit.locked_seed(brand.id, look.id), \
        "a batch is exactly where the locked seed matters -- the set must match"


def test_every_photo_in_a_locked_run_gets_the_same_seed(client):
    """The point of the feature, asserted directly."""
    brand = next(iter(web_server.BRANDS.values()))
    post_batch(client, count=3, brand_id=brand.id, look_id=brand.looks[0].id)

    seeds = {call["seed"] for call in client.submitted}
    assert len(seeds) == 1, f"a locked run produced {len(seeds)} different seeds"


# --- result routing ----------------------------------------------------------------

def test_a_batch_job_writes_its_frame_instead_of_pushing_to_a_session():
    """The routing decision batch mode exists on top of. A batch outlives the
    page that started it -- and may have had none -- so its result cannot be
    delivered over a websocket."""
    run = batch.create_run(["a.png"], "aurora", "coastline", "Coastal Morning")
    item = run.items[0]
    item.prompt_id = "p1"
    Image.new("RGBA", (32, 32), (0, 255, 0, 255)).save(run.path_for("cutout", item))

    job = {"batch_run_id": run.run_id, "provenance": {"seed": 99}, "kind": "background"}
    web_server._finish_batch_item(job, "p1", Image.new("RGB", (32, 32), (255, 0, 0)))

    assert item.status == batch.DONE
    assert item.provenance["seed"] == 99
    assert run.path_for("output", item).exists()


def test_a_frame_for_a_deleted_run_is_dropped_quietly():
    """Deleting a run mid-flight is a normal operator action; the in-flight
    frame then has nowhere to go and must not take the relay down with it."""
    run = batch.create_run(["a.png"], None, None, None)
    run.items[0].prompt_id = "p1"
    batch.delete_run(run.run_id)

    web_server._finish_batch_item(
        {"batch_run_id": run.run_id, "provenance": {}, "kind": "background"},
        "p1", Image.new("RGB", (8, 8)))


def test_a_failed_batch_generation_marks_its_item_rather_than_going_quiet():
    import asyncio

    run = batch.create_run(["a.png"], None, None, None)
    run.items[0].prompt_id = "p1"
    job = {"batch_run_id": run.run_id, "session_id": f"batch:{run.run_id}"}

    asyncio.run(web_server._report_job_error(job, "p1", "CUDA out of memory"))

    assert run.items[0].status == batch.FAILED
    assert "CUDA out of memory" in run.items[0].error


def test_a_missing_cutout_fails_the_item_instead_of_the_process():
    run = batch.create_run(["a.png"], None, None, None)
    item = run.items[0]
    item.prompt_id = "p1"
    # No cutout written -- as if analysis had been interrupted.

    web_server._finish_batch_item(
        {"batch_run_id": run.run_id, "provenance": {}, "kind": "background"},
        "p1", Image.new("RGB", (8, 8)))

    assert item.status == batch.FAILED
    assert "compositing failed" in item.error
