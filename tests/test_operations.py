"""
Tests for the parts an operator or a container runtime talks to: the health
probes and the shutdown path.

Both are easy to write in a way that looks right and is actively harmful --
a liveness probe that restarts the process because a *dependency* is down, or
a "graceful" shutdown that is graceful about everything except the work
somebody was waiting for. Those two are the first tests here.
"""

import asyncio

import pytest
from fastapi.testclient import TestClient

import job_queue
import web_server
from tests.test_batch import InlineQueue, png_bytes


@pytest.fixture(autouse=True)
def restore_process_state():
    """These are the only tests that deliberately put the process into
    "shutting down", and those flags are module-level. Restoring them here
    rather than per-test: `_shutdown` clears them as its first action, so any
    test that calls it leaks a draining server into whatever runs next -- which
    it did, and four unrelated batch tests started returning 503."""
    yield
    web_server.MODELS_READY.set()
    web_server.READY.set()
    web_server.ACCEPTING.set()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(web_server, "GENERATION_QUEUE", InlineQueue())
    web_server.MODELS_READY.set()
    web_server.READY.set()
    web_server.ACCEPTING.set()
    return TestClient(web_server.app)


class FakeBreaker:
    def __init__(self, state="closed"):
        self._state = state

    def stats(self):
        return {"state": self._state, "failures": 0}


# --- liveness ------------------------------------------------------------------

def test_liveness_does_not_depend_on_comfyui(client, monkeypatch):
    """The most important assertion in this file. A liveness probe that
    consults a dependency is a restart loop waiting for that dependency to
    have a bad minute -- and restarting this process while ComfyUI is down
    destroys the queue, the attention items and every in-flight job, none of
    which would help."""
    monkeypatch.setattr(web_server.backend, "breaker", FakeBreaker("open"))

    resp = client.get("/healthz")

    assert resp.status_code == 200
    assert resp.json()["status"] == "alive"


def test_liveness_reports_the_build(client):
    body = client.get("/healthz").json()
    assert body["version"] and body["build"]


# --- readiness -----------------------------------------------------------------

def test_readiness_fails_while_the_models_are_still_loading(client):
    """~40s of rotoscope/pose/depth loading at startup, during which the
    process answers HTTP but every generation would block on the import lock.
    Answering "ready" there means a load balancer sends real guests into a
    hang."""
    web_server.MODELS_READY.clear()
    try:
        resp = client.get("/readyz")
    finally:
        web_server.MODELS_READY.set()

    assert resp.status_code == 503
    assert any("models" in reason for reason in resp.json()["reasons"])


def test_readiness_fails_with_an_open_breaker_and_says_why(client, monkeypatch):
    monkeypatch.setattr(web_server.backend, "breaker", FakeBreaker("open"))

    resp = client.get("/readyz")

    assert resp.status_code == 503
    assert any("ComfyUI" in reason for reason in resp.json()["reasons"])


def test_readiness_fails_while_draining(client):
    web_server.ACCEPTING.clear()

    resp = client.get("/readyz")

    assert resp.status_code == 503
    assert any("draining" in reason for reason in resp.json()["reasons"])


def test_a_ready_server_answers_200_with_no_reasons(client, monkeypatch):
    monkeypatch.setattr(web_server.backend, "breaker", FakeBreaker("closed"))

    resp = client.get("/readyz")

    assert resp.status_code == 200
    assert resp.json() == {**resp.json(), "ready": True, "reasons": []}


def test_readiness_carries_what_an_operator_would_ask_next(client, monkeypatch):
    """A probe that fails without saying why turns into somebody reading
    source at 2am."""
    monkeypatch.setattr(web_server.backend, "breaker", FakeBreaker("closed"))

    body = client.get("/readyz").json()

    assert body["queue"]["max_depth"] > 0
    assert body["breaker"]["state"] == "closed"
    assert body["version"]
    assert any(s["name"] == "BATCH_RETAIN_DAYS" for s in body["settings"])


# --- refusing work while draining ----------------------------------------------

def test_a_draining_server_refuses_a_batch_before_reading_the_uploads(client, tmp_path,
                                                                     monkeypatch):
    """503 before a single byte is written. A draining server that accepted
    fifty photographs and then dropped them would be worse than one that never
    took them -- and it would put them on disk on the way."""
    monkeypatch.setattr(web_server.batch, "BATCH_ROOT", tmp_path / "runs")
    web_server.ACCEPTING.clear()

    resp = client.post("/api/batch",
                       files=[("files", ("a.png", png_bytes(), "image/png"))],
                       data={"prompt": "a rooftop"})

    assert resp.status_code == 503
    root = tmp_path / "runs"
    assert not root.exists() or not any(root.iterdir())


# --- shutdown ------------------------------------------------------------------

def build_queue(submitted, run_blocking=None):
    async def on_accepted(job, prompt_id, seed):
        submitted.append(prompt_id)

    async def on_failed(job, exc):
        submitted.append(("failed", job.job_id))

    return job_queue.InProcessJobQueue(
        on_accepted=on_accepted, on_failed=on_failed, workers=1,
        run_blocking=run_blocking or _inline)


async def _inline(fn, *args):
    return fn(*args)


async def _slow(fn, *args):
    """Stands in for the real submit, which is ~14s of rotoscope and three
    uploads. Slow enough that jobs are genuinely still queued when the drain
    starts, which is the only condition under which the drain is being tested
    at all."""
    await asyncio.sleep(0.02)
    return fn(*args)


def test_shutdown_drains_queued_work_instead_of_cancelling_it(monkeypatch):
    """The bug this replaced: `stop()` cancels the worker tasks, so anything
    still waiting was dropped without so much as an error frame. `docker stop`
    during a batch lost a guest's photo and told nobody.

    The submit is *slow* on purpose. An instant one lets every job finish
    before the drain is even reached, so the test passes whether or not the
    drain does anything -- which is exactly how the first version of this test
    was written, and it would have gone green against the old `stop()`.
    """
    submitted = []
    queue = build_queue(submitted, run_blocking=_slow)
    monkeypatch.setattr(web_server, "GENERATION_QUEUE", queue)

    async def scenario():
        await queue.start()
        for i in range(6):
            await queue.submit(job_queue.GenerationJob(
                session_id="s", kind="background", job_id=f"p{i}",
                submit=lambda prompt_id: (prompt_id, 1)))
        assert queue.stats()["waiting"] == 6, "nothing should have drained yet"
        await web_server._shutdown([])

    asyncio.run(scenario())

    assert sorted(submitted) == [f"p{i}" for i in range(6)], \
        "every accepted job should have reached ComfyUI before the process exited"


def test_shutdown_stops_accepting_before_it_starts_tearing_down(monkeypatch):
    """Order matters more than speed here: readiness has to go false, and
    admission has to close, *before* anything is dismantled -- otherwise a
    request that arrives during the drain is admitted into a dying process."""
    seen = {}

    class Recorder(InlineQueue):
        async def drain(self):
            seen["accepting_during_drain"] = web_server.ACCEPTING.is_set()
            seen["ready_during_drain"] = web_server.READY.is_set()

        async def stop(self):
            pass

    monkeypatch.setattr(web_server, "GENERATION_QUEUE", Recorder())
    web_server.ACCEPTING.set()
    web_server.READY.set()

    asyncio.run(web_server._shutdown([]))

    assert seen == {"accepting_during_drain": False, "ready_during_drain": False}


def test_a_drain_that_cannot_finish_fails_the_stragglers_loudly(monkeypatch):
    """A bounded drain, because a shutdown that hangs teaches an operator to
    reach for `kill -9`, which loses everything the drain was protecting. Past
    the deadline the remaining work is *failed* -- the browser is told -- not
    silently dropped, which is the whole difference from the old behaviour."""
    told = []

    class NeverDrains(InlineQueue):
        async def drain(self):
            await asyncio.sleep(60)

        async def stop(self):
            pass

        def stats(self):
            return {"waiting": 3, "running": 0, "submitted": 0, "failed": 0,
                    "workers": 1, "max_depth": 64}

    monkeypatch.setattr(web_server, "GENERATION_QUEUE", NeverDrains())
    monkeypatch.setattr(web_server, "SHUTDOWN_DRAIN_SECONDS", 0.05)
    monkeypatch.setattr(web_server, "send_json_to",
                        lambda session_id, payload: _capture(told, payload))
    web_server.JOBS["p1"] = {"session_id": "s1", "kind": "background", "provenance": None}
    try:
        asyncio.run(web_server._shutdown([]))
    finally:
        web_server.JOBS.clear()

    assert [m["type"] for m in told] == ["error"]
    assert "shut down" in told[0]["message"]


async def _capture(sink, payload):
    sink.append(payload)
