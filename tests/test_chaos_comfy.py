"""
The real ComfyUI client, driven against a ComfyUI that is failing on purpose.

test_resilience.py proves the retry ladder is correct in isolation. This file
answers the different question: does the *backend* survive a dependency having
a bad night -- with real HTTP, real status codes, and the reconciliation path
that only matters when a submission may or may not have landed.

Everything runs against `chaos_comfy.app` in-process through a TestClient, so
there is no server to start and no GPU anywhere near it.
"""

import random

import pytest
import requests
from fastapi.testclient import TestClient

import chaos_comfy
import resilience
from backends.comfy import ComfyBackend


@pytest.fixture
def chaos(monkeypatch, tmp_path):
    """A fake ComfyUI with the dice under the test's control."""
    monkeypatch.setattr(chaos_comfy, "STORE", tmp_path)
    (tmp_path / "input").mkdir(parents=True, exist_ok=True)
    (tmp_path / "output").mkdir(parents=True, exist_ok=True)
    chaos_comfy.HISTORY.clear()
    chaos_comfy.QUEUE.clear()
    chaos_comfy.SOCKETS.clear()
    chaos_comfy.SCRIPT.clear()
    for key in chaos_comfy.STATS:
        chaos_comfy.STATS[key] = 0
    monkeypatch.setattr(chaos_comfy, "FAILURE_RATE", 0.0)
    monkeypatch.setattr(chaos_comfy, "SLOW_RATE", 0.0)
    monkeypatch.setattr(chaos_comfy, "RENDER_SECONDS", 0.01)
    monkeypatch.setattr(chaos_comfy, "STEPS", 2)
    return TestClient(chaos_comfy.app)


def as_requests_response(httpx_response) -> requests.Response:
    """Converts the TestClient's httpx response into a real requests.Response.

    Not cosmetic. `raise_for_status()` on an httpx response raises
    `httpx.HTTPStatusError`, which `resilience.classify` has never heard of and
    would file under TERMINAL -- so a test written against raw httpx would
    "prove" that a 503 is not retried, which is the exact opposite of the
    behaviour in production. Keeping the production exception types is the
    whole point of the shim.
    """
    response = requests.Response()
    response.status_code = httpx_response.status_code
    response._content = httpx_response.content
    response.headers.update(httpx_response.headers)
    response.url = str(httpx_response.request.url)
    return response


@pytest.fixture
def backend(chaos, monkeypatch):
    """A real ComfyBackend whose HTTP calls land on the fake, with retry
    delays removed so the suite doesn't sleep through the backoff."""

    def shim(method):
        def call(url, **kwargs):
            kwargs.pop("timeout", None)
            path = "/" + url.split("://", 1)[-1].split("/", 1)[1] if "://" in url else url
            return as_requests_response(chaos.request(method, path, **kwargs))
        return call

    monkeypatch.setattr(requests, "post", shim("POST"))
    monkeypatch.setattr(requests, "get", shim("GET"))

    original = resilience.call
    monkeypatch.setattr(
        resilience, "call",
        lambda fn, **kw: original(fn, **{**kw, "sleep": lambda d: None,
                                         "rng": random.Random(0)}))
    return ComfyBackend(server_address="testserver")


# --- retry against real HTTP ---------------------------------------------------

def test_a_transient_503_on_upload_is_retried_and_succeeds(backend, chaos):
    from PIL import Image

    chaos_comfy.SCRIPT[:] = [503, 502, None]
    name = backend._upload_image(Image.new("RGB", (8, 8)), "subject")

    assert name.startswith("subject_")
    assert chaos_comfy.SCRIPT == [], "all three scripted outcomes should have been consumed"
    assert chaos_comfy.STATS["uploads"] == 1, "exactly one upload should have landed"


def test_a_retried_upload_sends_the_whole_png_not_a_truncated_one(backend, chaos):
    """The retry has to rewind the buffer. A half-consumed stream uploads a
    truncated PNG that ComfyUI accepts and then fails on much later, inside the
    graph, as an unrelated-looking decode error."""
    from PIL import Image

    chaos_comfy.SCRIPT[:] = [503, None]
    name = backend._upload_image(Image.new("RGB", (64, 64), (10, 200, 30)), "subject")

    uploaded = Image.open(chaos_comfy.STORE / "input" / name)
    assert uploaded.size == (64, 64)
    assert uploaded.convert("RGB").getpixel((32, 32)) == (10, 200, 30)


def test_a_400_is_not_retried(backend, chaos):
    from PIL import Image

    chaos_comfy.SCRIPT[:] = [400, 400, 400, 400]
    with pytest.raises(requests.exceptions.HTTPError):
        backend._upload_image(Image.new("RGB", (8, 8)), "subject")

    assert len(chaos_comfy.SCRIPT) == 3, "a bad request is just as bad the second time"


def test_the_breaker_opens_after_a_sustained_outage(backend, chaos):
    """And then reports it, rather than every later call paying for the full
    ladder to learn the same thing."""
    from PIL import Image

    chaos_comfy.SCRIPT[:] = [503] * 60
    for _ in range(3):
        with pytest.raises((requests.exceptions.HTTPError, resilience.CircuitOpenError)):
            backend._upload_image(Image.new("RGB", (8, 8)), "subject")

    assert backend.breaker.state == resilience.CircuitBreaker.OPEN

    remaining = len(chaos_comfy.SCRIPT)
    with pytest.raises(resilience.CircuitOpenError):
        backend._upload_image(Image.new("RGB", (8, 8)), "subject")
    assert len(chaos_comfy.SCRIPT) == remaining, \
        "an open breaker should not make the call at all"


# --- the reconciliation path ----------------------------------------------------

def test_a_prompt_that_landed_despite_a_timeout_is_not_submitted_twice(backend, chaos):
    """The duplicate-generation case, end to end.

    A read timeout on POST /prompt is genuinely ambiguous: ComfyUI may already
    be rendering. Because the app chooses the prompt_id, it can look instead of
    guess -- and must, since a duplicate burns a second slot on a serial GPU
    and bills a guest's wait twice.
    """
    prompt_id = "known-prompt-id"
    chaos_comfy.QUEUE.append([0, prompt_id, {}, {}, []])   # it did land

    submissions = {"n": 0}

    def timeout_once(url, **kwargs):
        submissions["n"] += 1
        raise requests.exceptions.ReadTimeout()

    import backends.comfy as comfy_module
    original_post = comfy_module.requests.post
    comfy_module.requests.post = timeout_once
    try:
        result = backend._queue_prompt({"1": {"class_type": "X", "inputs": {}}},
                                       prompt_id=prompt_id)
    finally:
        comfy_module.requests.post = original_post

    assert result == prompt_id
    assert submissions["n"] == 1, "reconciliation found it queued; nothing should be resubmitted"


def test_a_prompt_still_in_the_running_queue_counts_as_landed(backend, chaos):
    """A job queued one second ago is in neither history nor finished. Checking
    only /history would answer "no" for the most likely case and cause the
    duplicate this is meant to prevent."""
    prompt_id = "running-now"
    chaos_comfy.QUEUE.append([0, prompt_id, {}, {}, []])

    assert backend._prompt_landed(prompt_id) == prompt_id


def test_a_prompt_that_never_landed_reconciles_to_none(backend, chaos):
    assert backend._prompt_landed("never-existed") is None


def test_a_finished_prompt_is_found_in_history(backend, chaos):
    chaos_comfy.HISTORY["done-id"] = {"outputs": {}, "status": {"completed": True}}
    assert backend._prompt_landed("done-id") == "done-id"


# --- the fake's own fidelity -----------------------------------------------------

def test_the_fake_reproduces_comfyuis_one_socket_per_client_id_rule(chaos):
    """The bug that cost an afternoon: ComfyUI keeps one socket per clientId
    and silently drops the older one, so a second app instance sharing an id
    makes the first go permanently deaf -- its generations still run, it just
    never hears about them.

    Reproducing it here is what makes the app's fix (a per-process client id)
    a tested property rather than a comment.
    """
    with chaos.websocket_connect("/ws?clientId=shared") as first:
        first.receive_json()
        assert chaos_comfy.SOCKETS["shared"] is not None
        first_socket = chaos_comfy.SOCKETS["shared"]

        with chaos.websocket_connect("/ws?clientId=shared") as second:
            second.receive_json()
            assert chaos_comfy.SOCKETS["shared"] is not first_socket, \
                "the second connection should have displaced the first"

    with chaos.websocket_connect("/ws?clientId=a") as a, \
            chaos.websocket_connect("/ws?clientId=b") as b:
        a.receive_json()
        b.receive_json()
        assert set(chaos_comfy.SOCKETS) >= {"a", "b"}, "distinct ids must coexist"


def test_the_fake_serves_the_endpoints_the_app_depends_on(chaos):
    assert chaos.get("/system_stats").status_code == 200
    info = chaos.get("/object_info").json()
    assert "RealVisXL_V5.0_fp16.safetensors" in info["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
    assert chaos.get("/queue").json() == {"queue_running": [], "queue_pending": []}
    assert chaos.get("/history/nope").json() == {}


def test_injected_failures_use_retryable_status_codes(chaos, monkeypatch):
    """If the harness returned 400s, the app would correctly refuse to retry
    and the whole exercise would prove nothing."""
    monkeypatch.setattr(chaos_comfy, "FAILURE_RATE", 1.0)
    monkeypatch.setattr(chaos_comfy, "_rng", random.Random(7))

    statuses = {chaos.post("/prompt", json={"prompt": {}}).status_code for _ in range(15)}

    assert statuses, "the harness should have produced some failures"
    assert statuses <= {500, 502, 503}
    assert all(resilience.classify(_http_error(s)) is resilience.Verdict.RETRY for s in statuses)


def _http_error(status):
    response = requests.Response()
    response.status_code = status
    return requests.exceptions.HTTPError(response=response)


def test_a_seed_makes_the_failures_reproducible(chaos, monkeypatch):
    """A bug you can only reproduce one time in five is a bug you cannot fix."""
    monkeypatch.setattr(chaos_comfy, "FAILURE_RATE", 0.5)

    def run():
        monkeypatch.setattr(chaos_comfy, "_rng", random.Random(99))
        return [chaos.post("/prompt", json={"prompt": {}}).status_code for _ in range(12)]

    assert run() == run()
