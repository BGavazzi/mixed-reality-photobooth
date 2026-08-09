"""
Tests for web_server.py's multi-session job routing, against a fake backend.

This is the concurrency logic the multi-session refactor introduced: ComfyUI
events carry a prompt_id, and JOBS maps that back to the session that queued
it, so two browser tabs generating at once each get *their own* result
instead of racing a single global "whoever connected last."

verify_multi_session.py already proves this end to end -- but only on a
machine with ComfyUI running, a GPU, and two subject photos, taking several
minutes of real generation per run. Everything below exercises the same
routing decisions in milliseconds with no GPU, which is what makes it
possible to check the cross-session-leak cases that are awkward to stage for
real: a result arriving for a session that already disconnected, a submit
failure rolling back its own bookkeeping, a relay drop stranding jobs.
"""

import asyncio
import json
import struct

import pytest
from PIL import Image

import web_server


class FakeWebSocket:
    """Captures what the server would have sent to one browser tab."""

    def __init__(self):
        self.json_messages = []
        self.binary_messages = []

    async def send_json(self, payload):
        self.json_messages.append(payload)

    async def send_bytes(self, data):
        self.binary_messages.append(data)

    def types(self):
        return [m["type"] for m in self.json_messages]

    def of_type(self, message_type):
        return [m for m in self.json_messages if m["type"] == message_type]


@pytest.fixture(autouse=True)
def clean_server_state():
    """SESSIONS/JOBS/executing_prompt_id are module-level state, so every
    test starts from empty and leaves nothing behind for the next one."""
    web_server.SESSIONS.clear()
    web_server.JOBS.clear()
    web_server.executing_prompt_id = None
    yield
    web_server.SESSIONS.clear()
    web_server.JOBS.clear()
    web_server.executing_prompt_id = None


@pytest.fixture
def two_sessions():
    sockets = {"a": FakeWebSocket(), "b": FakeWebSocket()}
    web_server.SESSIONS["session-a"] = sockets["a"]
    web_server.SESSIONS["session-b"] = sockets["b"]
    return sockets


def queue_fake_job(session_id, kind="background", seed=1234):
    """Runs the real _queue_job() with a submit that always succeeds."""
    asyncio.run(web_server._queue_job(
        session_id, kind,
        lambda prompt_id: (prompt_id, seed),
        {"prompt": f"prompt for {session_id}", "controlnet": "depth.safetensors",
         "controlnet_strength": 0.75, "denoise": 0.85},
    ))
    return next(pid for pid, job in web_server.JOBS.items() if job["session_id"] == session_id)


def comfy_event(event_type, **data):
    return json.dumps({"type": event_type, "data": data})


# --- registration -----------------------------------------------------------

def test_queued_job_is_registered_against_its_own_session(two_sessions):
    prompt_id_a = queue_fake_job("session-a")
    prompt_id_b = queue_fake_job("session-b")

    assert prompt_id_a != prompt_id_b
    assert web_server.JOBS[prompt_id_a]["session_id"] == "session-a"
    assert web_server.JOBS[prompt_id_b]["session_id"] == "session-b"
    assert two_sessions["a"].of_type("queued")[0]["prompt_id"] == prompt_id_a
    assert two_sessions["b"].of_type("queued")[0]["prompt_id"] == prompt_id_b


def test_provenance_records_the_real_seed_and_model(two_sessions):
    prompt_id = queue_fake_job("session-a", seed=987654)
    provenance = web_server.JOBS[prompt_id]["provenance"]

    assert provenance["seed"] == 987654
    assert provenance["kind"] == "background"
    assert provenance["checkpoint"] == web_server.CHECKPOINT_NAME
    assert provenance["generated_at"].endswith("+00:00"), "timestamps must be timezone-aware UTC"


def test_failed_submission_rolls_back_its_job_entry(two_sessions):
    """A pre-registered JOBS entry whose submit then fails would otherwise
    sit there for the life of the process -- and the browser would never
    hear why its generation never started."""
    def failing_submit(prompt_id):
        raise ConnectionError("ComfyUI unreachable")

    asyncio.run(web_server._queue_job("session-a", "image", failing_submit, {"prompt": "x"}))

    assert web_server.JOBS == {}, "the pre-registered entry must be rolled back"
    assert two_sessions["a"].types() == ["error"]
    assert "ComfyUI unreachable" in two_sessions["a"].json_messages[0]["message"]
    assert two_sessions["b"].json_messages == [], "the other session must not be told anything"


# --- event routing ----------------------------------------------------------

def test_progress_reaches_only_the_session_that_queued_the_job(two_sessions):
    prompt_id_a = queue_fake_job("session-a")
    queue_fake_job("session-b")

    asyncio.run(web_server.handle_comfy_message(
        comfy_event("progress", prompt_id=prompt_id_a, value=7, max=30)))

    assert two_sessions["a"].of_type("progress") == [{"type": "progress", "value": 7, "max": 30}]
    assert two_sessions["b"].of_type("progress") == [], "cross-session leak"


def test_completed_job_delivers_its_own_result_and_provenance(monkeypatch, two_sessions):
    prompt_id_a = queue_fake_job("session-a")
    queue_fake_job("session-b")
    monkeypatch.setattr(web_server.backend, "get_result_image",
                        lambda pid: Image.new("RGB", (8, 8), (10, 20, 30)))

    asyncio.run(web_server.handle_comfy_message(
        comfy_event("executing", prompt_id=prompt_id_a, node=None)))

    done = two_sessions["a"].of_type("done")
    assert len(done) == 1
    assert done[0]["provenance"]["prompt"] == "prompt for session-a"
    assert done[0]["image_base64"]
    assert two_sessions["b"].of_type("done") == []
    assert prompt_id_a not in web_server.JOBS, "a finished job must not stay in JOBS"


def test_unretrievable_result_reports_an_error_instead_of_going_quiet(monkeypatch, two_sessions):
    """If /history can't produce the image, the browser must be told. Both a
    raised exception and a None return used to end the same way: JOBS
    cleaned up, nothing sent, and a permanently disabled Generate button."""
    prompt_id = queue_fake_job("session-a")
    monkeypatch.setattr(web_server.backend, "get_result_image", lambda pid: None)

    asyncio.run(web_server.handle_comfy_message(
        comfy_event("executing", prompt_id=prompt_id, node=None)))

    assert two_sessions["a"].types() == ["queued", "error"]
    assert prompt_id not in web_server.JOBS


def test_result_fetch_raising_does_not_escape_the_relay(monkeypatch, two_sessions):
    """An exception here used to propagate into comfy_relay_loop's `async
    for`, tearing down the ComfyUI websocket for every other session too."""
    prompt_id = queue_fake_job("session-a")

    def boom(pid):
        raise RuntimeError("history endpoint returned 500")

    monkeypatch.setattr(web_server.backend, "get_result_image", boom)
    asyncio.run(web_server.handle_comfy_message(
        comfy_event("executing", prompt_id=prompt_id, node=None)))

    assert two_sessions["a"].types() == ["queued", "error"]


def test_execution_error_is_routed_and_cleaned_up(two_sessions):
    prompt_id = queue_fake_job("session-b")

    asyncio.run(web_server.handle_comfy_message(
        comfy_event("execution_error", prompt_id=prompt_id, exception_message="OOM in VAEDecode")))

    assert "OOM in VAEDecode" in two_sessions["b"].of_type("error")[0]["message"]
    assert two_sessions["a"].of_type("error") == []
    assert web_server.JOBS == {}


def test_interruption_from_comfy_s_own_ui_cleans_up_the_job(two_sessions):
    """Clearing the queue in ComfyUI's own UI fires execution_interrupted
    and nothing else -- without handling it the JOBS entry would leak for
    the life of the process."""
    prompt_id = queue_fake_job("session-a")

    asyncio.run(web_server.handle_comfy_message(
        comfy_event("execution_interrupted", prompt_id=prompt_id)))

    assert two_sessions["a"].of_type("error")[0]["message"] == "generation was interrupted"
    assert web_server.JOBS == {}


def test_events_for_unknown_prompt_ids_are_ignored(two_sessions):
    queue_fake_job("session-a")
    asyncio.run(web_server.handle_comfy_message(
        comfy_event("progress", prompt_id="a-prompt-we-never-queued", value=1, max=30)))

    assert two_sessions["a"].of_type("progress") == []
    assert two_sessions["b"].of_type("progress") == []


def test_status_events_broadcast_to_every_session(two_sessions):
    """Queue depth is not job-specific: a session with nothing in flight
    still wants to see that the GPU is busy."""
    asyncio.run(web_server.handle_comfy_message(
        json.dumps({"type": "status", "data": {"status": {"exec_info": {"queue_remaining": 3}}}})))

    for socket in two_sessions.values():
        assert socket.of_type("status") == [{"type": "status", "queue_remaining": 3}]


def test_malformed_json_from_comfy_is_ignored(two_sessions):
    queue_fake_job("session-a")
    asyncio.run(web_server.handle_comfy_message("not json at all"))
    assert two_sessions["a"].types() == ["queued"]


# --- binary preview frames --------------------------------------------------

def _preview_frame(payload=b"\xff\xd8jpegbytes"):
    # ComfyUI's wire format: 4-byte event type + 4-byte image format + data.
    return struct.pack(">I", web_server.BINARY_PREVIEW_EVENT) + struct.pack(">I", 1) + payload


def test_preview_frames_follow_the_currently_executing_job(two_sessions):
    """Binary frames carry no prompt_id, so they route via whichever job
    last emitted a JSON event -- which is unambiguous because ComfyUI
    executes one graph at a time."""
    prompt_id_a = queue_fake_job("session-a")
    queue_fake_job("session-b")

    asyncio.run(web_server.handle_comfy_message(
        comfy_event("progress", prompt_id=prompt_id_a, value=1, max=30)))
    asyncio.run(web_server.handle_comfy_message(_preview_frame()))

    assert two_sessions["a"].binary_messages == [b"\xff\xd8jpegbytes"], "header must be stripped"
    assert two_sessions["b"].binary_messages == []


def test_preview_frames_are_dropped_when_nothing_is_executing(two_sessions):
    queue_fake_job("session-a")
    asyncio.run(web_server.handle_comfy_message(_preview_frame()))
    assert two_sessions["a"].binary_messages == []


def test_truncated_binary_frames_are_ignored(two_sessions):
    prompt_id = queue_fake_job("session-a")
    asyncio.run(web_server.handle_comfy_message(
        comfy_event("progress", prompt_id=prompt_id, value=1, max=30)))
    asyncio.run(web_server.handle_comfy_message(b"\x00\x01"))
    assert two_sessions["a"].binary_messages == []


# --- disconnects and relay drops --------------------------------------------

def test_results_for_a_departed_session_do_not_reach_anyone_else(monkeypatch, two_sessions):
    """A tab closed mid-generation leaves its job running in ComfyUI. The
    result must be dropped quietly, never handed to the surviving session."""
    prompt_id = queue_fake_job("session-a")
    del web_server.SESSIONS["session-a"]
    monkeypatch.setattr(web_server.backend, "get_result_image", lambda pid: Image.new("RGB", (8, 8)))

    asyncio.run(web_server.handle_comfy_message(
        comfy_event("executing", prompt_id=prompt_id, node=None)))

    assert two_sessions["b"].json_messages == []
    assert web_server.JOBS == {}


def test_relay_drop_fails_every_in_flight_job(two_sessions):
    """ComfyUI does not replay events on reconnect, so jobs in flight when
    the websocket drops can never reach a terminal event. Left alone they
    accumulated in JOBS forever and stranded their browsers on a disabled
    Generate button."""
    queue_fake_job("session-a")
    queue_fake_job("session-b")
    web_server.executing_prompt_id = "whatever-was-running"

    asyncio.run(web_server._fail_orphaned_jobs("lost the connection to ComfyUI mid-generation"))

    assert web_server.JOBS == {}
    assert web_server.executing_prompt_id is None
    for socket in two_sessions.values():
        assert "lost the connection" in socket.of_type("error")[0]["message"]


def test_sending_to_a_dead_socket_does_not_raise(two_sessions):
    """send_json_to swallows failures on purpose: a browser that vanished
    mid-write must not take down the relay loop for everyone else."""
    class DeadSocket:
        async def send_json(self, payload):
            raise ConnectionResetError("client went away")

    web_server.SESSIONS["session-dead"] = DeadSocket()
    asyncio.run(web_server.send_json_to("session-dead", {"type": "progress"}))
    asyncio.run(web_server.send_json_to("no-such-session", {"type": "progress"}))
