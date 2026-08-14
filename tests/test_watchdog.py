"""
Tests for the failure that does not announce itself.

Every other failure path in this app is driven by something arriving -- an
exception, an error event, a dropped socket. This one is driven by nothing
arriving, which makes two things easy to get wrong and both are tested here:

  * timing out work that is merely *waiting* (a job behind forty others in
    ComfyUI's queue is healthy, and killing it would break the queueing the
    whole app is built around), and
  * declaring a job dead when the render actually finished and only the
    *event* was lost -- which is the most likely cause, and destroying a
    finished picture would be a worse bug than the one being fixed.
"""

import asyncio

import pytest

import obs
import watchdog


@pytest.fixture(autouse=True)
def capture_logs(monkeypatch):
    monkeypatch.setattr(obs, "SINK", [])
    yield obs.SINK


def job(accepted_at, **extra):
    return {"session_id": "s", "kind": "background", "accepted_at": accepted_at, **extra}


# --- which jobs are overdue ----------------------------------------------------

def test_a_job_past_its_total_budget_is_overdue():
    jobs = {"p1": job(accepted_at=0)}

    overdue = watchdog.find_overdue(jobs, now=2000, max_seconds=1800,
                                    stall_seconds=300, last_event=1999)

    assert [(o.prompt_id, o.reason) for o in overdue] == [("p1", "budget")]


def test_a_recent_job_is_not_overdue_however_quiet_it_is():
    jobs = {"p1": job(accepted_at=1000)}

    overdue = watchdog.find_overdue(jobs, now=1010, max_seconds=1800,
                                    stall_seconds=300, last_event=0)

    assert overdue == [], "a job ten seconds old cannot be stuck yet"


def test_progress_on_another_job_keeps_a_waiting_job_alive():
    """The case that makes a naive per-job timeout wrong. A photo sitting in
    ComfyUI's queue behind forty others emits no events of its own for a very
    long time and is perfectly healthy -- the evidence that the system is
    working is somebody else's progress."""
    jobs = {"waiting": job(accepted_at=0)}

    overdue = watchdog.find_overdue(jobs, now=600, max_seconds=1800,
                                    stall_seconds=300, last_event=599)

    assert overdue == []


def test_total_silence_makes_an_old_job_suspect():
    jobs = {"p1": job(accepted_at=0)}

    overdue = watchdog.find_overdue(jobs, now=600, max_seconds=1800,
                                    stall_seconds=300, last_event=100)

    assert [(o.prompt_id, o.reason) for o in overdue] == [("p1", "stall")]


def test_a_job_not_yet_accepted_is_left_alone():
    """Entries are registered before submission so the relay cannot race them.
    Treating a missing accepted_at as "infinitely old" would fail every job in
    the instant between registration and acceptance."""
    jobs = {"p1": {"session_id": "s", "kind": "background", "provenance": None}}

    assert watchdog.find_overdue(jobs, now=99999, max_seconds=1, stall_seconds=1,
                                 last_event=0) == []


def test_note_event_resets_the_stall_clock():
    watchdog.note_event(now=500)
    assert watchdog.silence(now=560) == 60
    watchdog.note_event(now=560)
    assert watchdog.silence(now=560) == 0


# --- what the sweep does about them --------------------------------------------

def run(coro):
    """The repo's convention: coroutines are driven from sync tests with
    asyncio.run rather than a plugin, so the suite has no extra dependency."""
    return asyncio.run(coro)


def test_a_finished_render_is_recovered_not_failed():
    """The headline case. ComfyUI rendered the picture and the event was lost
    -- exactly what a shared clientId did in this repo. The photo exists; the
    watchdog's job is to go and get it."""
    recovered, failed = [], []

    async def probe(prompt_id):
        return watchdog.FINISHED, "the-image"

    tally = run(watchdog.sweep(
        {"p1": job(accepted_at=0)},
        probe=probe,
        recover=lambda pid, payload: _record(recovered, (pid, payload)),
        fail=lambda pid, item: _record(failed, pid),
        now=2000, max_seconds=1800, stall_seconds=300, last_event=1999))

    assert recovered == [("p1", "the-image")]
    assert failed == []
    assert tally["recovered"] == 1


def test_a_job_comfyui_has_never_heard_of_is_failed():
    failed = []

    async def probe(prompt_id):
        return watchdog.UNKNOWN, None

    run(watchdog.sweep(
        {"p1": job(accepted_at=0)},
        probe=probe,
        recover=lambda pid, payload: _record([], None),
        fail=lambda pid, item: _record(failed, (pid, item.reason)),
        now=2000, max_seconds=1800, stall_seconds=300, last_event=1999))

    assert failed == [("p1", "budget")]


def test_a_stalled_job_comfyui_still_holds_is_left_alone():
    """Silence with the job still in ComfyUI's queue is suspicious, not proof.
    Destroying work the dependency says it still has would be the watchdog
    causing the failure it exists to detect; the budget clock is the backstop."""
    failed, recovered = [], []

    async def probe(prompt_id):
        return watchdog.QUEUED, None

    tally = run(watchdog.sweep(
        {"p1": job(accepted_at=0)},
        probe=probe,
        recover=lambda pid, payload: _record(recovered, pid),
        fail=lambda pid, item: _record(failed, pid),
        now=600, max_seconds=1800, stall_seconds=300, last_event=0))

    assert (failed, recovered) == ([], [])
    assert tally["waiting"] == 1


def test_a_job_still_queued_past_the_total_budget_is_given_up_on():
    """The backstop actually backstopping: half an hour in ComfyUI's queue
    with nothing happening is not a photo anyone is still waiting for."""
    failed = []

    async def probe(prompt_id):
        return watchdog.QUEUED, None

    run(watchdog.sweep(
        {"p1": job(accepted_at=0)},
        probe=probe,
        recover=lambda pid, payload: _record([], None),
        fail=lambda pid, item: _record(failed, pid),
        now=2000, max_seconds=1800, stall_seconds=300, last_event=1999))

    assert failed == ["p1"]


def test_an_unreachable_comfyui_does_not_destroy_jobs():
    """A probe that raises means ComfyUI is unreachable, which is news about
    the dependency and not about this job. The breaker already handles that;
    the watchdog must not turn it into fifty destroyed photos."""
    failed = []

    async def probe(prompt_id):
        raise ConnectionError("comfy is down")

    tally = run(watchdog.sweep(
        {"p1": job(accepted_at=0)},
        probe=probe,
        recover=lambda pid, payload: _record([], None),
        fail=lambda pid, item: _record(failed, pid),
        now=2000, max_seconds=1800, stall_seconds=300, last_event=1999))

    assert failed == []
    assert tally["waiting"] == 1


def test_the_sweep_logs_under_the_jobs_id(capture_logs):
    """So the line about giving up on a photo sits in a grep next to that
    photo's queue, retry and relay lines."""
    async def probe(prompt_id):
        return watchdog.UNKNOWN, None

    run(watchdog.sweep(
        {"abcdef12-0000": job(accepted_at=0)},
        probe=probe,
        recover=lambda pid, payload: _record([], None),
        fail=lambda pid, item: _record([], None),
        now=2000, max_seconds=1800, stall_seconds=300, last_event=1999))

    assert all(record["job"] == "abcdef12-0000" for record in capture_logs)


async def _record(sink, value):
    sink.append(value)
