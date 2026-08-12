"""
Tests for the generation queue.

The queue's job is to be boring under pressure, so the interesting cases are
all failures: a job whose submission raises must not take a worker down with
it, a full queue must refuse rather than grow, and shutdown must not strand
work or leave cancelled tasks screaming into the interpreter's exit.

`run_blocking` is injected throughout so the "blocking" submit runs inline
instead of on a real thread. Otherwise every assertion here would be racing a
thread pool, and the flakiness would be the test's fault rather than the
code's.
"""

import asyncio

import pytest

import job_queue
from job_queue import GenerationJob, InProcessJobQueue, QueueFullError


async def inline(fn, *args):
    """Stands in for asyncio.to_thread, running the callable synchronously."""
    return fn(*args)


class Recorder:
    def __init__(self):
        self.accepted = []
        self.failed = []

    async def on_accepted(self, job, prompt_id, seed):
        self.accepted.append((job.job_id, prompt_id, seed))

    async def on_failed(self, job, exc):
        self.failed.append((job.job_id, str(exc)))


def make_queue(recorder, **kwargs):
    kwargs.setdefault("run_blocking", inline)
    kwargs.setdefault("workers", 1)
    return InProcessJobQueue(recorder.on_accepted, recorder.on_failed, **kwargs)


def job(job_id="j1", submit=None, **kwargs):
    return GenerationJob(
        session_id=kwargs.pop("session_id", "s1"),
        kind=kwargs.pop("kind", "background"),
        submit=submit or (lambda prompt_id: (prompt_id, 1234)),
        job_id=job_id,
        **kwargs,
    )


def run(coro_factory):
    return asyncio.run(coro_factory())


# --- the happy path -------------------------------------------------------------

def test_a_submitted_job_reaches_the_backend_and_reports_back():
    recorder = Recorder()

    async def main():
        queue = make_queue(recorder)
        await queue.start()
        await queue.submit(job("j1"))
        await queue.drain()
        await queue.stop()

    run(lambda: main())
    assert recorder.accepted == [("j1", "j1", 1234)]
    assert recorder.failed == []


def test_the_job_id_is_handed_to_submit_as_the_prompt_id():
    """web_server pre-registers JOBS under this id before the queue ever runs
    the job, so the two must be the same value or every event routes nowhere."""
    seen = []
    recorder = Recorder()

    async def main():
        queue = make_queue(recorder)
        await queue.start()
        await queue.submit(job("abc-123", submit=lambda pid: (seen.append(pid), (pid, 7))[1]))
        await queue.drain()
        await queue.stop()

    run(lambda: main())
    assert seen == ["abc-123"]


def test_submit_returns_a_queue_position():
    recorder = Recorder()

    async def main():
        # No workers started, so nothing drains and positions are observable.
        queue = make_queue(recorder)
        queue._queue = asyncio.Queue()
        return [await queue.submit(job(f"j{i}")) for i in range(3)]

    assert run(lambda: main()) == [1, 2, 3]


def test_jobs_are_submitted_in_order_with_a_single_worker():
    order = []
    recorder = Recorder()

    async def main():
        queue = make_queue(recorder, workers=1)
        await queue.start()
        for i in range(5):
            await queue.submit(job(f"j{i}", submit=lambda pid: (order.append(pid), (pid, 0))[1]))
        await queue.drain()
        await queue.stop()

    run(lambda: main())
    assert order == [f"j{i}" for i in range(5)]


# --- failures -------------------------------------------------------------------

def test_a_failing_job_is_reported_and_does_not_kill_the_worker():
    """The important half is the second: a worker that dies takes the pool's
    capacity with it and every later job waits forever."""
    recorder = Recorder()

    def explode(prompt_id):
        raise RuntimeError("ComfyUI unreachable")

    async def main():
        queue = make_queue(recorder, workers=1)
        await queue.start()
        await queue.submit(job("bad", submit=explode))
        await queue.submit(job("good"))
        await queue.drain()
        await queue.stop()

    run(lambda: main())
    assert [j for j, _ in recorder.failed] == ["bad"]
    assert "ComfyUI unreachable" in recorder.failed[0][1]
    assert [j for j, _, _ in recorder.accepted] == ["good"], \
        "the worker must still be serving after a failed job"


def test_a_raising_failure_callback_does_not_kill_the_worker_either():
    """The error path's own error path. A send to a disconnected websocket
    raising here would otherwise be a second way to lose a worker."""
    class BadRecorder(Recorder):
        async def on_failed(self, job, exc):
            self.failed.append((job.job_id, str(exc)))
            raise RuntimeError("the callback itself blew up")

    recorder = BadRecorder()

    async def main():
        queue = make_queue(recorder, workers=1)
        await queue.start()
        await queue.submit(job("bad", submit=lambda pid: (_ for _ in ()).throw(ValueError("nope"))))
        await queue.submit(job("good"))
        await queue.drain()
        await queue.stop()

    run(lambda: main())
    assert [j for j, _, _ in recorder.accepted] == ["good"]


def test_stats_count_both_outcomes():
    recorder = Recorder()

    async def main():
        queue = make_queue(recorder, workers=1)
        await queue.start()
        await queue.submit(job("ok1"))
        await queue.submit(job("bad", submit=lambda pid: (_ for _ in ()).throw(ValueError("x"))))
        await queue.submit(job("ok2"))
        await queue.drain()
        stats = queue.stats()
        await queue.stop()
        return stats

    stats = run(lambda: main())
    assert stats["submitted"] == 2
    assert stats["failed"] == 1
    assert stats["waiting"] == 0


# --- admission control -----------------------------------------------------------

def test_a_full_queue_refuses_rather_than_growing():
    """Accepting unboundedly would mean a hundred queued photos are a hundred
    live coroutines holding decoded images, all waiting on one GPU."""
    recorder = Recorder()

    async def main():
        queue = make_queue(recorder, max_depth=2)
        queue._queue = asyncio.Queue()  # no workers, so nothing drains
        await queue.submit(job("j1"))
        await queue.submit(job("j2"))
        with pytest.raises(QueueFullError) as excinfo:
            await queue.submit(job("j3"))
        return str(excinfo.value)

    message = run(lambda: main())
    assert "full" in message and "2 waiting" in message, \
        "the refusal should say how deep the queue actually is"


def test_using_the_queue_before_start_is_a_clear_error():
    """Rather than an AttributeError on None, which would be the symptom of a
    wiring mistake in app startup and should say so."""
    recorder = Recorder()

    async def main():
        queue = make_queue(recorder)
        with pytest.raises(RuntimeError, match="before start"):
            await queue.submit(job())

    run(lambda: main())


# --- lifecycle --------------------------------------------------------------------

def test_start_is_idempotent():
    recorder = Recorder()

    async def main():
        queue = make_queue(recorder, workers=2)
        await queue.start()
        await queue.start()
        count = len(queue._tasks)
        await queue.stop()
        return count

    assert run(lambda: main()) == 2


def test_stop_cancels_every_worker():
    recorder = Recorder()

    async def main():
        queue = make_queue(recorder, workers=3)
        await queue.start()
        tasks = list(queue._tasks)
        await queue.stop()
        return tasks

    tasks = run(lambda: main())
    assert tasks and all(t.done() for t in tasks)


def test_stop_before_start_is_harmless():
    """Startup can fail before the queue is up; shutdown still runs."""
    recorder = Recorder()
    run(lambda: make_queue(recorder).stop())


def test_the_default_queue_satisfies_the_abstract_interface():
    """Guards the seam: if InProcessJobQueue drifts from JobQueue, the
    swap-in-a-real-broker story stops being true."""
    assert issubclass(InProcessJobQueue, job_queue.JobQueue)
    for method in ("submit", "start", "stop", "drain", "stats"):
        assert getattr(InProcessJobQueue, method) is not getattr(job_queue.JobQueue, method), \
            f"{method} is still abstract"
