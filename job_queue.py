"""
An admission-controlled queue in front of ComfyUI.

Before this, every websocket message spawned `asyncio.create_task(...)` which
submitted to ComfyUI's `/prompt` immediately. That works -- ComfyUI has its
own queue and executes one graph at a time -- but it leaves the app with no
say in anything:

  * **No admission control.** A hundred queued photos are a hundred
    in-flight coroutines, each holding decoded PIL images in memory, all
    waiting on a GPU that will get to them one at a time. Nothing pushes
    back; the process just grows until it doesn't.
  * **No visibility.** "Queued" was all a client could be told. Not
    position, not depth, not how long the wait has been.
  * **No seam.** The step that has to move off this box first, if this ever
    becomes multi-machine, was inlined into the websocket handler.

So submission goes through a `JobQueue`. The default implementation is
in-process asyncio -- no Redis, no broker, nothing new to install -- but the
interface is the one a distributed queue would implement, so replacing it is
a class rather than a rewrite of the handlers.

**What this deliberately is not:** it is not a claim that the app now scales
horizontally. ComfyUI still renders one graph at a time and this still runs
on one machine. What it changes is that the boundary now exists, is typed,
and is tested -- and the handlers no longer know how submission happens.

Worker count is worth a note, since more workers do *not* mean faster
generation. The GPU is serial. What overlapping workers buy is that the HTTP
uploads for job N+1 (three PNGs to ComfyUI's /upload/image) can happen while
job N is still denoising, instead of strictly after it. Two is enough for
that; more just deepens a queue ComfyUI already has.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Awaitable, Callable

# Beyond this many waiting jobs, submission is refused rather than accepted
# and quietly starved. At ~35s per generation a depth of 64 is already a
# 35-minute wait; anything past that is a promise the app can't keep.
DEFAULT_MAX_DEPTH = 64
DEFAULT_WORKERS = 2


class QueueFullError(RuntimeError):
    """Raised on submit when the queue is at capacity. Deliberately an error
    the caller must handle, rather than a silent drop or an unbounded wait."""


@dataclass
class GenerationJob:
    """One unit of work: everything needed to submit it, and to route its
    result back afterwards.

    `submit` is a callable rather than a workflow because the queue has no
    business knowing what a workflow is -- it schedules work and reports
    outcomes. That is also what makes it testable without ComfyUI.
    """
    session_id: str
    kind: str
    submit: Callable[[str], tuple[str, int]]
    provenance: dict = field(default_factory=dict)
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    enqueued_at: float = field(default_factory=time.monotonic)

    @property
    def waited(self) -> float:
        return time.monotonic() - self.enqueued_at


class JobQueue(ABC):
    """The seam. A Redis/Celery-backed implementation would satisfy this same
    interface; nothing above it knows which one it has."""

    @abstractmethod
    async def submit(self, job: GenerationJob) -> int:
        """Accepts a job, returning its position in the queue (1 = next up).
        Raises QueueFullError if at capacity."""

    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...

    @abstractmethod
    async def drain(self) -> None:
        """Returns once everything currently queued has been submitted."""

    @abstractmethod
    def stats(self) -> dict: ...


class InProcessJobQueue(JobQueue):
    """asyncio.Queue plus a fixed pool of worker tasks.

    The two callbacks are how results leave the queue. They are supplied by
    the caller rather than the queue reaching back into the app, so the queue
    depends on nothing -- which is what lets the tests drive it with plain
    functions and no server.
    """

    def __init__(
        self,
        on_accepted: Callable[[GenerationJob, str, int], Awaitable[None]],
        on_failed: Callable[[GenerationJob, Exception], Awaitable[None]],
        workers: int = DEFAULT_WORKERS,
        max_depth: int = DEFAULT_MAX_DEPTH,
        run_blocking: Callable[..., Awaitable] = asyncio.to_thread,
    ):
        self._on_accepted = on_accepted
        self._on_failed = on_failed
        self._workers = max(1, workers)
        self._max_depth = max_depth
        # Injected so tests can run the "blocking" submit inline instead of
        # in a real thread -- otherwise every queue test pays for thread
        # scheduling and becomes timing-dependent.
        self._run_blocking = run_blocking
        # Created in start(), not here. This object is constructed at module
        # import, outside any event loop; an asyncio.Queue built there and
        # then used from a loop created later binds its internal waiter
        # futures to the wrong loop. Deferring construction to start() keeps
        # the queue and the loop that drives it the same age.
        self._queue: asyncio.Queue[GenerationJob] | None = None
        self._tasks: list[asyncio.Task] = []
        self._submitted = 0
        self._failed = 0
        self._running = 0

    # --- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        if self._tasks:
            return
        self._queue = asyncio.Queue()
        self._tasks = [asyncio.create_task(self._worker(i), name=f"jobqueue-worker-{i}")
                       for i in range(self._workers)]
        print(f"[queue] started {self._workers} worker(s), max depth {self._max_depth}")

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        # Gathered rather than left dangling: an un-awaited cancelled task
        # logs "Task exception was never retrieved" on interpreter shutdown,
        # which looks like a crash in the server's final output.
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []
        self._queue = None

    async def drain(self) -> None:
        """Waits until everything currently queued has been submitted.

        Real callers want this -- a batch run needs to know when its whole set
        is in ComfyUI's hands, and shutdown wants to not strand work. It is
        also what lets the tests assert on outcomes without sleeping.
        """
        if self._queue is not None:
            await self._queue.join()

    # --- submission --------------------------------------------------------

    async def submit(self, job: GenerationJob) -> int:
        if self._queue is None:
            raise RuntimeError("job queue used before start(); call start() in app startup")
        if self._queue.qsize() >= self._max_depth:
            raise QueueFullError(
                f"the generation queue is full ({self._queue.qsize()} waiting); "
                f"wait for it to drain before adding more")
        self._queue.put_nowait(job)
        return self._queue.qsize()

    # --- the worker loop ---------------------------------------------------

    async def _worker(self, index: int) -> None:
        while True:
            job = await self._queue.get()
            self._running += 1
            try:
                # The backend's submit is blocking HTTP (it uploads several
                # PNGs), so it runs off the event loop -- otherwise a single
                # upload would stall the ComfyUI relay and every other
                # session's progress events with it.
                prompt_id, seed = await self._run_blocking(job.submit, job.job_id)
                self._submitted += 1
                await self._on_accepted(job, prompt_id, seed)
            except asyncio.CancelledError:
                # Shutdown, not failure. Re-raised so the task actually ends;
                # swallowing it here would make stop() hang forever.
                raise
            except Exception as exc:
                self._failed += 1
                # A worker that dies on a bad job takes the pool's capacity
                # down with it and every later job waits forever. Report and
                # keep serving.
                print(f"[queue] worker {index} job {job.job_id} failed: {exc!r}")
                try:
                    await self._on_failed(job, exc)
                except Exception as callback_exc:
                    print(f"[queue] failure callback itself raised: {callback_exc!r}")
            finally:
                self._running -= 1
                self._queue.task_done()

    # --- introspection -----------------------------------------------------

    def stats(self) -> dict:
        return {
            "waiting": self._queue.qsize() if self._queue is not None else 0,
            "running": self._running,
            "submitted": self._submitted,
            "failed": self._failed,
            "workers": self._workers,
            "max_depth": self._max_depth,
        }
