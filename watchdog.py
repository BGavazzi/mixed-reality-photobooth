"""Jobs that never end.

Every failure this app handles so far announces itself. A refused connection
raises, a 503 raises, an `execution_error` event arrives, a dropped websocket
is noticed by the relay. The retry ladder, the breaker and the orphan sweep all
depend on something *happening*.

The failure with no announcement is a job that is accepted and then simply
never spoken of again. ComfyUI took the prompt, answered 200, and no terminal
event ever arrives -- because the render OOMed and took the worker with it,
because the websocket reconnected during the one second that carried the
`executed` event (ComfyUI does not replay), or because two processes shared a
`clientId` and the events went to the other one. That last is not hypothetical:
it happened in this repo, and the symptom was a browser sitting on "queued…"
forever while the GPU had long since finished the picture.

Nothing in the app noticed, and nothing could have, because *not receiving a
message* is not an event. So this is the one component that works on a timer.

**Two clocks, because a long wait is not the same as a hang.**

*A total budget* per job. Generous, and a backstop rather than a policy: at
some point a photo that has been in flight for half an hour is not coming
back, whatever ComfyUI believes.

*A stall detector*, which is the interesting one. A job sitting in ComfyUI's
own queue behind forty others is perfectly healthy and produces no events of
its own for a very long time -- timing it out on its own silence would kill
exactly the work that admission control exists to allow. So the stall clock
reads the *global* event stream: any event about any job proves ComfyUI is
alive and working, and only when the whole relay has been silent do in-flight
jobs become suspect. Progress on somebody else's photo is evidence about
yours.

**Ask before declaring death.** An overdue job is not necessarily a failed one
-- the most likely cause is a missed event, and the render may be sitting
finished in `/history` right now. So the sweep probes ComfyUI first and
*recovers* the result when it is there. Failing a photo that actually
succeeded would be a worse bug than the one this module fixes, and it is the
bug the naive version of a timeout ships with.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import config
import obs

# Total wall-clock a single generation may be in flight. Half an hour is far
# past any real render on this hardware (~35s) and past any plausible queue
# behind it; this is the "it is never coming back" line, not an SLA.
MAX_SECONDS = config.env_float(
    "JOB_MAX_SECONDS", 1800.0, "seconds a generation may be in flight before it is failed",
    minimum=10, maximum=86400)

# How long the *whole* ComfyUI event stream may be silent before in-flight
# jobs are treated as suspect. Five minutes rather than one: a cold model load
# on a laptop GPU genuinely produces no events for a minute or more, and a
# watchdog that fires during a legitimate first-run load would make the app
# less reliable, not more.
STALL_SECONDS = config.env_float(
    "JOB_STALL_SECONDS", 300.0, "seconds of total ComfyUI silence before jobs are suspect",
    minimum=5, maximum=86400)

INTERVAL_SECONDS = config.env_float(
    "WATCHDOG_INTERVAL", 15.0, "seconds between watchdog sweeps",
    minimum=1, maximum=3600)

# Probe outcomes. What ComfyUI says about a prompt it was asked to render.
FINISHED = "finished"   # it is in /history with outputs -- recoverable
QUEUED = "queued"       # still in queue_running/queue_pending -- not our business yet
UNKNOWN = "unknown"     # ComfyUI has never heard of it, or cannot be asked

_last_event_at = time.monotonic()


def note_event(now: float | None = None) -> None:
    """Called for every ComfyUI websocket event. Cheap on purpose: it runs on
    the hot path of the relay, several times a second during a render."""
    global _last_event_at
    _last_event_at = time.monotonic() if now is None else now


def last_event_at() -> float:
    return _last_event_at


def silence(now: float | None = None) -> float:
    return (time.monotonic() if now is None else now) - _last_event_at


@dataclass(frozen=True)
class Overdue:
    prompt_id: str
    reason: str      # "budget" | "stall"
    age: float

    @property
    def message(self) -> str:
        if self.reason == "budget":
            return (f"the render did not finish within {MAX_SECONDS:.0f}s and has been "
                    f"given up on")
        return ("the render service stopped responding mid-generation — an operator "
                "has been alerted")


def find_overdue(jobs: dict, *, now: float | None = None,
                 max_seconds: float | None = None,
                 stall_seconds: float | None = None,
                 last_event: float | None = None) -> list[Overdue]:
    """Which in-flight jobs are past a deadline. Pure, so the interesting cases
    are testable without a clock or a server.

    Entries with no `accepted_at` are skipped rather than treated as infinitely
    old: they are jobs registered a moment ago whose accept callback has not
    run yet, and failing those would break the ordering the relay depends on.
    """
    now = time.monotonic() if now is None else now
    max_seconds = MAX_SECONDS if max_seconds is None else max_seconds
    stall_seconds = STALL_SECONDS if stall_seconds is None else stall_seconds
    last_event = _last_event_at if last_event is None else last_event

    quiet_for = now - last_event
    overdue = []
    for prompt_id, entry in jobs.items():
        accepted_at = entry.get("accepted_at")
        if accepted_at is None:
            continue
        age = now - accepted_at
        if age > max_seconds:
            overdue.append(Overdue(prompt_id, "budget", age))
        elif age > stall_seconds and quiet_for > stall_seconds:
            # Both clocks, deliberately. A job younger than the stall window
            # has not waited long enough to be called stuck even if ComfyUI is
            # quiet -- it may have been submitted into that quiet a second ago.
            overdue.append(Overdue(prompt_id, "stall", age))
    return overdue


async def sweep(jobs: dict, *, probe, recover, fail, **limits) -> dict:
    """One pass. Returns a small tally, mostly so tests and logs can assert on
    what it did rather than on what it printed.

    `probe(prompt_id) -> (state, payload)` asks ComfyUI what it thinks;
    `recover(prompt_id, payload)` delivers a result that was found finished;
    `fail(prompt_id, overdue)` gives up on one. All three are injected because
    the watchdog has no business knowing how a result reaches a browser.
    """
    tally = {"checked": 0, "recovered": 0, "failed": 0, "waiting": 0}
    for item in find_overdue(jobs, **limits):
        tally["checked"] += 1
        with obs.bind(item.prompt_id):
            try:
                state, payload = await probe(item.prompt_id)
            except Exception as exc:                     # noqa: BLE001
                # A probe that fails is not evidence the job failed -- it is
                # evidence ComfyUI is unreachable, which the breaker already
                # handles. Leave the job alone and look again next sweep.
                obs.log("watchdog", "probe failed, leaving job alone",
                        error=repr(exc), age_s=round(item.age))
                tally["waiting"] += 1
                continue

            if state == FINISHED:
                obs.log("watchdog", "recovered a finished render whose event was missed",
                        age_s=round(item.age), reason=item.reason)
                await recover(item.prompt_id, payload)
                tally["recovered"] += 1
            elif state == QUEUED and item.reason == "stall":
                # ComfyUI still holds it. Silence with a job in the queue is
                # suspicious, but "suspicious" is not grounds for destroying
                # work the dependency says it still has. The budget clock is
                # the backstop if it really never moves.
                obs.log("watchdog", "still queued in ComfyUI despite silence",
                        age_s=round(item.age))
                tally["waiting"] += 1
            else:
                obs.log("watchdog", "giving up on a job", reason=item.reason,
                        age_s=round(item.age), comfy_says=state)
                await fail(item.prompt_id, item)
                tally["failed"] += 1
    return tally
