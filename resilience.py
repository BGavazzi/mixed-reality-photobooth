"""Retrying calls to ComfyUI without lying to the person standing at the booth.

ComfyUI is a real dependency with real failure modes -- VRAM pressure, model
reloads, an OOM fallback mid-generation, a dropped upload -- and until now this
app's entire response to any of them was to hand the browser an error. Every
HTTP call had a timeout and not one of them was ever retried.

Three ideas do most of the work here.

**Not every failure deserves a retry.** A 503 means ComfyUI declined the work,
so trying again is free and often succeeds. A 400 means the workflow is wrong,
and retrying it four times just makes the guest wait four times longer for the
same answer. Classification is the difference between resilience and a busy
loop, and it is the first thing that gets skipped when someone reaches for a
`@retry(3)` decorator.

**A retry can be unsafe.** `POST /prompt` queues GPU work. If the request timed
out on the *read*, ComfyUI may have accepted it and started rendering -- naively
retrying then burns a second slot of a serial GPU and produces a duplicate. So
failures are sorted three ways, not two: terminal, safe to retry, and
*ambiguous*. Ambiguous cases get reconciled (did the work actually land?) before
anything is retried. See `call` and its `reconcile` argument.

**A retry ladder is the wrong answer when the dependency is simply down.** If
ComfyUI is not running, the useful response is to say so immediately, not to
make every guest in the queue wait through the full backoff first. That is what
the circuit breaker is for.

Deliberately no `tenacity`: the interesting part here is the classification and
the reconciliation, neither of which a decorator would do for us, and this file
stays readable by someone who has never seen the library.
"""

from __future__ import annotations

import random
import socket
import threading
import time
import urllib.error
from dataclasses import dataclass
from enum import Enum
from typing import Callable

import requests

# Status codes worth trying again. 5xx is the server saying it could not do the
# work; 429 is it asking us to slow down. Everything else in 4xx is a statement
# about the request, which will be exactly as wrong the second time.
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class Verdict(Enum):
    """How a failure should be treated.

    AMBIGUOUS exists because of `POST /prompt`: a read timeout means we do not
    know whether the server accepted the work. Collapsing it into RETRY would
    risk duplicate generations; collapsing it into TERMINAL would fail requests
    that actually succeeded. Neither is acceptable, so the caller is told it
    does not know and given a chance to find out.
    """
    TERMINAL = "terminal"
    RETRY = "retry"
    AMBIGUOUS = "ambiguous"


class CircuitOpenError(RuntimeError):
    """Raised instead of calling, while the breaker is open. Distinct from the
    underlying error so callers can say "ComfyUI appears to be down" rather
    than replaying whatever the last failure happened to be."""


def classify(exc: BaseException) -> Verdict:
    """Sorts an exception from `requests` or `urllib` into a Verdict.

    Connection errors are RETRY rather than AMBIGUOUS on purpose: the
    connection was never established, so the server cannot have seen the
    request, which makes a retry safe even for calls that have side effects.
    """
    if isinstance(exc, requests.exceptions.HTTPError) and exc.response is not None:
        return Verdict.RETRY if exc.response.status_code in RETRYABLE_STATUS else Verdict.TERMINAL

    if isinstance(exc, urllib.error.HTTPError):
        return Verdict.RETRY if exc.code in RETRYABLE_STATUS else Verdict.TERMINAL

    # Read timeouts: the request went out and we never heard back. Whether the
    # server did the work is exactly the thing we cannot know from here.
    if isinstance(exc, (requests.exceptions.ReadTimeout, socket.timeout)):
        return Verdict.AMBIGUOUS
    if isinstance(exc, requests.exceptions.ConnectTimeout):
        return Verdict.RETRY  # never established, so nothing happened server-side

    if isinstance(exc, requests.exceptions.ConnectionError):
        return Verdict.RETRY
    if isinstance(exc, urllib.error.URLError):
        # URLError wraps the socket error; a timeout inside it is still
        # ambiguous for the same reason as above.
        return Verdict.AMBIGUOUS if isinstance(exc.reason, socket.timeout) else Verdict.RETRY

    if isinstance(exc, requests.exceptions.Timeout):
        return Verdict.AMBIGUOUS

    # An unrecognised exception is a bug in our code far more often than a
    # blip in theirs, and retrying a bug wastes the guest's time.
    return Verdict.TERMINAL


@dataclass(frozen=True)
class RetryPolicy:
    """How hard to try, and for how long.

    `budget_seconds` matters more than `attempts` here. There is a person
    watching a progress bar, so the honest limit is wall-clock: better to say
    "the render service is struggling, I've flagged it" after 20 seconds than
    to keep a guest waiting through a technically-correct backoff ladder.
    """
    attempts: int = 4
    base_delay: float = 0.5
    max_delay: float = 8.0
    budget_seconds: float = 20.0

    def delay_for(self, attempt: int, rng: random.Random) -> float:
        """Exponential backoff with **full** jitter: a uniform draw from
        [0, backoff], not backoff plus a wiggle.

        A booth has several sessions failing at the same instant on the same
        ComfyUI. Equal delays would send them back in a synchronised wave and
        re-break whatever just recovered; full jitter spreads them out. It is
        the one detail that separates backoff that helps from backoff that
        merely feels responsible.
        """
        return rng.uniform(0, min(self.max_delay, self.base_delay * (2 ** attempt)))


class CircuitBreaker:
    """Fails fast once a dependency looks properly down, and probes to reopen.

    Thread-safe because the queue's workers call through it from a thread pool
    (`asyncio.to_thread`), so the counters are genuinely shared state.

    The clock is injected so tests can prove the recovery behaviour without
    sleeping through it -- a breaker tested with real sleeps is a breaker whose
    timings are never actually tested.
    """

    CLOSED, OPEN, HALF_OPEN = "closed", "open", "half_open"

    def __init__(self, failure_threshold: int = 5, recovery_seconds: float = 30.0,
                 clock: Callable[[], float] = time.monotonic, name: str = "comfyui"):
        self.failure_threshold = failure_threshold
        self.recovery_seconds = recovery_seconds
        self._clock = clock
        self.name = name
        self._lock = threading.Lock()
        self._consecutive_failures = 0
        self._opened_at: float | None = None
        self._half_open_probe = False
        self.trips = 0

    @property
    def state(self) -> str:
        with self._lock:
            return self._state_locked()

    def _state_locked(self) -> str:
        if self._opened_at is None:
            return self.CLOSED
        if self._clock() - self._opened_at >= self.recovery_seconds:
            return self.HALF_OPEN
        return self.OPEN

    def before_call(self) -> None:
        """Raises CircuitOpenError if the call should not be attempted."""
        with self._lock:
            state = self._state_locked()
            if state == self.OPEN:
                waited = self._clock() - self._opened_at
                raise CircuitOpenError(
                    f"{self.name} looks unavailable ({self._consecutive_failures} "
                    f"consecutive failures); not retrying for another "
                    f"{self.recovery_seconds - waited:.0f}s")
            if state == self.HALF_OPEN:
                if self._half_open_probe:
                    # Exactly one request gets to find out whether it recovered.
                    # Letting the whole backlog through would re-flatten a
                    # service that is only just back on its feet.
                    raise CircuitOpenError(
                        f"{self.name} is being probed for recovery; try again shortly")
                self._half_open_probe = True

    def record_success(self) -> None:
        with self._lock:
            self._consecutive_failures = 0
            self._opened_at = None
            self._half_open_probe = False

    def record_failure(self) -> None:
        with self._lock:
            self._half_open_probe = False
            if self._opened_at is not None:
                # Failed the probe: restart the clock rather than letting a
                # half-open breaker retry every recovery_seconds forever.
                self._opened_at = self._clock()
                return
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.failure_threshold:
                self._opened_at = self._clock()
                self.trips += 1

    def stats(self) -> dict:
        with self._lock:
            return {
                "name": self.name,
                "state": self._state_locked(),
                "consecutive_failures": self._consecutive_failures,
                "trips": self.trips,
            }


def call(
    fn: Callable[[], object],
    *,
    policy: RetryPolicy | None = None,
    breaker: CircuitBreaker | None = None,
    reconcile: Callable[[], object] | None = None,
    label: str = "call",
    on_attempt: Callable[[int, BaseException, float | None], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    rng: random.Random | None = None,
):
    """Calls `fn`, retrying what is worth retrying.

    reconcile: for calls with side effects. When a failure is AMBIGUOUS -- we
        sent the request and never heard back -- this is asked whether the work
        actually landed. Returning non-None means it did, and that value
        becomes the result; returning None means it did not, and the call is
        retried like any other. Without this, an ambiguous failure on
        `POST /prompt` has only two bad options: duplicate the generation, or
        fail one that already succeeded.

    on_attempt: called with (attempt_number, exception, delay_or_None) for each
        failure, so callers can log every attempt rather than only the last.
        The failure that eventually succeeded is worth seeing in the log too --
        it is the early warning that a dependency is degrading.
    """
    policy = policy or RetryPolicy()
    rng = rng or random.Random()
    started = clock()
    last_exc: BaseException | None = None

    for attempt in range(policy.attempts):
        if breaker is not None:
            breaker.before_call()   # raises CircuitOpenError; deliberately not caught
        try:
            result = fn()
        except BaseException as exc:      # noqa: BLE001 -- re-raised below unless retryable
            last_exc = exc
            verdict = classify(exc)

            if verdict is Verdict.AMBIGUOUS and reconcile is not None:
                # Ask before assuming. This is the whole reason AMBIGUOUS is
                # a separate verdict.
                try:
                    landed = reconcile()
                except Exception:         # noqa: BLE001 -- reconciliation is best-effort
                    landed = None
                if landed is not None:
                    if breaker is not None:
                        breaker.record_success()
                    return landed

            if breaker is not None:
                breaker.record_failure()

            if verdict is Verdict.TERMINAL:
                if on_attempt:
                    on_attempt(attempt + 1, exc, None)
                raise

            delay = policy.delay_for(attempt, rng)
            elapsed = clock() - started
            out_of_budget = elapsed + delay > policy.budget_seconds
            last_attempt = attempt == policy.attempts - 1
            if on_attempt:
                on_attempt(attempt + 1, exc, None if (last_attempt or out_of_budget) else delay)
            if last_attempt or out_of_budget:
                raise
            sleep(delay)
        else:
            if breaker is not None:
                breaker.record_success()
            return result

    raise last_exc  # unreachable; the loop either returns or raises
