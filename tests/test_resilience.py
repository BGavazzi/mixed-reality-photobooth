"""
Tests for the retry ladder, the failure classification and the circuit breaker.

Everything here runs against a fake clock and a fake sleep. That is not just
for speed: a breaker whose recovery window is exercised with real `time.sleep`
is a breaker whose timing is never actually asserted, only waited out. With the
clock injected, "it reopens after exactly 30 seconds" is a claim the suite can
make.

The cases worth reading are the ambiguous ones. A read timeout on a call that
queues GPU work is the failure this module exists for, and "retry it" and
"fail it" are both wrong answers.
"""

import socket
import urllib.error

import pytest
import requests

import resilience
from resilience import CircuitBreaker, CircuitOpenError, RetryPolicy, Verdict


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def http_error(status):
    response = requests.Response()
    response.status_code = status
    return requests.exceptions.HTTPError(response=response)


class Recorder:
    """Collects on_attempt callbacks so a test can assert on what was logged."""

    def __init__(self):
        self.calls = []

    def __call__(self, attempt, exc, delay):
        self.calls.append((attempt, type(exc).__name__, delay))


# --- classification -----------------------------------------------------------

@pytest.mark.parametrize("status,expected", [
    (500, Verdict.RETRY), (502, Verdict.RETRY), (503, Verdict.RETRY),
    (504, Verdict.RETRY), (429, Verdict.RETRY),
    (400, Verdict.TERMINAL), (404, Verdict.TERMINAL), (422, Verdict.TERMINAL),
])
def test_status_codes_are_sorted_by_whether_trying_again_could_help(status, expected):
    """5xx is the server declining the work; 4xx is a statement about the
    request, which will be just as wrong the second time."""
    assert resilience.classify(http_error(status)) is expected


def test_a_connection_error_is_safely_retryable():
    """Nothing was ever sent, so even a call with side effects can be retried
    without risking a duplicate."""
    assert resilience.classify(requests.exceptions.ConnectionError()) is Verdict.RETRY


def test_a_connect_timeout_is_safe_but_a_read_timeout_is_ambiguous():
    """The distinction the whole module turns on: failing to *reach* the server
    is different from failing to hear back from it."""
    assert resilience.classify(requests.exceptions.ConnectTimeout()) is Verdict.RETRY
    assert resilience.classify(requests.exceptions.ReadTimeout()) is Verdict.AMBIGUOUS


def test_urllib_errors_are_classified_too():
    """The /view download uses urllib, not requests, so its exception types
    have to be understood as well -- otherwise every image download failure
    would fall through to TERMINAL and never retry."""
    assert resilience.classify(urllib.error.HTTPError("u", 503, "x", {}, None)) is Verdict.RETRY
    assert resilience.classify(urllib.error.HTTPError("u", 404, "x", {}, None)) is Verdict.TERMINAL
    assert resilience.classify(urllib.error.URLError(socket.timeout())) is Verdict.AMBIGUOUS
    assert resilience.classify(urllib.error.URLError("refused")) is Verdict.RETRY


def test_an_unrecognised_exception_is_terminal():
    """A TypeError is a bug in this codebase, not a blip in ComfyUI, and
    retrying a bug just makes the guest wait four times as long for it."""
    assert resilience.classify(TypeError("nope")) is Verdict.TERMINAL


# --- the retry ladder ---------------------------------------------------------

def test_a_transient_failure_is_retried_and_the_result_returned():
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise http_error(503)
        return "ok"

    assert resilience.call(flaky, sleep=lambda d: None) == "ok"
    assert len(attempts) == 3


def test_a_terminal_failure_is_not_retried_at_all():
    """The busy-loop case: four attempts at a 400 is four times the wait for
    exactly the same answer."""
    attempts = []

    def broken():
        attempts.append(1)
        raise http_error(400)

    with pytest.raises(requests.exceptions.HTTPError):
        resilience.call(broken, sleep=lambda d: None)
    assert len(attempts) == 1


def test_attempts_are_capped_and_the_last_error_is_raised():
    attempts = []
    recorder = Recorder()

    def always_down():
        attempts.append(1)
        raise http_error(503)

    with pytest.raises(requests.exceptions.HTTPError):
        resilience.call(always_down, policy=RetryPolicy(attempts=3),
                        sleep=lambda d: None, on_attempt=recorder)
    assert len(attempts) == 3
    assert recorder.calls[-1][2] is None, "the final attempt should not announce a retry delay"


def test_the_wall_clock_budget_stops_the_ladder_early():
    """There is a person watching a progress bar. Better to say "the render
    service is struggling" after 20s than to finish a technically-correct
    backoff ladder at 60."""
    clock = FakeClock()
    attempts = []

    def slow_and_broken():
        attempts.append(1)
        clock.advance(9)          # each attempt burns most of the budget
        raise http_error(503)

    with pytest.raises(requests.exceptions.HTTPError):
        resilience.call(slow_and_broken, policy=RetryPolicy(attempts=10, budget_seconds=20),
                        sleep=clock.advance, clock=clock)
    assert len(attempts) < 10, "the budget, not the attempt count, should have ended it"


def test_backoff_uses_full_jitter():
    """Delays are drawn from [0, backoff], not backoff-plus-a-wiggle. A booth
    fails several sessions at the same instant against the same ComfyUI;
    equal delays send them back as a synchronised wave."""
    import random
    policy = RetryPolicy(base_delay=1.0, max_delay=8.0)
    draws = [policy.delay_for(2, random.Random(seed)) for seed in range(200)]

    assert min(draws) < 0.5, "full jitter should sometimes retry almost immediately"
    assert max(draws) <= 4.0, "and never exceed the exponential ceiling for that attempt"
    assert len(set(draws)) > 100, "delays should be spread, not clustered"


def test_every_attempt_is_reported_not_just_the_last():
    """A call that failed twice and then succeeded is the early warning that a
    dependency is degrading; logging only the outcome throws that away."""
    recorder = Recorder()
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise http_error(502)
        return "ok"

    resilience.call(flaky, sleep=lambda d: None, on_attempt=recorder)

    assert [c[0] for c in recorder.calls] == [1, 2]
    assert all(c[2] is not None for c in recorder.calls), "each retry should log its delay"


# --- ambiguity and reconciliation ---------------------------------------------

def test_an_ambiguous_failure_that_actually_landed_is_not_retried():
    """The duplicate-generation case. A read timeout on POST /prompt may mean
    ComfyUI is already rendering; retrying would burn a second slot on a
    serial GPU and produce two images for one guest."""
    attempts = []

    def submit():
        attempts.append(1)
        raise requests.exceptions.ReadTimeout()

    result = resilience.call(submit, reconcile=lambda: "prompt-123", sleep=lambda d: None)

    assert result == "prompt-123"
    assert len(attempts) == 1, "reconciliation said it landed, so nothing should be resubmitted"


def test_an_ambiguous_failure_that_did_not_land_is_retried():
    attempts = []

    def submit():
        attempts.append(1)
        if len(attempts) < 2:
            raise requests.exceptions.ReadTimeout()
        return "prompt-123"

    result = resilience.call(submit, reconcile=lambda: None, sleep=lambda d: None)

    assert result == "prompt-123"
    assert len(attempts) == 2


def test_a_reconciliation_that_itself_fails_falls_back_to_retrying():
    """Reconciliation is best-effort -- if ComfyUI is too sick to answer
    /history, that is not a reason to fail a request that might still work."""
    attempts = []

    def submit():
        attempts.append(1)
        if len(attempts) < 2:
            raise requests.exceptions.ReadTimeout()
        return "ok"

    def broken_reconcile():
        raise ConnectionError("history unreachable too")

    assert resilience.call(submit, reconcile=broken_reconcile, sleep=lambda d: None) == "ok"


def test_without_a_reconciler_an_ambiguous_failure_is_just_retried():
    attempts = []

    def submit():
        attempts.append(1)
        if len(attempts) < 3:
            raise requests.exceptions.ReadTimeout()
        return "ok"

    assert resilience.call(submit, sleep=lambda d: None) == "ok"
    assert len(attempts) == 3


# --- the circuit breaker ------------------------------------------------------

def test_the_breaker_opens_after_consecutive_failures_and_then_fails_fast():
    """Once ComfyUI is properly down, the useful answer is immediate. Making
    the sixth guest wait through a full backoff ladder to be told the same
    thing is just a slower error."""
    clock = FakeClock()
    breaker = CircuitBreaker(failure_threshold=3, recovery_seconds=30, clock=clock)

    for _ in range(3):
        breaker.before_call()
        breaker.record_failure()

    assert breaker.state == CircuitBreaker.OPEN
    with pytest.raises(CircuitOpenError) as err:
        breaker.before_call()
    assert "30s" in str(err.value) or "unavailable" in str(err.value)


def test_a_success_resets_the_failure_count():
    """Four failures spread across a busy evening are not the same as four in
    a row; only consecutive failures mean the dependency is down."""
    clock = FakeClock()
    breaker = CircuitBreaker(failure_threshold=3, clock=clock)

    breaker.record_failure()
    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()
    breaker.record_failure()

    assert breaker.state == CircuitBreaker.CLOSED


def test_after_the_recovery_window_exactly_one_probe_is_allowed_through():
    clock = FakeClock()
    breaker = CircuitBreaker(failure_threshold=1, recovery_seconds=30, clock=clock)
    breaker.record_failure()

    clock.advance(29)
    with pytest.raises(CircuitOpenError):
        breaker.before_call()

    clock.advance(2)
    breaker.before_call()          # the probe: allowed
    with pytest.raises(CircuitOpenError):
        breaker.before_call()      # everyone else: still refused


def test_a_failed_probe_restarts_the_recovery_clock():
    """Otherwise a half-open breaker lets one request through every recovery
    window forever, which is a slow leak of guest-facing failures."""
    clock = FakeClock()
    breaker = CircuitBreaker(failure_threshold=1, recovery_seconds=30, clock=clock)
    breaker.record_failure()

    clock.advance(31)
    breaker.before_call()
    breaker.record_failure()       # the probe failed

    clock.advance(5)
    assert breaker.state == CircuitBreaker.OPEN
    with pytest.raises(CircuitOpenError):
        breaker.before_call()


def test_a_successful_probe_closes_the_breaker():
    clock = FakeClock()
    breaker = CircuitBreaker(failure_threshold=1, recovery_seconds=30, clock=clock)
    breaker.record_failure()
    clock.advance(31)

    breaker.before_call()
    breaker.record_success()

    assert breaker.state == CircuitBreaker.CLOSED
    breaker.before_call()          # traffic flows again


def test_an_open_breaker_short_circuits_call_without_invoking_the_function():
    clock = FakeClock()
    breaker = CircuitBreaker(failure_threshold=1, clock=clock)
    breaker.record_failure()
    called = []

    with pytest.raises(CircuitOpenError):
        resilience.call(lambda: called.append(1), breaker=breaker, sleep=lambda d: None)

    assert called == [], "the point of an open breaker is not making the call at all"


def test_breaker_stats_report_what_an_operator_needs():
    breaker = CircuitBreaker(failure_threshold=2, name="comfyui")
    breaker.record_failure()
    breaker.record_failure()

    stats = breaker.stats()
    assert stats["state"] == CircuitBreaker.OPEN
    assert stats["trips"] == 1
    assert stats["name"] == "comfyui"
