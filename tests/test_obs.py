"""
Tests for the one thing that makes a log worth having on a busy booth: being
able to pull one photo's lines out of four guests' interleaved output.
"""

import asyncio
import json

import pytest

import obs


@pytest.fixture(autouse=True)
def capture(monkeypatch):
    monkeypatch.setattr(obs, "SINK", [])
    yield obs.SINK


def test_lines_logged_inside_a_binding_carry_the_job_id(capture):
    with obs.bind("photo-42"):
        obs.log("queue", "submitted")
    obs.log("queue", "unrelated")

    assert [r.get("job") for r in capture] == ["photo-42", None]


def test_bindings_restore_rather_than_clear(capture):
    """They nest: a batch item's work happens inside its run's context, and
    clearing on exit would silently drop the outer attribution for everything
    after the first inner block closed."""
    with obs.bind("outer"):
        with obs.bind("inner"):
            obs.log("batch", "inner work")
        obs.log("batch", "back outside")

    assert [r["job"] for r in capture] == ["inner", "outer"]


def test_the_binding_survives_a_hop_into_a_thread(capture):
    """The property the whole design rests on. The queue binds the id and then
    calls the backend via asyncio.to_thread; the retry ladder that logs in
    there has never heard of a job and must not have to."""
    async def scenario():
        with obs.bind("photo-7"):
            await asyncio.to_thread(obs.log, "comfy", "call failed")

    asyncio.run(scenario())

    assert capture[0]["job"] == "photo-7"


def test_fields_that_are_none_are_dropped_not_rendered(capture):
    """An absent field and a field whose value is the string "None" look
    identical in a grep, and only one of them means anything."""
    obs.log("relay", "event", prompt_id="p1", node=None)

    assert "node" not in capture[0]
    assert capture[0]["prompt_id"] == "p1"


def test_the_human_format_reads_like_the_old_prefixes():
    """The app logged `[queue] ...` for years and people have greps for it."""
    line = obs.render({"ts": 1, "channel": "queue", "msg": "started", "workers": 2})

    assert line.startswith("[queue] started")
    assert "workers=2" in line


def test_the_human_format_shortens_the_job_id():
    line = obs.render({"ts": 1, "channel": "queue", "msg": "submitted",
                       "job": "abcdef12-3456-7890-abcd-ef1234567890"})

    assert "job=abcdef12 " in line, "36 characters of uuid is mostly noise to a reader"


def test_json_format_keeps_the_whole_id(monkeypatch):
    """Because there the reader is a program, and a truncated id cannot be
    joined against the manifest or ComfyUI's own history."""
    monkeypatch.setattr(obs, "JSON_LOGS", True)

    parsed = json.loads(obs.render({"ts": 1, "channel": "queue", "msg": "submitted",
                                    "job": "abcdef12-3456"}))

    assert parsed["job"] == "abcdef12-3456"
