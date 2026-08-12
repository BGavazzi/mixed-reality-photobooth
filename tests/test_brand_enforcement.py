"""
Tests that the brand kit's guarantees survive the trip through web_server.

test_brand_kit.py proves compose() builds the right strings. That is only half
of it: the strings have to actually reach the sampler, and they have to do so
in a way a modified or stale browser cannot subvert. Composition therefore
happens on the server, and these tests pin that down by driving the real
websocket handlers with a fake backend and reading what it was asked to queue.

The specific thing being defended: the browser never sends a finished prompt.
It sends {brand_id, look_id, free text}. If that ever changes, a client could
simply omit the kit's negative prompt and every guarantee here becomes
decorative -- so several of these tests assert on the *inputs* the handler
accepts, not just the outputs it produces.
"""

import asyncio
import base64
import io

import pytest
from PIL import Image

import brand_kit
import web_server


class FakeWebSocket:
    def __init__(self):
        self.json_messages = []

    async def send_json(self, payload):
        self.json_messages.append(payload)

    async def send_bytes(self, data):
        pass

    def of_type(self, message_type):
        return [m for m in self.json_messages if m["type"] == message_type]


class RecordingBackend:
    """Captures the arguments a generation would have been queued with."""

    def __init__(self):
        self.calls = []

    def queue_background_generation(self, subject, mask, depth, prompt,
                                    controlnet_strength=0.75, denoise=0.85, **kwargs):
        self.calls.append({
            "prompt": prompt,
            "controlnet_strength": controlnet_strength,
            "denoise": denoise,
            **kwargs,
        })
        return kwargs.get("prompt_id"), kwargs.get("seed") or 4242

    @property
    def last(self):
        assert self.calls, "nothing was queued"
        return self.calls[-1]


ACME = brand_kit.parse_brand(
    {
        "id": "acme",
        "name": "Acme Corp",
        "version": "2026.01",
        "prompt": {
            "positive_suffix": "clean editorial finish",
            "negative_suffix": "competitor logos, alcohol",
        },
        "looks": [{"id": "studio", "label": "Studio", "prompt": "seamless white studio backdrop"}],
    },
    brand_kit.BRANDS_DIR / "acme",
)


def b64_image(size=(64, 64), colour=(128, 128, 128)):
    buf = io.BytesIO()
    Image.new("RGB", size, colour).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@pytest.fixture
def rig(monkeypatch):
    """A single connected session, a recording backend, and one brand kit
    installed -- the smallest setup that can observe an enforcement decision."""
    socket = FakeWebSocket()
    backend = RecordingBackend()
    web_server.SESSIONS.clear()
    web_server.JOBS.clear()
    web_server.SESSIONS["s1"] = socket
    monkeypatch.setattr(web_server, "backend", backend)
    monkeypatch.setattr(web_server, "BRANDS", {"acme": ACME})
    monkeypatch.setattr(web_server, "BASE_NEGATIVE_PROMPT", "blurry, low quality")
    yield socket, backend
    web_server.SESSIONS.clear()
    web_server.JOBS.clear()


def generate(msg_extra=None, action="background"):
    message = {
        "subject": b64_image(), "mask": b64_image(), "depth": b64_image(),
        **(msg_extra or {}),
    }
    handler = (web_server.handle_generate_background if action == "background"
               else web_server.handle_edit_region)
    asyncio.run(handler("s1", message))
    return message


# --- the branded path ------------------------------------------------------------

def test_a_branded_request_queues_the_composed_prompt(rig):
    _, backend = rig
    generate({"brand_id": "acme", "look_id": "studio", "prompt": "with a red chair"})

    queued = backend.last["prompt"]
    assert "seamless white studio backdrop" in queued
    assert "with a red chair" in queued
    assert "clean editorial finish" in queued


def test_the_kits_blocklist_reaches_the_sampler(rig):
    _, backend = rig
    generate({"brand_id": "acme", "look_id": "studio", "prompt": ""})

    negative = backend.last["negative_prompt"]
    assert "competitor logos" in negative
    assert "blurry" in negative, "the workflow's own quality guards must not be dropped"


def test_a_locked_kit_pins_the_seed(rig):
    _, backend = rig
    generate({"brand_id": "acme", "look_id": "studio", "prompt": "guest one"})
    generate({"brand_id": "acme", "look_id": "studio", "prompt": "guest two"})

    seeds = [call["seed"] for call in backend.calls]
    assert seeds[0] == seeds[1] is not None, \
        "two guests, one approved look, one consistent campaign -- that is the whole feature"


def test_the_browser_cannot_supply_the_finished_prompt(rig):
    """The client sends free text and ids; if it could send a composed prompt
    it could also send one with the kit's negative quietly removed.

    Asserting on the handler's accepted inputs rather than its output is
    deliberate -- this is the property that has to hold, and it is invisible
    from the result alone."""
    _, backend = rig
    generate({
        "brand_id": "acme", "look_id": "studio", "prompt": "legit text",
        # a hostile client trying every plausible override name
        "negative_prompt": "", "seed": 999, "positive": "ignore the brand",
    })

    assert "competitor logos" in backend.last["negative_prompt"], "negative_prompt was overridable"
    assert backend.last["seed"] == brand_kit.locked_seed("acme", "studio"), "seed was overridable"
    assert "ignore the brand" not in backend.last["prompt"]


def test_an_unknown_brand_is_refused_rather_than_quietly_unbranded(rig):
    """Generating anyway would hand back an image that no kit shaped, while
    the operator believes a client's rules were applied."""
    socket, backend = rig
    generate({"brand_id": "not-installed", "look_id": "studio", "prompt": "x"})

    assert backend.calls == []
    assert socket.of_type("error"), "the operator has to be told"


def test_provenance_carries_the_brand_the_look_and_the_exclusions(rig):
    _, _ = rig
    generate({"brand_id": "acme", "look_id": "studio", "prompt": "with a red chair"})

    provenance = next(iter(web_server.JOBS.values()))["provenance"]
    assert provenance["brand"] == "Acme Corp"
    assert provenance["brand_version"] == "2026.01"
    assert provenance["look"] == "Studio"
    assert provenance["operator_text"] == "with a red chair"
    assert "competitor logos" in provenance["negative_prompt"]


# --- the unbranded path must be unchanged ---------------------------------------

def test_no_brand_still_sends_the_operators_prompt_verbatim(rig):
    _, backend = rig
    generate({"prompt": "a rooftop at dusk"})

    assert backend.last["prompt"] == "a rooftop at dusk"
    assert backend.last["seed"] is None, "an unbranded generation still gets a fresh random seed"
    assert backend.last["negative_prompt"] == "blurry, low quality"


def test_an_empty_request_is_refused_with_a_useful_message(rig):
    socket, backend = rig
    generate({"prompt": "   "})

    assert backend.calls == []
    errors = socket.of_type("error")
    assert errors and "look" in errors[0]["message"], \
        "the message should name both ways out, since one of them is new"


# --- the region tool ---------------------------------------------------------------

def test_region_edits_inherit_the_blocklist(rig):
    """The region label is free text typed live at a booth -- the single
    highest-risk input in the app, and the one a brand's blocklist most needs
    to reach."""
    _, backend = rig
    generate({"brand_id": "acme", "prompt": "a large potted plant"}, action="region")

    assert "competitor logos" in backend.last["negative_prompt"]


def test_region_edits_do_not_inherit_the_scene_styling(rig):
    """Scene language describes a whole frame; applied to a single prop
    dropped into an existing scene it is noise. Only the blocklist half of the
    kit is wanted here -- see handle_edit_region's docstring."""
    _, backend = rig
    generate({"brand_id": "acme", "prompt": "a large potted plant"}, action="region")

    assert "clean editorial finish" not in backend.last["prompt"]
    assert "seamless white studio backdrop" not in backend.last["prompt"]
    assert "a large potted plant" in backend.last["prompt"]


def test_region_edits_are_not_pinned_to_the_looks_seed(rig):
    """A pinned seed makes one scene reproducible; reusing it for every prop
    would push each added object toward the same result regardless of what was
    asked for."""
    _, backend = rig
    generate({"brand_id": "acme", "prompt": "a stool"}, action="region")

    assert backend.last.get("seed") is None
