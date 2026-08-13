"""Recording who agreed to have their photograph processed.

This app points a camera at members of the public and writes their faces to
disk. Until now the only thing it recorded about a person was the seed used to
regenerate the wall behind them -- excellent provenance for the image, none at
all for the human in it.

The design follows the brand kit exactly, and for the same reason. A brand's
locked negative is enforced on the server because a guarantee the client
assembles is a guarantee a stale client can drop. Consent is a stronger version
of the same argument: a checkbox in a page is a claim about a browser, not a
record about a person, so the *server* refuses to start a batch that has no
basis attached, and the basis is written into the manifest that ships with the
images.

**What this is not.** It is not legal advice and it does not make the app
compliant with anything. Consent is a conversation between an operator and a
guest, and no code can have that conversation. What it can do is make the
absence of consent visible, and make sure the record travels with the
photographs instead of living in somebody's memory of the evening.

**Off by default -- a deliberate choice, and an honest one.** The gate started
out mandatory: no basis, no batch. It is now opt-in (`--require-consent`, or
`BOOTH_REQUIRE_CONSENT=1`), because a blocking two-field form in front of every
test run buys less than it costs. The argument for switching it off is worth
writing down rather than hiding in a default:

*The control is weak relative to the leak it claims to address.* An app that
refuses a batch has done nothing about the operator's screenshot, the camera
roll on the device that took the photograph, the projector the output is being
thrown onto, or the SD card in somebody's bag. If any of those are unhandled --
and at an event they usually are -- a server-side form field is a record of
intent, not a boundary. Treating it as the latter is worse than not having it,
because it lets everyone stop thinking at the point where the real exposure
starts.

*So what is left is what it was always actually good at:* writing the basis into
the manifest that ships with the images, so that months later there is an answer
to "what were these people told?" that is not somebody's memory. That is worth
having, and it does not require blocking anything -- an operator who declares a
basis gets it recorded, and a run with no declaration is recorded as exactly
that, `not_recorded`, rather than quietly omitted.

Turning enforcement back on is one flag, and the strict path below is unchanged
and still tested. What changed is the default, not the capability.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import config

# Whether a run with no consent declaration is refused. Off by default; see the
# module docstring. `web_server.py --require-consent` and batch_cli flip it.
REQUIRED = config.env_bool(
    "BOOTH_REQUIRE_CONSENT", False,
    "refuse a batch run that declares no consent basis")

# The bases the booth actually supports, with the wording an operator would
# use. Deliberately a closed set rather than free text: "consent: yes" in a
# free-text field records that somebody typed something, which is worth nothing
# to the reviewer who eventually asks what these people were told.
BASES = {
    "guest_verbal": (
        "The guest was told on the spot what the booth does and agreed out loud."),
    "guest_signed": (
        "The guest signed a release (event waiver, model release)."),
    "event_notice": (
        "A visible notice at the event covers photography and AI processing, "
        "and the guest entered the booth having seen it."),
    "internal_test": (
        "Not a member of the public: staff, stock or synthetic photographs used "
        "for testing. Never for images of real guests."),
}

# Not a basis, and deliberately not in BASES: it is the absence of one. It
# exists so that a manifest can say "nobody recorded this" out loud instead of
# leaving the field off and letting a later reader assume it was lost.
NOT_RECORDED = "not_recorded"

DESCRIPTIONS = {
    **BASES,
    NOT_RECORDED: (
        "No consent basis was recorded for this run. The booth was running with "
        "the consent gate off (the default); this is not a claim that consent "
        "was obtained, and not a claim that it was not."),
}


class ConsentError(ValueError):
    """Raised when a run has no usable consent record. A distinct type so the
    HTTP layer can answer 400 with the operator's actual options rather than a
    generic validation blob."""


@dataclass(frozen=True)
class ConsentRecord:
    basis: str
    recorded_by: str
    note: str = ""
    recorded_at: float = 0.0

    @property
    def recorded(self) -> bool:
        return self.basis != NOT_RECORDED

    def to_dict(self) -> dict:
        return {
            "basis": self.basis,
            "recorded": self.recorded,
            "description": DESCRIPTIONS.get(self.basis, ""),
            "recorded_by": self.recorded_by,
            "note": self.note,
            "recorded_at": self.recorded_at,
        }


def unrecorded() -> ConsentRecord:
    """The record written when nobody declared anything. Still a record: the
    manifest says `not_recorded` rather than omitting the field, because a
    missing key reads as data loss and this is a fact about the run."""
    return ConsentRecord(basis=NOT_RECORDED, recorded_by="", recorded_at=time.time())


def parse(basis: str, recorded_by: str, note: str = "",
          *, required: bool | None = None) -> ConsentRecord:
    """Validates an operator's consent declaration, or explains what was wrong.

    `recorded_by` is required and is a person, not a checkbox: consent is
    something someone obtained, and a record with nobody's name on it cannot be
    followed up when a guest later asks for their photographs to be deleted.

    With `required` false (the default, see the module docstring) a run that
    declares *nothing* is allowed through and recorded as `not_recorded`. A run
    that declares *something* is still validated in full -- a half-filled form
    is the one case worth refusing either way, since a basis with nobody's name
    on it looks like a record and is not one, and a misspelled basis would
    otherwise be silently accepted as a claim nobody can interpret.
    """
    basis = (basis or "").strip()
    recorded_by = (recorded_by or "").strip()
    required = REQUIRED if required is None else required

    if not required and not basis and not recorded_by:
        return unrecorded()

    if not basis:
        raise ConsentError(
            "this run has no consent basis; pass one of: " + ", ".join(sorted(BASES)))
    if basis not in BASES:
        raise ConsentError(
            f"unknown consent basis {basis!r}; expected one of: " + ", ".join(sorted(BASES)))
    if not recorded_by:
        raise ConsentError(
            "consent must name who recorded it -- an operator, not a checkbox, "
            "so a later deletion request has someone to go to")

    return ConsentRecord(basis=basis, recorded_by=recorded_by,
                         note=note.strip(), recorded_at=time.time())


def options() -> list[dict]:
    """The bases, for the UI to render. Served from here rather than duplicated
    in the page, so the list an operator picks from and the list the server
    accepts cannot drift apart."""
    return [{"id": key, "description": value} for key, value in sorted(BASES.items())]
