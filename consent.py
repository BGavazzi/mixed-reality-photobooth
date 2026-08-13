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
absence of consent visible and blocking, rather than silent and default -- and
make sure the record travels with the photographs instead of living in
somebody's memory of the evening.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

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

    def to_dict(self) -> dict:
        return {
            "basis": self.basis,
            "description": BASES[self.basis],
            "recorded_by": self.recorded_by,
            "note": self.note,
            "recorded_at": self.recorded_at,
        }


def parse(basis: str, recorded_by: str, note: str = "") -> ConsentRecord:
    """Validates an operator's consent declaration, or explains what was wrong.

    `recorded_by` is required and is a person, not a checkbox: consent is
    something someone obtained, and a record with nobody's name on it cannot be
    followed up when a guest later asks for their photographs to be deleted.
    """
    basis = (basis or "").strip()
    recorded_by = (recorded_by or "").strip()

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
