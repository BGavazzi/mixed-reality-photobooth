"""Things that need a person.

The booth could previously do exactly two things with a problem: retry it, or
show the guest an error and forget it happened. There was no third option, and
the third option is the one an operator standing two metres away can actually
act on -- refill the paper, restart ComfyUI, re-shoot the photo where the
rotoscope ate someone's arm.

So this is the handoff queue, and the interesting part is the criteria rather
than the mechanism. A rule for escalating has to be defensible in both
directions:

**Escalate when a human can change the outcome.** ComfyUI unreachable, a photo
that failed after the whole retry ladder, a subject the segmenter could not
find. Someone can restart a service, re-shoot a frame, move a guest a step away
from the backdrop.

**Do not escalate when they cannot.** A guest whose upload is not an image gets
told so directly -- putting that in front of an operator is noise, and a queue
that fills with noise is a queue nobody reads. This is the same distinction the
brand kit already draws between clamping a logo's minimum size (machine-
decidable) and warning about clear space (needs judgement); an alert nobody can
act on is worse than no alert, because it teaches people to ignore the ones
that matter.

**Deduplicate, or the queue is useless.** A fifty-photo batch against a dead
ComfyUI produces fifty identical failures. That is one problem, and it should
read as one line with a count, not fifty lines that bury everything else.
"""

from __future__ import annotations

import itertools
import threading
import time
from dataclasses import dataclass, field

# What can raise an item, and how urgent it is. Kept as a small closed set: an
# open-ended `kind` string turns into a dozen near-synonyms within a month, and
# then nothing can be filtered or counted.
DEPENDENCY_DOWN = "dependency_down"
GENERATION_FAILED = "generation_failed"
BATCH_ITEM_FAILED = "batch_item_failed"
SUBJECT_NOT_FOUND = "subject_not_found"

SEVERITY = {
    DEPENDENCY_DOWN: "high",      # nothing works until someone looks
    GENERATION_FAILED: "medium",  # this guest is stuck, the booth is not
    BATCH_ITEM_FAILED: "low",     # one frame of many; the run continues
    SUBJECT_NOT_FOUND: "medium",  # re-shoot, usually a framing problem
}

_counter = itertools.count(1)
_lock = threading.Lock()
ITEMS: dict[int, "AttentionItem"] = {}

# Beyond this many open items, the oldest resolved-nothing entries are dropped.
# An unbounded list in a long-running booth is a slow memory leak whose growth
# rate is exactly the app's error rate -- worst possible time to run out of RAM.
MAX_OPEN = 200


@dataclass
class AttentionItem:
    id: int
    kind: str
    summary: str
    detail: str = ""
    context: dict = field(default_factory=dict)
    count: int = 1
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    resolved_at: float | None = None
    resolved_by: str = ""

    @property
    def severity(self) -> str:
        return SEVERITY.get(self.kind, "medium")

    @property
    def dedupe_key(self) -> tuple:
        return (self.kind, self.summary)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "severity": self.severity,
            "summary": self.summary,
            "detail": self.detail,
            "context": self.context,
            "count": self.count,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "resolved_at": self.resolved_at,
            "resolved_by": self.resolved_by,
        }


def raise_item(kind: str, summary: str, /, detail: str = "", **context) -> AttentionItem:
    """Files something for a human, merging it with an identical open item.

    `kind` and `summary` are positional-only so that a caller can put anything
    it likes in the context -- including a field of its own called `kind`,
    which is natural for a job and which otherwise collides with this
    signature. That collision was not hypothetical: it raised a TypeError
    inside the failure handler, which swallowed the error message the browser
    was waiting for and turned a failed generation into a hang.

    Merging on (kind, summary) rather than on the full detail is deliberate:
    fifty photos failing with fifty different prompt_ids are still one dead
    ComfyUI, and the operator needs to see one problem with a count of fifty.
    The differing detail of the most recent occurrence is kept, since that is
    the one they will investigate.
    """
    with _lock:
        for item in ITEMS.values():
            if item.resolved_at is None and item.dedupe_key == (kind, summary):
                item.count += 1
                item.last_seen = time.time()
                item.detail = detail or item.detail
                item.context.update(context)
                return item

        item = AttentionItem(id=next(_counter), kind=kind, summary=summary,
                             detail=detail, context=context)
        ITEMS[item.id] = item
        _trim_locked()
        print(f"[attention] {item.severity}: {summary}" + (f" -- {detail}" if detail else ""))
        return item


_DROP_ORDER = {"low": 0, "medium": 1, "high": 2}


def _trim_locked() -> None:
    if len(ITEMS) <= MAX_OPEN:
        return
    # Resolved items go first: they are already handled.
    resolved = sorted((i for i in ITEMS.values() if i.resolved_at is not None),
                      key=lambda i: i.last_seen)
    for item in resolved:
        del ITEMS[item.id]
        if len(ITEMS) <= MAX_OPEN:
            return

    # Then open ones, least important first -- and *severity before age*.
    # Sorting by age alone drops "ComfyUI is down" to make room for the
    # fortieth "one frame failed", which is precisely backwards: the flood of
    # low-severity items is usually a symptom of the high-severity one that
    # would be discarded to make room for it.
    droppable = sorted(ITEMS.values(), key=lambda i: (_DROP_ORDER.get(i.severity, 1),
                                                      i.last_seen))
    for item in droppable[:len(ITEMS) - MAX_OPEN]:
        del ITEMS[item.id]
    print(f"[attention] queue over {MAX_OPEN}; dropped the least severe open items")


def resolve(item_id: int, by: str = "operator") -> AttentionItem | None:
    with _lock:
        item = ITEMS.get(item_id)
        if item is None or item.resolved_at is not None:
            return None
        item.resolved_at = time.time()
        item.resolved_by = by
        return item


def resolve_kind(kind: str, by: str = "system") -> int:
    """Closes every open item of a kind. Used when the app can tell on its own
    that a problem is over -- a circuit breaker closing means ComfyUI answered,
    and leaving "ComfyUI unreachable" on screen after it recovered is how an
    operator learns to distrust the panel."""
    with _lock:
        closed = 0
        for item in ITEMS.values():
            if item.kind == kind and item.resolved_at is None:
                item.resolved_at = time.time()
                item.resolved_by = by
                closed += 1
        return closed


def open_items() -> list[AttentionItem]:
    with _lock:
        items = [i for i in ITEMS.values() if i.resolved_at is None]
    order = {"high": 0, "medium": 1, "low": 2}
    return sorted(items, key=lambda i: (order.get(i.severity, 1), -i.last_seen))


def snapshot() -> dict:
    items = open_items()
    return {
        "open": len(items),
        "highest_severity": items[0].severity if items else None,
        "items": [i.to_dict() for i in items],
    }


def clear() -> None:
    """Test helper. Not exposed over HTTP: an operator resolving items one at a
    time is a record of what happened, and a "clear all" button is how that
    record stops existing."""
    with _lock:
        ITEMS.clear()
