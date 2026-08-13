"""Log lines you can follow one photo through.

The app logged with `print()`, which is not the problem. The problem is that
nothing tied the lines together. A single guest's photo produces output from
the HTTP handler, the queue worker, the retry ladder, the ComfyUI relay, the
compositor and possibly the attention queue -- on a busy booth, interleaved
with three other guests' lines, and with no field in common. Answering "what
happened to the photo that failed at 21:40?" meant reading the whole log and
guessing which lines belonged together.

So there is one identifier, and it is the one that already exists: the
`prompt_id`, which this app generates itself before submission (see
`web_server.enqueue_generation`) and which ComfyUI echoes back on every event.
It is already the key of the `JOBS` dict and already travels to ComfyUI and
back, so making it the correlation id costs nothing and means the log agrees
with the rest of the system instead of inventing a parallel identity.

`bind()` puts it in a context variable, so code that logs does not have to
thread an id through five call signatures to say which photo it is talking
about. Context variables are inherited by tasks created inside the context,
which is what makes this work across `asyncio.to_thread` and the queue's
workers without any of them knowing about it.

**Format.** Default is human-readable key=value, because the primary reader is
a person watching a terminal at an event. `BOOTH_LOG_FORMAT=json` switches to
one JSON object per line for when something is shipping these somewhere. Both
carry the same fields; neither is the "real" one with the other as a
degradation.

**Deliberately not here:** log levels, handlers, rotation, a `logging` config.
The app has one output and one consumer. A level filter would be the second
thing to configure wrong at 2am, and the lines that exist are all ones
somebody wants to see.
"""

from __future__ import annotations

import contextvars
import json
import os
import sys
import time
from contextlib import contextmanager

# The current photo, if any. A context variable rather than a parameter so the
# retry ladder and the compositor -- neither of which has any business knowing
# about jobs -- still produce attributable lines.
_job_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("job_id", default=None)

# Read directly rather than through config.py: obs is imported by config's
# consumers and by tests that must not need an app import, and one string with
# two valid values does not need range checking.
JSON_LOGS = os.environ.get("BOOTH_LOG_FORMAT", "").strip().lower() == "json"

# Test hook. Set to a list to capture instead of printing.
SINK: list[dict] | None = None


def current_job() -> str | None:
    return _job_id.get()


@contextmanager
def bind(job_id: str | None):
    """Attributes everything logged inside the block to one photo.

    Restores the previous value rather than clearing it, because these nest:
    a batch item's work happens inside a run's context, and clearing on exit
    would silently drop the outer attribution for everything after the first
    inner block closes.
    """
    token = _job_id.set(job_id)
    try:
        yield job_id
    finally:
        _job_id.reset(token)


def log(channel: str, message: str, **fields) -> dict:
    """One line. `channel` is the subsystem -- queue, relay, comfy, batch --
    and matches the bracketed prefixes the app already used, so existing greps
    keep working."""
    record = {"ts": round(time.time(), 3), "channel": channel, "msg": message}
    job_id = _job_id.get()
    if job_id:
        record["job"] = job_id
    # Dropped rather than rendered as "None": an absent field and a field whose
    # value is the string None look identical in a grep, and only one of them
    # means anything.
    record.update({k: v for k, v in fields.items() if v is not None})

    if SINK is not None:
        SINK.append(record)
        return record
    print(render(record), file=sys.stdout, flush=True)
    return record


def render(record: dict) -> str:
    if JSON_LOGS:
        return json.dumps(record, default=str)
    head = f"[{record['channel']}]"
    if record.get("job"):
        # Short form: a uuid4 is 36 characters of mostly noise, and the first
        # eight are already unique across any one booth's evening. The full id
        # stays in the JSON form, where something is parsing rather than
        # reading.
        head += f" job={record['job'][:8]}"
    extras = " ".join(
        f"{k}={_render_value(v)}" for k, v in record.items()
        if k not in ("ts", "channel", "msg", "job"))
    return f"{head} {record['msg']}" + (f" {extras}" if extras else "")


def _render_value(value) -> str:
    text = str(value)
    return f'"{text}"' if " " in text else text
