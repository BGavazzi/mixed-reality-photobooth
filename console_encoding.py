"""
UTF-8 console setup, called explicitly by the entry points that need it.

Windows consoles default to cp1252, which raises `UnicodeEncodeError` on
most non-ASCII prompt text (accents, emoji, non-English words). Since both
apps log prompts verbatim, that turns "a prompt with an accent in it" into a
crash partway through a request.

This used to run as a side effect of importing `backends.comfy`, which meant
anything that touched the backend -- a test, a script, another library --
silently reconfigured the process's stdout and stderr as a condition of
importing it. Reconfiguring global streams is a decision for whoever owns
the process, so it belongs at the entry points and nowhere else.
"""

import sys


def use_utf8_console():
    """Idempotent; safe to call from any entry point."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
