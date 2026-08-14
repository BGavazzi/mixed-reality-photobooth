"""Every knob this app has, read once, validated, and printed at boot.

The settings were spread across six modules as bare `os.environ.get(...)`
calls, which is fine until it isn't. Three specific problems, all of which
have a wrong answer today:

**Nothing validates.** `COMFY_RETRY_ATTEMPTS=four` crashed at *import* with a
raw `ValueError` traceback and no mention of which variable was wrong.
`BATCH_RETAIN_DAYS=-1` was worse -- it validated fine and silently meant "keep
strangers' photographs forever", because the sweep treats a non-positive
window as "retention off". A typo should not be able to quietly disable a
privacy control.

**Nothing reports.** An operator debugging a booth had no way to ask what the
process actually believes its settings are, short of reading source and then
reading their own shell history. "Is retention even on?" is not a question
that should require a code reading.

**Nothing enumerates.** There was no list of the knobs. They were discoverable
by grepping for `os.environ`, which means new ones arrive undocumented and old
ones stay after the code stops reading them.

So: every setting is declared here, parsed through a typed reader that
validates a range, recorded in a registry, and echoed at startup. Reading an
undeclared variable is now the unusual thing rather than the normal one.

**What this is not:** a settings framework. There is no file format, no
profiles, no hot reload. Environment variables and CLI flags are what a booth
and a container both already speak, and adding a config file would mean
adding precedence rules to answer a question nobody has asked.

**Two files deliberately do not use this**, and should stay that way.
`doctor.py` imports nothing from the app on purpose -- a preflight check that
crashes on a broken install cannot tell you the install is broken. And
`chaos_comfy.py` has to run when the app does not, since its whole job is
standing in for a dependency. Both read `os.environ` directly, which is the
right call for a diagnostic and a test double.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

VERSION = "1.0.0"


@dataclass
class Setting:
    """One knob, and where its current value came from. `source` matters more
    than it looks: "this is the default" and "someone set this to the same
    value as the default" are the same number and very different situations
    when a booth is behaving oddly."""
    name: str
    value: object
    default: object
    source: str  # "default" | "env" | "flag"
    help: str = ""

    @property
    def overridden(self) -> bool:
        return self.source != "default"


REGISTRY: dict[str, Setting] = {}


class ConfigError(ValueError):
    """A setting was present but unusable. Raised at import with the variable's
    name, its value and what was expected -- the three things a stack trace
    from `int()` does not tell you."""


def _record(name, value, default, source, help_text) -> object:
    REGISTRY[name] = Setting(name=name, value=value, default=default,
                             source=source, help=help_text)
    return value


def _raw(name: str) -> str | None:
    raw = os.environ.get(name)
    return raw if raw is not None and raw.strip() != "" else None


def env_str(name: str, default: str, help_text: str = "") -> str:
    raw = _raw(name)
    return _record(name, raw if raw is not None else default, default,
                   "env" if raw is not None else "default", help_text)


def env_int(name: str, default: int, help_text: str = "", *,
            minimum: int | None = None, maximum: int | None = None) -> int:
    raw = _raw(name)
    if raw is None:
        return _record(name, default, default, "default", help_text)
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name}={raw!r} is not a whole number") from exc
    _check_range(name, value, minimum, maximum)
    return _record(name, value, default, "env", help_text)


def env_float(name: str, default: float, help_text: str = "", *,
              minimum: float | None = None, maximum: float | None = None) -> float:
    raw = _raw(name)
    if raw is None:
        return _record(name, default, default, "default", help_text)
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name}={raw!r} is not a number") from exc
    _check_range(name, value, minimum, maximum)
    return _record(name, value, default, "env", help_text)


def env_bool(name: str, default: bool, help_text: str = "") -> bool:
    raw = _raw(name)
    if raw is None:
        return _record(name, default, default, "default", help_text)
    lowered = raw.strip().lower()
    if lowered in ("1", "true", "yes", "on"):
        return _record(name, True, default, "env", help_text)
    if lowered in ("0", "false", "no", "off"):
        return _record(name, False, default, "env", help_text)
    # Deliberately an error rather than falsy. The old `in ("1","true","yes")`
    # test read `BATCH_KEEP_INTERMEDIATES=True` (Python's repr, an easy thing
    # to paste) as *off*, which is the direction that loses someone's data
    # while looking like it was configured.
    raise ConfigError(
        f"{name}={raw!r} is not a yes/no value; use one of 1/0, true/false, yes/no, on/off")


def _check_range(name, value, minimum, maximum) -> None:
    if minimum is not None and value < minimum:
        raise ConfigError(f"{name}={value} is below the minimum of {minimum}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{name}={value} is above the maximum of {maximum}")


def set_flag(name: str, value: object) -> None:
    """Records a CLI flag's override so the boot banner and /healthz show the
    value actually in force. Without this the banner would confidently print
    the environment's value while the process ran on the flag's."""
    existing = REGISTRY.get(name)
    REGISTRY[name] = Setting(name=name, value=value,
                             default=existing.default if existing else None,
                             source="flag", help=existing.help if existing else "")


def build_sha() -> str:
    """Best effort, and explicitly unknown when it cannot be determined.

    Tried in the order that is actually reliable: an env var (how a Docker
    build passes it in, since the image has no .git), then git. Never raises --
    a missing sha is a worse answer than a version, not a reason to refuse to
    start."""
    from_env = _raw("BOOTH_BUILD_SHA")
    if from_env:
        return from_env
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent, capture_output=True, text=True, timeout=2)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:                                   # noqa: BLE001
        pass
    return "unknown"


def version_info() -> dict:
    return {"version": VERSION, "build": build_sha()}


def describe() -> list[dict]:
    """Every declared setting and its effective value, for the boot banner and
    the health endpoint. Sorted so two booths' output can be diffed."""
    return [
        {"name": s.name, "value": s.value, "default": s.default,
         "source": s.source, "help": s.help}
        for s in sorted(REGISTRY.values(), key=lambda s: s.name)
    ]


def banner() -> str:
    lines = [f"[config] photobooth {VERSION} ({build_sha()})"]
    for setting in describe():
        marker = " * " if setting["source"] != "default" else "   "
        lines.append(f"[config]{marker}{setting['name']} = {setting['value']!r}"
                     + (f"  ({setting['source']})" if setting["source"] != "default" else ""))
    lines.append("[config]  * = overridden")
    return "\n".join(lines)
