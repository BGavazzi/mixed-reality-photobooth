"""
Install doctor: says exactly what's missing and the command that fixes it.

    python doctor.py                 # photo booth (the primary demo)
    python doctor.py --resolume      # also check the OSC/Spout bridge
    python doctor.py --all           # everything, including optional backends

Deliberately imports nothing outside the standard library. A diagnostic that
needs the dependencies installed is useless for diagnosing a broken install,
so this has to run on a bare interpreter -- before `pip install`, before the
venv is even populated.

Exit code is 0 when nothing is broken (warnings are fine), 1 when something
would actually stop the app running.

All output is plain ASCII on purpose. Windows consoles default to cp1252,
and a diagnostic that prints mojibake -- or worse, dies on UnicodeEncodeError
while reporting someone's broken install -- is not much of a diagnostic. The
rest of the project fixes this by reconfiguring the stream at the entry point
(console_encoding.py); this file avoids needing the fix at all.
"""

import argparse
import importlib.util
import json
import os
import platform
import socket
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).parent
MIN_PYTHON = (3, 10)
DEFAULT_COMFY_ADDRESS = os.environ.get("COMFY_ADDRESS", "127.0.0.1:8188")
PHOTOSHOOT_WORKFLOW = REPO_ROOT / "workflows" / "photoshoot_bg_api.json"

OK, WARN, FAIL = "ok", "warn", "fail"

# module name -> (pip name, what breaks without it)
CORE_PACKAGES = {
    "fastapi": ("fastapi", "the web server itself"),
    "uvicorn": ("uvicorn[standard]", "the web server itself"),
    "websockets": ("websockets", "the live ComfyUI progress/preview relay"),
    "multipart": ("python-multipart", "photo upload (FastAPI won't even start without it)"),
    "requests": ("requests", "every call to ComfyUI's REST API"),
    "PIL": ("Pillow", "all image handling"),
    "numpy": ("numpy", "the CV pipeline"),
    "cv2": ("opencv-python-headless", "mask cleanup (connected components)"),
    "matplotlib": ("matplotlib", "OpenPose hand rendering inside controlnet_aux"),
    "rembg": ("rembg", "the rotoscope stage"),
    "onnxruntime": ("onnxruntime", "rembg's runtime"),
    "controlnet_aux": ("controlnet_aux", "the pose and depth stages"),
}

RESOLUME_PACKAGES = {
    "pythonosc": ("python-osc", "receiving Resolume clip triggers"),
    "SpoutGL": ("SpoutGL", "publishing a Spout source (Windows only)"),
    "OpenGL": ("PyOpenGL", "publishing a Spout source (Windows only)"),
}

BACKEND_PACKAGES = {
    "runwayml": ("runwayml", "the --backend runway option"),
    "jwt": ("PyJWT", "the --backend kling option"),
}


class Finding:
    def __init__(self, level, title, detail="", fix=""):
        self.level, self.title, self.detail, self.fix = level, title, detail, fix


# --- pure checks (no I/O, so they're unit-testable) --------------------------

def check_python_version(version_info=None) -> Finding:
    version_info = version_info or sys.version_info
    shown = ".".join(str(p) for p in version_info[:3])
    if tuple(version_info[:2]) < MIN_PYTHON:
        return Finding(
            FAIL, f"Python {shown}",
            f"this project needs Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer "
            f"(it uses `X | None` type syntax and asyncio.to_thread)",
            f"install Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ and recreate the venv",
        )
    return Finding(OK, f"Python {shown}")


def missing_packages(group: dict, is_installed) -> list:
    """Returns [(module, pip_name, why)] for everything absent. `is_installed`
    is injected so this is testable without touching the environment."""
    return [(module, pip_name, why)
            for module, (pip_name, why) in group.items()
            if not is_installed(module)]


def packages_finding(label: str, group: dict, is_installed, requirements_file: str,
                      level_if_missing=FAIL) -> Finding:
    absent = missing_packages(group, is_installed)
    if not absent:
        return Finding(OK, f"{label}: all {len(group)} packages present")
    lines = [f"  - {pip_name:<26} {why}" for _module, pip_name, why in absent]
    return Finding(
        level_if_missing,
        f"{label}: {len(absent)} of {len(group)} packages missing",
        "\n".join(lines),
        f"pip install -r {requirements_file}",
    )


def ascii_only(text: str) -> str:
    """Strings that come from outside this file -- OS error messages above
    all -- are in the system locale, which on this machine renders Portuguese
    accents as mojibake in a cp1252 console. Everything printed goes through
    here so one localised errno can't undo the file's ASCII guarantee."""
    return str(text).encode("ascii", "replace").decode("ascii")


def check_opencv_build(cv2_module, wants_viewer: bool):
    """Returns a Finding, or None when there's nothing worth saying.

    The GUI build is required by exactly one optional file. Reporting a
    headless build as a failure when the user isn't running the Spout viewer
    is how this turned into a mandatory `--force-reinstall` step that everyone
    had to run and nobody understood.
    """
    if cv2_module is None:
        return None  # the package-group check already reports it; don't say it twice
    has_gui = hasattr(cv2_module, "imshow")
    build = "GUI" if has_gui else "headless"
    if not wants_viewer:
        return Finding(OK, f"opencv: {build} build (fine -- the photo booth needs no window)")
    if has_gui:
        return Finding(OK, "opencv: GUI build (spout_viewer.py can open its window)")
    return Finding(
        WARN, "opencv: headless build, but spout_viewer.py needs a window",
        "controlnet_aux depends on opencv-python-headless, and both builds provide the\n"
        "  same `cv2` -- whichever installs last wins. Everything except the viewer works.",
        "pip install --force-reinstall opencv-python",
    )


def check_models(available: dict, wanted: dict) -> Finding:
    """`available` maps loader input name -> list ComfyUI reports it can see;
    `wanted` maps the same key -> the name this repo's workflow asks for."""
    absent = {key: name for key, name in wanted.items()
              if name and name not in available.get(key, [])}
    if not absent:
        return Finding(OK, f"models: all {len(wanted)} present in ComfyUI")
    lines = [f"  - {name}" for name in absent.values()]
    return Finding(
        FAIL, f"models: {len(absent)} missing from ComfyUI",
        "\n".join(lines) + "\n  Generation will fail partway through with these absent.",
        "see the README's Models section for download links",
    )


# --- probes (the I/O half) ---------------------------------------------------

def is_installed(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def import_optional(module_name):
    if not is_installed(module_name):
        return None
    try:
        return __import__(module_name)
    except Exception:
        return None


def check_virtualenv() -> Finding:
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    if in_venv:
        return Finding(OK, f"virtualenv: active ({Path(sys.prefix).name})")
    return Finding(
        WARN, "virtualenv: not active",
        "  Installing into the system interpreter works but is easy to regret --\n"
        "  this project pulls in torch and a few GB of model runtime.",
        "python -m venv .venv  &&  .venv\\Scripts\\activate   (Windows)\n"
        "       python -m venv .venv  &&  source .venv/bin/activate   (macOS/Linux)",
    )


def fetch_json(url: str, timeout: float = 5.0):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def check_comfyui(address: str) -> tuple:
    """Returns (Finding, reachable). Everything model-related depends on this,
    so the caller skips those checks entirely when it's down rather than
    printing a cascade of failures that all have the same cause."""
    host, _, port = address.partition(":")
    try:
        with socket.create_connection((host, int(port or 8188)), timeout=3):
            pass
    except OSError as exc:
        return Finding(
            WARN, f"ComfyUI: not reachable at {address}",
            f"  {ascii_only(exc)}\n"
            "  Not a problem if you haven't started it yet -- but nothing can generate\n"
            "  until it's up. The analysis stages (rotoscope/pose/depth) work without it.",
            "start_demo.ps1   (or run ComfyUI yourself with --preview-method auto)",
        ), False
    try:
        stats = fetch_json(f"http://{address}/system_stats")
        devices = stats.get("devices") or [{}]
        name = devices[0].get("name", "unknown device")
        vram = devices[0].get("vram_total")
        detail = f"  {ascii_only(name)}" + (f", {vram // (1024**3)}GB VRAM" if isinstance(vram, int) else "")
        return Finding(OK, f"ComfyUI: reachable at {address}", detail), True
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return Finding(
            WARN, f"ComfyUI: port {address} is open but /system_stats failed",
            f"  {ascii_only(exc)}\n  Something else may be listening on that port.",
        ), False


def workflow_model_names() -> dict:
    """What this repo's photoshoot workflow actually asks for, read by node
    class_type rather than by node ID (a re-export renumbers the IDs)."""
    wanted = {"ckpt_name": None, "control_net_name": None}
    try:
        workflow = json.loads(PHOTOSHOOT_WORKFLOW.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return wanted
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs", {})
        if node.get("class_type") == "CheckpointLoaderSimple":
            wanted["ckpt_name"] = inputs.get("ckpt_name")
        elif node.get("class_type") == "ControlNetLoader":
            wanted["control_net_name"] = inputs.get("control_net_name")
    return wanted


def available_model_names(address: str) -> dict:
    """Asks ComfyUI which model files it can actually see on disk -- the same
    enum the loader node itself would validate against."""
    available = {}
    for node, key in (("CheckpointLoaderSimple", "ckpt_name"), ("ControlNetLoader", "control_net_name")):
        try:
            info = fetch_json(f"http://{address}/object_info/{node}")
            available[key] = info[node]["input"]["required"][key][0]
        except (urllib.error.URLError, OSError, ValueError, KeyError, IndexError):
            available[key] = []
    return available


# --- reporting ---------------------------------------------------------------

SYMBOL = {OK: "  OK ", WARN: "WARN ", FAIL: "FAIL "}


def report(findings) -> int:
    print()
    for finding in findings:
        print(f"{SYMBOL[finding.level]} {finding.title}")
        if finding.detail:
            print(finding.detail if finding.detail.startswith("  ") else f"  {finding.detail}")
        if finding.fix and finding.level != OK:
            for i, line in enumerate(finding.fix.split("\n")):
                print(f"  {'fix:' if i == 0 else '    '} {line.strip()}")
        print()

    failures = [f for f in findings if f.level == FAIL]
    warnings = [f for f in findings if f.level == WARN]
    print("-" * 64)
    if failures:
        print(f"{len(failures)} problem(s) will stop the app running. Fix those first.")
    elif warnings:
        print(f"Ready to run. {len(warnings)} optional thing(s) noted above.")
        print("  powershell -ExecutionPolicy Bypass -File start_demo.ps1")
    else:
        print("Everything checks out.")
        print("  powershell -ExecutionPolicy Bypass -File start_demo.ps1")
    print("-" * 64)
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resolume", action="store_true", help="also check the OSC/Spout bridge")
    parser.add_argument("--backends", action="store_true", help="also check the hosted Runway/Kling backends")
    parser.add_argument("--all", action="store_true", help="check everything")
    parser.add_argument("--comfy", default=DEFAULT_COMFY_ADDRESS, help="ComfyUI host:port")
    args = parser.parse_args()
    want_resolume = args.resolume or args.all
    want_backends = args.backends or args.all

    print(f"mixed-reality-photobooth doctor -- {platform.system()} {platform.release()}, "
          f"{sys.implementation.name} {platform.python_version()}")
    print(f"interpreter: {sys.executable}")

    findings = [check_python_version(), check_virtualenv()]
    findings.append(packages_finding("photo booth", CORE_PACKAGES, is_installed, "requirements.txt"))
    opencv_finding = check_opencv_build(import_optional("cv2"), wants_viewer=want_resolume)
    if opencv_finding:
        findings.append(opencv_finding)

    if want_resolume:
        group = dict(RESOLUME_PACKAGES)
        if sys.platform != "win32":  # Spout is Windows-only; absence isn't a fault elsewhere
            group.pop("SpoutGL", None)
            group.pop("OpenGL", None)
            findings.append(Finding(WARN, f"Spout: unavailable on {sys.platform}",
                                    "  Spout is Windows-only. Everything else in the bridge works."))
        findings.append(packages_finding("resolume bridge", group, is_installed,
                                          "requirements-resolume.txt", level_if_missing=WARN))
    if want_backends:
        findings.append(packages_finding("hosted backends", BACKEND_PACKAGES, is_installed,
                                          "requirements-backends.txt", level_if_missing=WARN))

    comfy_finding, reachable = check_comfyui(args.comfy)
    findings.append(comfy_finding)
    if reachable:
        findings.append(check_models(available_model_names(args.comfy), workflow_model_names()))

    sys.exit(report(findings))


if __name__ == "__main__":
    main()
