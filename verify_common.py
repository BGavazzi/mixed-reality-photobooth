"""
Shared plumbing for the verify_*.py end-to-end scripts.

All of them need the same three things: somewhere to put screenshots, a
subject photo, and a clear message when that photo isn't there. The default
paths point into `test_images/` and `fallback_examples/`, both gitignored --
so on a fresh clone every one of these scripts used to die on an opaque
Playwright timeout instead of saying "pass me a photo."
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent
OUT_DIR = REPO_ROOT / "verify_out"
DEFAULT_BASE_URL = "http://127.0.0.1:8000"

# Whatever happens to be in the dev machine's working set. Neither directory
# is tracked; both are only defaults so the scripts stay one-word runnable
# there, never a requirement anywhere else.
DEFAULT_PORTRAIT_PHOTO = REPO_ROOT / "test_images" / "pexels_6668809.jpg"
DEFAULT_SECOND_PHOTO = REPO_ROOT / "test_images" / "pexels_8217535.jpg"
DEFAULT_SQUARE_PHOTO = REPO_ROOT / "fallback_examples" / "loft_interior_preview.png"


def out_dir() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR


def resolve_image(path, flag: str, wanted: str) -> str:
    """Exits with an actionable message rather than letting Playwright fail
    on a missing file several steps later."""
    resolved = Path(path)
    if not resolved.exists():
        print(f"Subject photo not found: {resolved}\n"
              f"  This script needs {wanted}.\n"
              f"  The default lives in a gitignored directory, so a fresh clone won't have it.\n"
              f"  Pass your own: {flag} path/to/photo.jpg")
        sys.exit(1)
    return str(resolved)


def preflight(base_url: str):
    """Fails fast and says what to start, instead of timing out in the
    browser on a server that was never running."""
    import requests
    try:
        requests.get(base_url, timeout=5).raise_for_status()
    except Exception as exc:
        print(f"Can't reach {base_url} ({exc}).\n"
              f"Start ComfyUI + web_server.py first (e.g. start_demo.ps1), then re-run.")
        sys.exit(1)
