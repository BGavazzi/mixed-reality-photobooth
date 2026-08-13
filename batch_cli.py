"""
Command-line client for batch mode.

    python batch_cli.py photos/ --brand aurora --look coastline -o shoot.zip

Deliberately a *client* rather than a second implementation: it posts to the
same /api/batch the browser uses, so there is one code path through the queue,
the brand kit and the compositor. A CLI that reimplemented the pipeline would
be a second thing to keep correct, and the first to quietly drift.

Why have it at all, then: a batch is the one operation here that genuinely
does not want a browser. It runs for half an hour, it outlives the page, and
the natural place to start it is a directory of files that already exist on
disk -- which is a shell problem, not a drag-and-drop problem.
"""

import argparse
import sys
import time
from pathlib import Path

import requests

from console_encoding import use_utf8_console

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
POLL_SECONDS = 3


def collect_photos(inputs: list[str]) -> list[Path]:
    """Accepts files, directories, or a mix. Directories are read one level
    deep and sorted, so `photos/` behaves the way `photos/*` would in a shell
    that expands globs -- and identically in PowerShell, which doesn't."""
    photos: list[Path] = []
    for entry in inputs:
        path = Path(entry)
        if path.is_dir():
            photos.extend(sorted(p for p in path.iterdir()
                                 if p.suffix.lower() in IMAGE_SUFFIXES))
        elif path.is_file():
            photos.append(path)
        else:
            print(f"skipping {path}: not a file or directory", file=sys.stderr)
    return photos


def start(server: str, photos: list[Path], args) -> dict:
    files = [("files", (p.name, p.read_bytes(), "application/octet-stream")) for p in photos]
    data = {
        "brand_id": args.brand or "",
        "look_id": args.look or "",
        "prompt": args.prompt or "",
        "controlnet_strength": str(args.controlnet_strength),
        "denoise": str(args.denoise),
        "consent_basis": args.consent or "",
        "consent_by": args.consent_by or "",
        "consent_note": args.consent_note or "",
    }
    resp = requests.post(f"http://{server}/api/batch", files=files, data=data, timeout=300)
    if resp.status_code >= 400:
        # The server's message is the useful part (unknown brand, no look
        # selected, too many files); a bare status code would send someone
        # reading source instead of reading the sentence.
        raise SystemExit(f"batch rejected ({resp.status_code}): "
                         f"{resp.json().get('detail', resp.text)}")
    return resp.json()


def wait(server: str, run_id: str, total: int) -> dict:
    """Polls until every item is terminal, printing a single updating line.

    Polling rather than a websocket because this is the coarse-grained view --
    the per-step denoise progress the browser shows is not useful across fifty
    photos, and one line that says 12/50 is.
    """
    last = None
    while True:
        status = requests.get(f"http://{server}/api/batch/{run_id}", timeout=30).json()
        counts = status["counts"]
        line = (f"  {counts['done']:>3}/{total} done"
                f"  {counts['generating']} generating"
                f"  {counts['analyzing']} analyzing"
                f"  {counts['pending']} queued"
                + (f"  {counts['failed']} failed" if counts["failed"] else ""))
        if line != last:
            print(line, end="\r", flush=True)
            last = line
        if status["finished"]:
            print()
            return status
        time.sleep(POLL_SECONDS)


def download(server: str, run_id: str, out: Path):
    resp = requests.get(f"http://{server}/api/batch/{run_id}/download", timeout=300)
    if resp.status_code >= 400:
        raise SystemExit(f"download failed ({resp.status_code}): {resp.text}")
    out.write_bytes(resp.content)
    print(f"wrote {out}  ({len(resp.content) / 1024 / 1024:.1f} MB)")


def main():
    use_utf8_console()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", help="photo files and/or directories")
    parser.add_argument("--server", default="127.0.0.1:8000", help="web_server.py host:port")
    parser.add_argument("--brand", help="brand kit id (see /api/config)")
    parser.add_argument("--look", help="approved look id within that kit")
    parser.add_argument("--prompt", help="extra direction, added to the approved look")
    parser.add_argument("--controlnet-strength", type=float, default=0.75)
    parser.add_argument("--denoise", type=float, default=0.85)
    # Required by the server, not defaulted here. A CLI that quietly supplied
    # "internal_test" for anyone who forgot the flag would turn a deliberate
    # declaration into a formality, which is the failure mode this is meant to
    # prevent. See consent.py; `GET /api/config` lists the accepted bases.
    parser.add_argument("--consent", help="consent basis (see /api/config for the list)")
    parser.add_argument("--consent-by", help="who recorded that consent")
    parser.add_argument("--consent-note", default="", help="optional free-text detail")
    parser.add_argument("-o", "--out", type=Path, help="write the zip here when finished")
    parser.add_argument("--no-wait", action="store_true",
                        help="queue the run and exit, instead of following it")
    args = parser.parse_args()

    photos = collect_photos(args.inputs)
    if not photos:
        raise SystemExit("no images found in the given paths")

    print(f"queueing {len(photos)} photo(s) to {args.server}"
          + (f", brand={args.brand} look={args.look}" if args.brand else ", no brand kit"))
    run = start(args.server, photos, args)
    print(f"run {run['run_id']}: {run['queued']} queued"
          + (f", {run['counts']['failed']} rejected" if run["counts"]["failed"] else ""))

    if args.no_wait:
        print(f"follow it at http://{args.server}/api/batch/{run['run_id']}")
        return 0

    status = wait(args.server, run["run_id"], run["total"])
    for item in status["items"]:
        if item["status"] == "failed":
            print(f"  failed: {item['filename']} -- {item['error']}")

    if args.out:
        download(args.server, run["run_id"], args.out)
    else:
        print(f"download: http://{args.server}/api/batch/{run['run_id']}/download")
    # Non-zero if nothing succeeded, so a scripted run can tell total failure
    # from partial -- a batch where 49 of 50 worked is a success with a note.
    return 0 if status["counts"]["done"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
