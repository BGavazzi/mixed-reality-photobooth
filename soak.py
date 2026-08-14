"""
Push photos through the running server until something interesting happens.

Every claim this repo makes about load is currently an argument. The queue is
bounded "because fifty photos is half an hour of GPU"; two workers are enough
"because the GPU is serial and what overlaps is the uploads". Both are
reasonable and neither is measured, and a reasonable unmeasured claim about
performance is a guess with a footnote.

So this drives the real HTTP API -- not the internals -- for N photos and
reports what actually happened: throughput, where the time went, whether
memory grew, and what the app did when asked for more than it could take.

Run it against `chaos_comfy.py` rather than a GPU. The point is the app's
behaviour under load, and a real ComfyUI would make every run a 30-minute
measurement of somebody else's sampler:

    python chaos_comfy.py --port 8188 --render-seconds 0.2 --failure-rate 0
    python web_server.py --port 8010
    python soak.py --server 127.0.0.1:8010 --photos 100

**What the numbers mean, and do not.** Stage timings come from the run
manifest and the batch status endpoint, so they are wall-clock as the app saw
them, including queueing. With a fake ComfyUI the *generation* numbers measure
this app's overhead and nothing about image quality or GPU speed -- that is
the point, since GPU speed is the one thing here that is not this app's
behaviour. The analysis stage is real CPU work either way.

**Memory is sampled from the server process, not this one.** A load generator
measuring its own RSS would be measuring the wrong process, which is an easy
and completely useless mistake.
"""

from __future__ import annotations

import argparse
import io
import json
import statistics
import sys
import time
from dataclasses import dataclass, field

import requests
from PIL import Image, ImageDraw


def synthetic_photo(index: int, size=(768, 1024)) -> bytes:
    """A subject-shaped blob on a background. Not a photograph, deliberately:
    a soak run should not need a directory of real people's faces to exist,
    and the segmenter's accuracy is not what is being measured."""
    image = Image.new("RGB", size, (40, 60 + (index * 7) % 120, 90))
    draw = ImageDraw.Draw(image)
    cx, cy = size[0] // 2, size[1] // 2
    draw.ellipse([cx - 120, cy - 320, cx + 120, cy - 80], fill=(210, 180, 160))
    draw.rectangle([cx - 170, cy - 100, cx + 170, size[1]], fill=(200, 170, 150))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@dataclass
class Result:
    submitted: int = 0
    refused: int = 0
    done: int = 0
    failed: int = 0
    started_at: float = 0.0
    finished_at: float = 0.0
    item_seconds: list[float] = field(default_factory=list)
    rss_samples: list[float] = field(default_factory=list)
    readiness: list[str] = field(default_factory=list)
    last_line: str = ""

    @property
    def elapsed(self) -> float:
        return self.finished_at - self.started_at


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((p / 100) * (len(ordered) - 1))))
    return ordered[index]


def server_rss_mb(server: str) -> float | None:
    """Best effort, and honest when it cannot: psutil is not a dependency of
    this app, and a soak run that refuses to start over a memory sample would
    be trading the measurement for the metric."""
    try:
        import psutil
    except ImportError:
        return None
    try:
        resp = requests.get(f"http://{server}/readyz", timeout=5)
        # The server does not report its own pid, so match on the port it is
        # listening on -- which is the one thing we do know about it.
        port = int(server.rsplit(":", 1)[1])
        for conn in psutil.net_connections(kind="inet"):
            if conn.laddr and conn.laddr.port == port and conn.pid:
                return psutil.Process(conn.pid).memory_info().rss / (1024 * 1024)
    except Exception:                                    # noqa: BLE001
        return None
    return None


def submit_batch(server: str, photos: int, per_request: int, result: Result) -> list[str]:
    """Submits in chunks, because one HTTP request with a hundred files is not
    how a booth is used and would measure multipart parsing rather than the
    queue."""
    run_ids = []
    for start in range(0, photos, per_request):
        chunk = min(per_request, photos - start)
        files = [("files", (f"soak_{start + i}.png", synthetic_photo(start + i), "image/png"))
                 for i in range(chunk)]
        resp = requests.post(f"http://{server}/api/batch", files=files,
                             data={"prompt": "a rooftop at golden hour",
                                   "consent_basis": "internal_test",
                                   "consent_by": "soak.py"},
                             timeout=300)
        if resp.status_code == 429 or resp.status_code == 503:
            # Not a failure of the run: this is admission control doing its
            # job, and how often it fires is one of the things being measured.
            result.refused += chunk
            print(f"  chunk of {chunk} refused: {resp.json().get('detail', resp.text)[:90]}")
            continue
        if resp.status_code >= 400:
            raise SystemExit(f"batch rejected ({resp.status_code}): {resp.text[:200]}")
        run_ids.append(resp.json()["run_id"])
        result.submitted += chunk
        print(f"  queued {chunk} photo(s) as {run_ids[-1]}")
    return run_ids


def follow(server: str, run_ids: list[str], result: Result, poll: float = 2.0) -> None:
    pending = set(run_ids)
    while pending:
        time.sleep(poll)
        sample = server_rss_mb(server)
        if sample is not None:
            result.rss_samples.append(sample)
        ready = requests.get(f"http://{server}/readyz", timeout=10)
        if ready.status_code != 200:
            result.readiness.extend(ready.json().get("reasons", []))

        # Aggregated across runs and printed only when it changes. One line per
        # run per poll is 200 lines of noise for a 40-photo soak, and it buries
        # the table this exists to produce.
        totals = {}
        for run_id in list(pending):
            body = requests.get(f"http://{server}/api/batch/{run_id}", timeout=10).json()
            if body.get("finished"):
                pending.discard(run_id)
                for item in body["items"]:
                    if item["status"] == "done":
                        result.done += 1
                    elif item["status"] == "failed":
                        result.failed += 1
            for status, count in body.get("counts", {}).items():
                totals[status] = totals.get(status, 0) + count

        line = " ".join(f"{k}={v}" for k, v in sorted(totals.items()) if v)
        rss = f"  rss={result.rss_samples[-1]:.0f}MB" if result.rss_samples else ""
        if line != result.last_line:
            print(f"  [{time.time() - result.started_at:6.0f}s] {line}{rss}", flush=True)
            result.last_line = line


def report(result: Result, photos: int, args) -> str:
    rows = [
        ("photos requested", photos),
        ("accepted", result.submitted),
        ("refused by admission control", result.refused),
        ("finished", result.done),
        ("failed", result.failed),
        ("wall clock", f"{result.elapsed:.1f}s"),
        ("throughput", f"{result.done / result.elapsed:.2f} photos/s"
                       if result.elapsed else "n/a"),
        ("seconds per photo", f"{result.elapsed / result.done:.2f}s"
                              if result.done else "n/a"),
    ]
    if result.rss_samples:
        rows += [
            ("server RSS at start", f"{result.rss_samples[0]:.0f} MB"),
            ("server RSS at end", f"{result.rss_samples[-1]:.0f} MB"),
            ("server RSS peak", f"{max(result.rss_samples):.0f} MB"),
            ("RSS growth", f"{result.rss_samples[-1] - result.rss_samples[0]:+.0f} MB"),
        ]
    else:
        rows.append(("server RSS", "not sampled (pip install psutil)"))
    if result.readiness:
        seen = {}
        for reason in result.readiness:
            seen[reason] = seen.get(reason, 0) + 1
        rows.append(("readiness said no", "; ".join(f"{k} (x{v})" for k, v in seen.items())))
    else:
        rows.append(("readiness", "green throughout"))

    width = max(len(str(k)) for k, _ in rows)
    lines = ["", "| measure | value |", "| --- | --- |"]
    lines += [f"| {str(k).ljust(width)} | {v} |" for k, v in rows]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server", default="127.0.0.1:8000")
    parser.add_argument("--photos", type=int, default=50)
    parser.add_argument("--per-request", type=int, default=10,
                        help="photos per /api/batch call (a booth submits in handfuls)")
    parser.add_argument("--json", action="store_true", help="machine-readable summary")
    args = parser.parse_args()

    # Checked first: a soak run that silently measures a server that was never
    # ready is a table of numbers about a warm-up.
    try:
        ready = requests.get(f"http://{args.server}/readyz", timeout=10)
    except requests.RequestException as exc:
        raise SystemExit(f"no server at {args.server}: {exc}")
    if ready.status_code != 200:
        raise SystemExit(f"server is not ready: {ready.json().get('reasons')}")
    print(f"server {ready.json().get('version')} ({ready.json().get('build')}) is ready")

    result = Result(started_at=time.time())
    sample = server_rss_mb(args.server)
    if sample is not None:
        result.rss_samples.append(sample)

    run_ids = submit_batch(args.server, args.photos, args.per_request, result)
    follow(args.server, run_ids, result)
    result.finished_at = time.time()

    if args.json:
        print(json.dumps({
            "photos": args.photos, "accepted": result.submitted,
            "refused": result.refused, "done": result.done, "failed": result.failed,
            "elapsed_s": round(result.elapsed, 2),
            "rss_mb": result.rss_samples,
        }, indent=2))
    else:
        print(report(result, args.photos, args))
    return 0 if result.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
