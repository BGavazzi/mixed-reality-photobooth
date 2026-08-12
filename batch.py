"""
Batch mode: many photos, one approved look.

The interactive app is one photo at a time, which is the wrong shape for the
thing it is actually for. A shoot produces dozens of frames and a client
expects them to look like one campaign -- which is only possible now that a
brand kit can pin the seed (see brand_kit.locked_seed). Batch mode is where
that stops being a property and becomes the product: point it at a directory,
pick a kit and an approved look, get back a set that belongs together.

Design notes worth reading before changing this:

**Runs live on disk, not in memory.** Each item's analysis output (cutout,
mask, depth) is written to the run directory and reloaded when its generation
finishes. Holding fifty subjects' worth of decoded PIL images while waiting on
a serial GPU is how a long batch turns into an OOM, and the intermediate files
are worth having anyway when a client asks why frame 31 looks wrong.

**The whole CPU pipeline happens inside the queued job.** `submit` does
analyze -> upload -> queue-to-ComfyUI, so the queue's worker pool overlaps one
photo's ~20s rotoscope with another photo's GPU time instead of paying for
every analysis up front. That is also why batch mode goes through
job_queue.py rather than calling the backend directly: fifty photos submitted
at once is exactly the load the admission control exists for.

**Results are collected by run, not by websocket session.** A batch outlives
the page that started it, and can be started with no page at all (see the CLI
at the bottom). So a job's result sink is either a browser session or a batch
run, which is why web_server's JOBS entries carry one or the other.
"""

from __future__ import annotations

import io
import json
import shutil
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

BATCH_ROOT = Path(__file__).parent / "batch_runs"

# Statuses an item moves through. Deliberately explicit rather than a bool:
# "queued but not yet analyzed" and "analyzed, waiting on the GPU" look the
# same to a caller otherwise, and they have very different expected durations.
PENDING = "pending"
ANALYZING = "analyzing"
GENERATING = "generating"
DONE = "done"
FAILED = "failed"

TERMINAL = (DONE, FAILED)


@dataclass
class BatchItem:
    index: int
    filename: str
    status: str = PENDING
    prompt_id: str | None = None
    error: str | None = None
    provenance: dict | None = None
    started_at: float = field(default_factory=time.monotonic)

    @property
    def stem(self) -> str:
        """A filesystem-safe, order-preserving name for this item's artifacts.

        The index prefix is not decoration: it keeps the output sorted the way
        the operator supplied the photos, and it disambiguates two uploads that
        happen to share a filename -- which is normal when files come from two
        cameras.
        """
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in Path(self.filename).stem)
        return f"{self.index:03d}_{safe[:40]}"

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "filename": self.filename,
            "status": self.status,
            "error": self.error,
            "seed": (self.provenance or {}).get("seed"),
        }


@dataclass
class BatchRun:
    """One batch. Mutated from the event loop and from queue worker threads,
    so every mutation goes through the lock."""
    run_id: str
    brand_id: str | None
    look_id: str | None
    look_label: str | None
    items: list[BatchItem]
    created_at: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def directory(self) -> Path:
        return BATCH_ROOT / self.run_id

    def path_for(self, kind: str, item: BatchItem, suffix: str = ".png") -> Path:
        directory = self.directory / kind
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{item.stem}{suffix}"

    def item_by_prompt(self, prompt_id: str) -> BatchItem | None:
        return next((i for i in self.items if i.prompt_id == prompt_id), None)

    def set_status(self, item: BatchItem, status: str, error: str | None = None):
        with self.lock:
            item.status = status
            if error:
                item.error = error

    @property
    def finished(self) -> bool:
        return all(i.status in TERMINAL for i in self.items)

    def counts(self) -> dict:
        counts = {status: 0 for status in (PENDING, ANALYZING, GENERATING, DONE, FAILED)}
        for item in self.items:
            counts[item.status] = counts.get(item.status, 0) + 1
        return counts

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "brand_id": self.brand_id,
            "look_id": self.look_id,
            "look_label": self.look_label,
            "created_at": self.created_at,
            "total": len(self.items),
            "counts": self.counts(),
            "finished": self.finished,
            "items": [i.to_dict() for i in self.items],
        }

    def write_manifest(self):
        """A machine-readable record of the run, written alongside the images.

        This is the artifact a brand-safety reviewer or a client actually gets
        handed with the frames -- every seed, every prompt, and which kit
        revision produced them. Rewritten on each completion rather than only
        at the end so an interrupted run still leaves a usable record.
        """
        manifest = {
            "run_id": self.run_id,
            "brand_id": self.brand_id,
            "look_id": self.look_id,
            "look_label": self.look_label,
            "created_at": self.created_at,
            "items": [
                {**item.to_dict(), "provenance": item.provenance}
                for item in self.items
            ],
        }
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")


# Process-local registry. A real deployment would keep this in Redis or a
# table -- it is the second thing after the queue that has to move off this box,
# and it is deliberately small and serialisable so that move stays cheap.
RUNS: dict[str, BatchRun] = {}


def create_run(filenames: list[str], brand_id: str | None,
               look_id: str | None, look_label: str | None) -> BatchRun:
    run = BatchRun(
        run_id=uuid.uuid4().hex[:12],
        brand_id=brand_id,
        look_id=look_id,
        look_label=look_label,
        items=[BatchItem(index=i + 1, filename=name) for i, name in enumerate(filenames)],
    )
    RUNS[run.run_id] = run
    run.directory.mkdir(parents=True, exist_ok=True)
    return run


def composite_subject_over(background: Image.Image, cutout_path: Path) -> Image.Image:
    """Puts the untouched subject back on top of its generated environment.

    This is the same compositing the browser canvas does, done server-side
    because a batch has no canvas. It stays pixel-exact on the subject for the
    same reason the interactive path does: those pixels are a real photograph
    of a real person and regenerating them is the one thing this tool must
    never do.
    """
    cutout = Image.open(cutout_path).convert("RGBA")
    canvas = background.convert("RGBA")
    if cutout.size != canvas.size:
        # The generated background comes back at the subject photo's own
        # resolution (the workflow has no resize node), so a mismatch means
        # something upstream changed -- resize rather than fail a whole batch
        # item, but it is worth being loud about in the log.
        print(f"[batch] size mismatch: background {canvas.size} vs cutout {cutout.size}, resizing")
        cutout = cutout.resize(canvas.size, Image.LANCZOS)
    canvas.alpha_composite(cutout)
    return canvas.convert("RGB")


def zip_run(run: BatchRun) -> io.BytesIO:
    """Zips a run's finished frames plus its manifest, in memory.

    In-memory because a batch of fifty 1024px PNGs is tens of megabytes, not
    gigabytes, and a temp file would need cleaning up on a path where the
    client may well have disconnected. If batches ever get big enough for this
    to matter, the fix is object storage, not a smarter temp file.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        output_dir = run.directory / "output"
        if output_dir.is_dir():
            for path in sorted(output_dir.iterdir()):
                archive.write(path, arcname=f"{run.run_id}/{path.name}")
        manifest = run.directory / "manifest.json"
        if manifest.exists():
            archive.write(manifest, arcname=f"{run.run_id}/manifest.json")
    buffer.seek(0)
    return buffer


def delete_run(run_id: str) -> bool:
    """Removes a run and its files. Batch output is intermediate by nature --
    the operator has the zip -- and a booth left running for a week should not
    slowly fill a disk with other people's photographs."""
    run = RUNS.pop(run_id, None)
    if run is None:
        return False
    shutil.rmtree(run.directory, ignore_errors=True)
    return True
