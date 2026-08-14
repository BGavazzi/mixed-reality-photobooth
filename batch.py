"""
Batch mode: many photos, one approved look.

The interactive app is one photo at a time, which is the wrong shape for the
thing it is actually for. A shoot produces dozens of frames and a client
expects them to look like one campaign -- which is only possible now that a
brand kit can pin the seed (see brand_kit.locked_seed). Batch mode is where
that stops being a property and becomes the product: point it at a directory,
pick a kit and an approved look, get back a set that belongs together.

Design notes worth reading before changing this:

**Runs live on disk, not in memory.** Each item's cutout is written to the run
directory and reloaded when its generation finishes. Holding fifty subjects'
worth of decoded PIL images while waiting on a serial GPU is how a long batch
turns into an OOM.

**What lands on disk is other people's faces, so as little as possible does,
for as short a time as possible.** Two rules follow from that, and both are
enforced here rather than left to an operator's diligence:

*Minimisation.* The original upload is deleted the moment analysis has produced
a cutout, because nothing downstream ever reads it again -- the compositor
needs the cutout, not the photograph. An earlier version of this file kept the
originals and justified it as being "worth having when a client asks why frame
31 looks wrong". That was a real convenience bought with someone else's
biometric data, which is the wrong trade to make silently; debugging now needs
`--keep-intermediates`, an explicit choice by whoever runs the booth.

*Retention.* Runs expire (see `sweep_expired`). A booth left running for a
season should not still be holding a photograph of everyone who walked past it
in the spring, and "the operator will remember to delete them" is not a
retention policy.

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
import os
import shutil
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

import config
import finish
import surfaces

BATCH_ROOT = Path(__file__).parent / "batch_runs"

# How long a finished run's files survive. Seven days is a working week: long
# enough that an operator who shot on Friday can still fetch the zip on
# Monday, short enough that a booth is not quietly accumulating a season's
# worth of strangers' faces. Overridable, because the right number is a
# business decision -- but the default is a policy, not "forever".
#
# Floored at zero rather than accepted as any number: a negative window means
# "retention off" to sweep_expired, so `BATCH_RETAIN_DAYS=-1` -- a plausible
# typo for "unset" -- used to silently disable the deletion of strangers'
# photographs while looking configured. Switching retention off is a decision,
# and it should be spelled `0`.
DEFAULT_RETAIN_DAYS = config.env_float(
    "BATCH_RETAIN_DAYS", 7.0, "days a finished batch run survives (0 = keep indefinitely)",
    minimum=0, maximum=3650)

# Keep the original uploads and the analysed copies. Off by default: the
# pipeline does not need them past analysis, and they are the most sensitive
# thing the app ever writes down.
KEEP_INTERMEDIATES = config.env_bool(
    "BATCH_KEEP_INTERMEDIATES", False,
    "keep original uploads and analysed copies for debugging")

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
    # Who said these photographs could be processed, and on what basis. Never
    # free text, and always present: a run where nobody declared anything is
    # recorded as `not_recorded` rather than left blank, so a later reader can
    # tell "nobody said" apart from "the field did not exist yet". Whether an
    # empty declaration blocks the run is consent.REQUIRED, off by default --
    # see consent.py for that argument.
    consent: dict = field(default_factory=dict)
    # Which declared output surfaces this run also renders (see surfaces.py).
    # Opt-in and empty by default: writing six reframes of every photo triples
    # the size of a fifty-photo zip for an operator who only wanted the frames.
    surface_ids: list = field(default_factory=list)
    retain_days: float = DEFAULT_RETAIN_DAYS
    created_at: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def directory(self) -> Path:
        return BATCH_ROOT / self.run_id

    @property
    def expires_at(self) -> float:
        return self.created_at + self.retain_days * 86400

    def path_for(self, kind: str, item: BatchItem, suffix: str = ".png") -> Path:
        directory = self.directory / kind
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{item.stem}{suffix}"

    def drop_original(self, item: BatchItem) -> None:
        """Deletes the uploaded photograph now that a cutout exists.

        Called from the worker the instant analysis succeeds, not at the end of
        the run: the window in which the app holds the full original is the
        thing being minimised, and a fifty-photo batch takes half an hour.
        """
        if KEEP_INTERMEDIATES:
            return
        for kind, suffix in (("input", ".orig"), ("analyzed", ".png")):
            path = self.directory / kind / f"{item.stem}{suffix}"
            path.unlink(missing_ok=True)

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
            "expires_at": self.expires_at,
            "consent": self.consent,
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
            # Which build made these frames. The kit revision was already here,
            # which answers "what were the rules?" but not "what was the code?"
            # -- and when a client asks in November why the September set looks
            # different, those are different questions with different fixes.
            "produced_by": config.version_info(),
            "brand_id": self.brand_id,
            "look_id": self.look_id,
            "look_label": self.look_label,
            "created_at": self.created_at,
            # The two facts a reviewer asks about first and that no amount of
            # seed provenance answers: what were these people told, and when
            # does this stop being kept.
            "consent": self.consent,
            "surfaces": [surfaces.get(s).to_dict() for s in self.surface_ids],
            "retention": {
                "retain_days": self.retain_days,
                "expires_at": self.expires_at,
                "originals_kept": KEEP_INTERMEDIATES,
            },
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
               look_id: str | None, look_label: str | None,
               consent: dict | None = None,
               surface_ids: list | None = None,
               retain_days: float | None = None) -> BatchRun:
    run = BatchRun(
        run_id=uuid.uuid4().hex[:12],
        brand_id=brand_id,
        look_id=look_id,
        look_label=look_label,
        items=[BatchItem(index=i + 1, filename=name) for i, name in enumerate(filenames)],
        consent=consent or {},
        surface_ids=list(surface_ids or []),
        retain_days=DEFAULT_RETAIN_DAYS if retain_days is None else retain_days,
    )
    RUNS[run.run_id] = run
    run.directory.mkdir(parents=True, exist_ok=True)
    return run


def sweep_expired(now: float | None = None, retain_days: float | None = None) -> list[str]:
    """Deletes runs whose retention window has passed. Returns what it removed.

    Sweeps the *directory*, not just the in-memory registry, because the case
    that matters most is the one where the process died: a crashed or
    restarted server would otherwise leave a folder of photographs that
    nothing in the app remembers and therefore nothing will ever clean up.
    That is also why it runs at startup and not only on a timer.

    Age comes from the manifest's created_at when there is one and the
    directory's mtime when there isn't. A run whose manifest never got written
    is exactly the interrupted case above, and refusing to age it because its
    paperwork is missing would keep the oldest data the longest.
    """
    now = time.time() if now is None else now
    retain_days = DEFAULT_RETAIN_DAYS if retain_days is None else retain_days
    if retain_days <= 0:
        return []          # 0 or negative means "keep indefinitely", set deliberately

    removed = []
    for run_id, run in list(RUNS.items()):
        if now >= run.expires_at:
            delete_run(run_id)
            removed.append(run_id)

    if not BATCH_ROOT.is_dir():
        return removed
    cutoff = now - retain_days * 86400
    for directory in BATCH_ROOT.iterdir():
        if not directory.is_dir() or directory.name in RUNS:
            continue
        created = directory.stat().st_mtime
        manifest = directory / "manifest.json"
        if manifest.exists():
            try:
                created = json.loads(manifest.read_text(encoding="utf-8")).get("created_at", created)
            except (json.JSONDecodeError, OSError):
                pass       # unreadable paperwork is not a reason to keep the photos
        if created < cutoff:
            shutil.rmtree(directory, ignore_errors=True)
            removed.append(directory.name)
    if removed:
        print(f"[batch] retention sweep removed {len(removed)} expired run(s): "
              f"{', '.join(removed[:5])}{'...' if len(removed) > 5 else ''}")
    return removed


def write_surfaces(run: "BatchRun", item: BatchItem, frame: Image.Image,
                   logo: Image.Image | None = None, logo_rules=None) -> list[str]:
    """Renders one finished frame for each of the run's declared surfaces.

    `frame` must be the *unbranded* composite. Each surface is a different crop
    of it, and the mark is placed after the crop, in that surface's own corner
    -- or not at all, on a live surface, where the guest's body would cover it.
    Branding first and cropping second put the postcard's logo off the edge of
    the postcard.

    The cutout is reloaded to locate the subject, because that is what lets a
    6x4 postcard crop keep a full-length portrait's face in the picture instead
    of delivering a print of their shoes.
    """
    if not run.surface_ids:
        return []
    cutout_path = run.path_for("cutout", item)
    mask = None
    if cutout_path.exists():
        mask = Image.open(cutout_path).convert("RGBA").getchannel("A")

    written = []
    for surface_id in run.surface_ids:
        surface = surfaces.get(surface_id)
        reframed = surfaces.reframe(frame, surface, mask=mask)
        if surface.logo_corner:
            reframed, _ = finish.brand(reframed, logo, logo_rules, surface.logo_corner,
                                       min_clear_space_pct=surface.safe_margin_pct)
        directory = run.directory / "surfaces" / surface.id
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{item.stem}.png"
        reframed.save(path, dpi=(surface.dpi, surface.dpi) if surface.dpi else None)
        written.append(surface.id)
    return written


def composite_subject_over(background: Image.Image, cutout_path: Path,
                            shadow_path: Path | None = None,
                            logo: Image.Image | None = None,
                            logo_rules=None,
                            grade_strength: float = finish.DEFAULT_GRADE_STRENGTH
                            ) -> tuple[Image.Image, dict]:
    """Puts the untouched subject back on top of its generated environment.

    This is the same compositing the browser canvas does, done server-side
    because a batch has no canvas -- and until recently it was *not* the same:
    it was a bare alpha_composite, so the deliverable had no contact shadow and
    no brand mark while the demo had both. See finish.py.

    It stays pixel-exact on the subject's geometry for the same reason the
    interactive path does: those pixels are a real photograph of a real person
    and regenerating them is the one thing this tool must never do. The colour
    grade is a fractional shift toward the plate's light, not a repaint.
    """
    cutout = Image.open(cutout_path).convert("RGBA")
    shadow = None
    if shadow_path is not None and shadow_path.exists():
        shadow = Image.open(shadow_path).convert("RGBA")
    return finish.finish_frame(background, cutout, shadow=shadow, logo=logo,
                               logo_rules=logo_rules, grade_strength=grade_strength)


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
        # Surface renders keep their own folders, so an operator handing a
        # client the zip can give the print shop one directory and the social
        # team another without re-sorting a hundred files by aspect ratio.
        surfaces_dir = run.directory / "surfaces"
        if surfaces_dir.is_dir():
            for surface_dir in sorted(surfaces_dir.iterdir()):
                for path in sorted(surface_dir.iterdir()):
                    archive.write(path, arcname=f"{run.run_id}/{surface_dir.name}/{path.name}")
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
