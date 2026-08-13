# Runbook

For whoever is standing next to the booth when it stops working — possibly you,
six months from now, at an event, with a queue of people waiting.

Every entry is **symptom → check → do**. The checks come first because most of
these have two causes with the same face, and the wrong fix for the other one
usually makes it worse.

## The two commands worth knowing

```
curl -s localhost:8000/readyz | python -m json.tool     # is it my fault or ComfyUI's?
python doctor.py                                        # is this box set up at all?
```

`/readyz` answers with a list of `reasons`. If it is green and something is
still wrong, the problem is downstream of this app.

---

## A guest's photo sits on "queued…" and never finishes

**Check** `/readyz` → `jobs_in_flight` and `comfy_silent_for`.

- `comfy_silent_for` climbing past 60s with jobs in flight means ComfyUI took
  the work and stopped talking.
- The watchdog will act on its own after `JOB_STALL_SECONDS` (default 300s): it
  asks ComfyUI's `/history` first, and if the render actually finished it
  **recovers** the picture and delivers it. You will see
  `[watchdog] recovered a finished render whose event was missed`.

**Do**

1. If you see recoveries, something is eating websocket events. The usual cause
   is **two processes sharing a `COMFY_CLIENT_ID`** — ComfyUI keeps one socket
   per client id and silently drops the older one. Check for a second instance
   (a Docker container alongside a native run is the classic). Stop one.
2. If ComfyUI is genuinely wedged, restart ComfyUI, not this app. In-flight
   jobs will fail with a message to the guest, the breaker will open and close
   itself, and the `DEPENDENCY_DOWN` alert clears automatically on recovery.
3. Restarting *this* app is the last resort — it loses the queue and the
   attention list. Drain first (below).

## Guests see "the render service is not responding"

That is the circuit breaker, open. It means ComfyUI failed
`COMFY_BREAKER_THRESHOLD` times in a row (default 5).

**Check** `/api/queue` → `breaker.state` and `breaker.opened_for`.

**Do** Look at ComfyUI itself: is the process alive, is the GPU out of memory,
did a model get moved? The breaker probes once every `COMFY_BREAKER_RECOVERY`
seconds (default 30) and closes itself the moment a call succeeds. **Do not
restart this app to clear it** — it clears on its own and a restart costs you
the queue.

## Everything is slow but nothing is failing

**Check** `/api/queue` → `waiting` and `running`.

The GPU is serial. A depth of 20 is a ~12-minute wait and the app is behaving
correctly. Workers do not make renders faster — they only overlap the uploads.

**Do** Tell people the real number, and if it is a shoot rather than a queue of
guests, use batch mode instead: `python batch_cli.py photos/ --brand aurora
--look coastline -o shoot.zip`.

## A batch run has items stuck in `generating`

**Check** `/api/batch/{run_id}` → per-item `status`, and the attention panel.

**Do** Items that failed are already recorded in the manifest with a reason and
the run continues; download what finished with the zip button — it works before
the run completes on purpose. The watchdog gives up on individual items after
`JOB_MAX_SECONDS` (default 1800) so a run cannot hang forever.

## The server refuses to start

**Check** the first lines of output. A `ConfigError` names the variable, its
value and what was expected:

```
config.ConfigError: BATCH_RETAIN_DAYS=-1 is below the minimum of 0
```

**Do** Fix the variable. Note that `0` means *keep indefinitely* and negative
values are refused specifically so that a typo cannot silently switch off the
deletion of people's photographs.

## Disk is filling up

**Check** `batch_runs/` — one directory per run.

**Do** Nothing, normally: runs expire after `BATCH_RETAIN_DAYS` (default 7) and
the sweep runs at startup and hourly, over the *directory* rather than only the
in-memory registry, so runs orphaned by a crash are collected too. To force it:
restart the server, or `DELETE /api/batch/{run_id}` for one run. If the
`retention sweep is failing` alert is up, that is why — read its detail.

## Someone asks to have their photographs deleted

**Check** the run's `manifest.json` → `consent` (who recorded it and how) and
`retention.expires_at`.

**Do** `DELETE /api/batch/{run_id}`. Be honest about the limits: consent is
recorded per *run*, not per person, so matching one guest to their frames is
manual, and this app cannot reach the operator's screenshots, the camera roll
on the capture device, or anything already exported to Resolume.

## Restarting cleanly

```
# Ctrl-C, or docker stop -t 60
```

Shutdown stops accepting new work, then drains what is already queued into
ComfyUI (deadline `SHUTDOWN_DRAIN_SECONDS`, default 30). Anything that does not
make it is **failed with a message to the browser** rather than dropped
silently. Measured on a 12-photo batch: 2 reached ComfyUI during the drain, the
remaining 8 were failed at the deadline, process gone at 30.8s.

What you will actually see from outside is `/readyz` going *unreachable*, not
503. Uvicorn closes the listening socket about two seconds after the signal,
before application shutdown runs — so by the time `READY` is cleared there is
nothing left to answer a probe. The in-process ordering still matters (a
request already in flight on a keep-alive connection is refused rather than
admitted), but do not expect to watch it turn red.

`docker stop` sends SIGTERM and waits 10s by default — less than the drain
deadline. Use `-t 60`, or lower `SHUTDOWN_DRAIN_SECONDS`.

## Reading the logs

One correlation id — the `prompt_id` — is on every line about a given photo,
from admission through the retry ladder to the compositor:

```
grep 'job=a1b2c3d4' server.log
```

`BOOTH_LOG_FORMAT=json` switches to one JSON object per line, with the full
id rather than the shortened one, for when something is shipping these
somewhere.

## Proving a change did not make things worse

```
python chaos_comfy.py --port 8189 --render-seconds 0.3 --failure-rate 0
COMFY_ADDRESS=127.0.0.1:8189 python web_server.py --port 8011
python soak.py --server 127.0.0.1:8011 --photos 100
```

No GPU required. Raise `--failure-rate` to soak against an unreliable
dependency; the numbers from a clean run are in the README.
