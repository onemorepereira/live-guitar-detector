"""Redis Streams consumer mode for the inference worker (DESIGN.md §5.3).

This is the second consumer of the same :class:`~app.pipeline.Pipeline` — the
first being the local webcam demo. Here, frames arrive over Redis instead of
OpenCV, and detection events go back over Redis instead of being painted into
a window.

Architecture
------------
A single :class:`Consumer` instance acts as a **supervisor**: it polls
``SMEMBERS sessions:active`` every 1s (DESIGN.md §5.3 "Worker discovers
sessions") and reconciles the set of running per-session consumer tasks
against it. New sessions get a new :func:`consume_session` task; departed
sessions get their task cancelled.

Each per-session task runs an ``XREADGROUP`` loop on ``frames:{sid}`` in the
``inference`` consumer group:

1. Read a small batch with a short ``BLOCK`` (so cancellation isn't laggy).
2. JPEG-decode each frame to numpy BGR.
3. Run :meth:`Pipeline.process_frame`.
4. ``XADD`` the detection event to ``detections:{sid}``.
5. ``XACK`` the source message.

If the pipeline raises, the message is **deliberately not acked** — it sits in
the consumer group's Pending Entries List and will redeliver to the next
claimant after the PEL timeout. That keeps transient model errors recoverable
while making persistent failures visible to operators (the PEL won't drain).

Why we don't share with the gateway's ``redis_io``
--------------------------------------------------
The wire format (DESIGN.md §5.1, §5.3) is the contract between gateway and
worker — not a Python helper signature. Extracting a shared library for two
small services would be overkill; instead we re-implement the few lines of
XADD-with-MAXLEN here and let the gateway's ``redis_io.publish_detection_event``
stay symmetric. Both functions must format ``{b"event": <json>}`` identically;
the cross-service test that pins that shape is the gateway-side
``test_publish_detection_event_writes_to_detections_stream``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json

import cv2
import numpy as np
import redis.asyncio as redis_async
from loguru import logger

from app.pipeline import Pipeline

# Group name worker(s) join on every ``frames:*`` stream. Pinned to a constant
# so tests can assert the wire-level group name without re-importing it from
# a "magic string" sprinkled through the module.
CONSUMER_GROUP = "inference"

# DESIGN.md §5.3 — detections stream is bounded to ~100 entries; older events
# are dropped under backpressure. Approximate trim is fine and is faster than
# the strict ``=`` form.
DETECTIONS_MAXLEN = 100

# DESIGN.md §5.3 "Worker discovers sessions: on a 1-second tick".
DISCOVERY_INTERVAL_S = 1.0

# Short BLOCK so a cancelled supervisor doesn't have to wait the full XREAD
# timeout before the per-session task observes ``stop_event``. 200ms is well
# under the typical session teardown budget yet long enough to amortise the
# syscall overhead at idle.
XREAD_BLOCK_MS = 200

# Small batch — pipeline.process_frame is heavyweight (YOLO + CLIP), so we
# prefer responsiveness to a brief queue burst over draining 100 frames in
# one go and starving the supervisor's reconcile cycle.
XREAD_COUNT = 10


def _frames_key(session_id: str) -> str:
    """Stream key the gateway publishes frames to. Mirrored from gateway/redis_io."""
    return f"frames:{session_id}"


def _detections_key(session_id: str) -> str:
    """Stream key the worker publishes detection events to."""
    return f"detections:{session_id}"


async def _ensure_group(r: redis_async.Redis, session_id: str) -> None:
    """Idempotently create the ``inference`` consumer group on ``frames:{sid}``.

    ``mkstream=True`` lets us create the group even before the gateway has
    XADD'd a single frame (the worker can win the startup race). ``BUSYGROUP``
    is the documented "group already exists" error — swallowed so the call is
    safe to retry on every reconcile.
    """
    key = _frames_key(session_id)
    try:
        await r.xgroup_create(key, CONSUMER_GROUP, id="0-0", mkstream=True)
    except redis_async.ResponseError as exc:
        if "BUSYGROUP" not in str(exc):
            raise


async def _publish_detection_event(r: redis_async.Redis, *, session_id: str, event: dict) -> bytes:
    """XADD a detection event to ``detections:{sid}``.

    Single ``event`` field carrying the JSON-encoded blob — same shape the
    gateway's ``consume_detections`` decodes. Approximate MAXLEN keeps the
    stream bounded while letting Redis batch trim work.
    """
    payload = json.dumps(event).encode()
    return await r.xadd(
        _detections_key(session_id),
        {b"event": payload},
        maxlen=DETECTIONS_MAXLEN,
        approximate=True,
    )


def _decode_frame_message(fields: dict[bytes, bytes]) -> tuple[np.ndarray, int, int]:
    """Extract ``(bgr_image, frame_id, frame_ts)`` from a raw stream entry.

    Wire fields are defined by the gateway (DESIGN.md §5.1): ``jpeg`` is the
    raw byte payload; ``frame_id`` and ``frame_ts`` are stringified integers.
    ``cv2.imdecode`` returns ``None`` on a malformed payload — we promote that
    to ``ValueError`` so the caller's ``except Exception`` branch can leave
    the entry in the PEL for inspection rather than silently dropping it.
    """
    jpeg = fields[b"jpeg"]
    arr = np.frombuffer(jpeg, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("JPEG decode failed")
    frame_id = int(fields[b"frame_id"])
    frame_ts = int(fields[b"frame_ts"])
    return bgr, frame_id, frame_ts


async def consume_session(
    r: redis_async.Redis,
    session_id: str,
    pipeline: Pipeline,
    *,
    consumer_name: str,
    stop_event: asyncio.Event,
) -> None:
    """Per-session ``XREADGROUP`` loop. Exits cleanly when ``stop_event`` is set.

    Lifetime is owned by :class:`Consumer` — the supervisor sets the event
    when the session leaves ``sessions:active`` and additionally ``.cancel()``s
    the task to unblock any in-flight ``XREADGROUP``.

    Error policy:

    * Redis I/O errors (``ConnectionError`` etc.) → log + brief backoff + retry.
      The supervisor doesn't need to know; transient redis blips shouldn't
      flap consumer registration.
    * Pipeline / decode errors → log and **skip the ack**. The entry stays in
      the consumer group's PEL and will redeliver to another consumer after
      the PEL timeout. This keeps a single bad frame from killing the loop
      while preserving operator visibility (a stuck PEL means recurring
      failure on the same entry).
    """
    key = _frames_key(session_id)
    await _ensure_group(r, session_id)

    while not stop_event.is_set():
        try:
            resp = await r.xreadgroup(
                CONSUMER_GROUP,
                consumer_name,
                {key: ">"},
                count=XREAD_COUNT,
                block=XREAD_BLOCK_MS,
            )
        except asyncio.CancelledError:
            # Propagate cancellation: the supervisor relies on awaiting the
            # cancelled task to confirm shutdown.
            raise
        except Exception as exc:
            logger.warning("session={} xreadgroup failed: {}", session_id, exc)
            await asyncio.sleep(0.1)
            continue

        if not resp:
            # BLOCK timeout — loop again so we can observe stop_event.
            # The explicit yield is defensive: fakeredis returns immediately
            # on an empty stream (instead of honouring BLOCK), and without a
            # yield the tight loop would starve other tasks on the same loop
            # — including the test harness's "set stop_event" task. Real
            # Redis honours BLOCK so this is a no-op in production.
            await asyncio.sleep(0)
            continue

        for _stream, entries in resp:
            for entry_id, fields in entries:
                try:
                    bgr, frame_id, frame_ts = _decode_frame_message(fields)
                    event = pipeline.process_frame(
                        bgr, frame_id, session_id=session_id, frame_ts=frame_ts
                    )
                    await _publish_detection_event(r, session_id=session_id, event=event)
                    await r.xack(key, CONSUMER_GROUP, entry_id)
                except Exception as exc:
                    # Intentionally do NOT ack. The PEL is the operator's
                    # signal that something is wrong — silent drops would hide
                    # it.
                    logger.warning(
                        "session={} entry={} pipeline failed: {} (not acked)",
                        session_id,
                        entry_id,
                        exc,
                    )


class Consumer:
    """Multi-session ``XREADGROUP`` supervisor.

    Polls ``SMEMBERS sessions:active`` every ``poll_s`` seconds and runs one
    :func:`consume_session` task per active session. Sessions added between
    ticks get picked up on the next reconcile; sessions removed get their
    task cancelled (and awaited, so the worker shuts down deterministically).

    The :class:`Pipeline` is shared across all per-session tasks — it is
    stateful per ByteTrack / vote registry, but those structures are keyed by
    track id which the detector assigns globally across sessions. For Phase 2
    we expect one worker process per node and one or two sessions at a time;
    if that changes we'd revisit pipeline-per-session.
    """

    def __init__(
        self,
        r: redis_async.Redis,
        pipeline: Pipeline,
        *,
        consumer_name: str = "worker-1",
        poll_s: float = DISCOVERY_INTERVAL_S,
    ) -> None:
        self._r = r
        self._pipeline = pipeline
        self._consumer_name = consumer_name
        self._poll_s = poll_s
        self._tasks: dict[str, asyncio.Task] = {}
        self._stops: dict[str, asyncio.Event] = {}
        self._stop_supervisor = asyncio.Event()

    async def run(self) -> None:
        """Supervisor loop: discover → reconcile → sleep, until :meth:`stop`.

        Wrapping the loop body in its own try/except keeps a single failed
        reconcile (e.g. Redis hiccup) from killing the supervisor and leaving
        zombie per-session tasks running.
        """
        try:
            while not self._stop_supervisor.is_set():
                try:
                    await self._reconcile()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning("reconcile failed: {}", exc)
                await asyncio.sleep(self._poll_s)
        finally:
            await self._shutdown()

    def stop(self) -> None:
        """Request a graceful supervisor shutdown.

        Non-async on purpose: signal handlers and other sync callers should
        be able to flip the flag without scheduling a coroutine.
        """
        self._stop_supervisor.set()

    async def _reconcile(self) -> None:
        """Compare active set vs. running tasks; start/stop to match.

        ``SMEMBERS`` returns bytes when the client is configured with
        ``decode_responses=False`` (which we are, to keep JPEGs binary-clean).
        Normalise to ``str`` here so downstream code can use plain string
        session ids.
        """
        raw = await self._r.smembers("sessions:active")
        active = {s.decode() if isinstance(s, bytes) else s for s in raw}
        for sid in active - self._tasks.keys():
            self._start(sid)
        # list(...) snapshot — _stop_session mutates self._tasks.
        for sid in list(self._tasks.keys() - active):
            await self._stop_session(sid)

    def _start(self, session_id: str) -> None:
        """Spawn a per-session consumer task and track it for later teardown."""
        stop = asyncio.Event()
        self._stops[session_id] = stop
        task = asyncio.create_task(
            consume_session(
                self._r,
                session_id,
                self._pipeline,
                consumer_name=self._consumer_name,
                stop_event=stop,
            ),
            name=f"consume:{session_id}",
        )
        self._tasks[session_id] = task
        logger.info("session={} consumer started", session_id)

    async def _stop_session(self, session_id: str) -> None:
        """Stop a running per-session task; awaited so shutdown is deterministic.

        We set the stop event *and* cancel the task: the event handles the
        normal case (loop exits between iterations), the cancel covers the
        case where the task is currently inside ``XREADGROUP BLOCK``.
        """
        stop = self._stops.pop(session_id, None)
        task = self._tasks.pop(session_id, None)
        if stop is not None:
            stop.set()
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        logger.info("session={} consumer stopped", session_id)

    async def _shutdown(self) -> None:
        """Cancel every per-session task; called from :meth:`run`'s finally."""
        for sid in list(self._tasks.keys()):
            await self._stop_session(sid)
