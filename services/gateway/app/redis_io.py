"""Redis Streams I/O for the gateway.

This module owns the wire format between gateway and worker:

- :func:`publish_frame` XADDs a JPEG frame to ``frames:{session_id}`` with
  ``MAXLEN ~ FRAMES_MAXLEN``, dropping the oldest entries under backpressure.
- :func:`consume_detections` is an async generator that XREADs from
  ``detections:{session_id}`` and yields parsed detection events.

The wire schema mirrors DESIGN.md §5.1: frames carry session_id / frame_id /
frame_ts / width / height / jpeg; detections carry the full DetectionEvent
JSON blob under a single ``event`` field (decoded on the consumer side).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import redis.asyncio as redis_async

# DESIGN.md §5.3 — backpressure caps on the streams. MAXLEN ~ is the
# approximate form (faster trim than the strict "=" form) — Redis is
# allowed to leave a few extra entries to avoid forcing a synchronous
# trim of every node in the radix tree on every XADD.
FRAMES_MAXLEN = 30
DETECTIONS_MAXLEN = 100


def _frames_key(session_id: str) -> str:
    return f"frames:{session_id}"


def _detections_key(session_id: str) -> str:
    return f"detections:{session_id}"


async def publish_frame(
    r: redis_async.Redis,
    *,
    session_id: str,
    jpeg_bytes: bytes,
    frame_id: int,
    frame_ts: int,
    width: int,
    height: int,
) -> bytes:
    """XADD a frame to ``frames:{session_id}``. Returns the new entry ID.

    All scalar fields are encoded as utf-8 byte strings so the client can
    run with ``decode_responses=False`` (required to keep the raw JPEG
    payload binary-clean through Redis).
    """
    fields = {
        b"session_id": session_id.encode(),
        b"frame_id": str(frame_id).encode(),
        b"frame_ts": str(frame_ts).encode(),
        b"width": str(width).encode(),
        b"height": str(height).encode(),
        b"jpeg": jpeg_bytes,
    }
    return await r.xadd(
        _frames_key(session_id),
        fields,
        maxlen=FRAMES_MAXLEN,
        approximate=True,
    )


async def publish_detection_event(
    r: redis_async.Redis,
    *,
    session_id: str,
    event: dict[str, Any],
) -> bytes:
    """XADD a detection event to ``detections:{session_id}``.

    Stored as a single ``event`` field with the JSON-encoded event blob.
    The worker writes; the gateway reads via :func:`consume_detections`.
    """
    payload = json.dumps(event).encode()
    return await r.xadd(
        _detections_key(session_id),
        {b"event": payload},
        maxlen=DETECTIONS_MAXLEN,
        approximate=True,
    )


async def consume_detections(
    r: redis_async.Redis,
    session_id: str,
    *,
    block_ms: int = 100,
    last_id: bytes = b"$",
) -> AsyncIterator[dict[str, Any]]:
    """Async iterator yielding detection events for one session.

    Starts at the ``"$"`` mark by default (only events arriving AFTER
    iterator construction). Pass a different ``last_id`` (e.g. ``b"0-0"``)
    to resume from a specific position or replay history.

    Exits when the caller breaks out of the ``async for``; there is no
    internal cancellation policy — the surrounding task owns lifecycle.
    """
    key = _detections_key(session_id)
    cursor = last_id
    while True:
        resp = await r.xread({key: cursor}, block=block_ms, count=10)
        if not resp:
            continue
        # resp shape: [(stream_name, [(entry_id, fields), ...])]
        for _stream, entries in resp:
            for entry_id, fields in entries:
                cursor = entry_id
                blob = fields.get(b"event")
                if blob is None:
                    continue
                try:
                    yield json.loads(blob)
                except json.JSONDecodeError:
                    # Malformed event — skip but don't crash the consumer.
                    continue
