"""Tests for ``app.consumer`` — Redis Streams consumer mode.

Uses ``fakeredis.aioredis`` for an in-memory Redis with reasonably faithful
Streams + consumer-group semantics. Tests cover:

* Frame decode helper (happy path + malformed JPEG).
* ``_ensure_group`` idempotency (BUSYGROUP swallowed on second call).
* Per-session loop: success → event published + ack; failure → not acked
  (entry stays in the consumer group's Pending Entries List).
* Supervisor reconciliation: sessions appearing in ``sessions:active`` get a
  task, sessions removed get torn down.

We deliberately use a real :class:`fakeredis.aioredis.FakeRedis` instead of
mocking ``r`` because the contract under test *is* the Redis protocol —
mocking would re-encode the contract in test code and make the test trivially
pass without exercising the actual XADD/XACK/XREADGROUP semantics.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from unittest.mock import MagicMock

import cv2
import fakeredis.aioredis
import numpy as np
import pytest
import pytest_asyncio

from app.consumer import (
    CONSUMER_GROUP,
    Consumer,
    _decode_frame_message,
    _ensure_group,
    consume_session,
)


@pytest_asyncio.fixture
async def r() -> AsyncIterator[fakeredis.aioredis.FakeRedis]:
    """Fresh, byte-clean fakeredis instance per test.

    ``decode_responses=False`` matches the production worker so the JPEG
    payload stays binary-clean — decoding to str would mangle the bytes.
    """
    client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    try:
        yield client
    finally:
        await client.aclose()


def _make_jpeg(width: int = 64, height: int = 64) -> bytes:
    """Synthesize a valid JPEG of the requested size (all-black image)."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img, (cv2.IMWRITE_JPEG_QUALITY, 75))
    assert ok
    return bytes(buf)


def _frame_fields(
    jpeg: bytes, *, sid: str, fid: int, fts: int, w: int, h: int
) -> dict[bytes, bytes]:
    """Build the exact field map the gateway XADDs (DESIGN.md §5.1).

    Pinning this in the test keeps a regression in field naming / encoding
    visible — the contract is bytes-on-the-wire, not Python kwarg shape.
    """
    return {
        b"session_id": sid.encode(),
        b"frame_id": str(fid).encode(),
        b"frame_ts": str(fts).encode(),
        b"width": str(w).encode(),
        b"height": str(h).encode(),
        b"jpeg": jpeg,
    }


def test_decode_frame_message_extracts_bgr_image_and_meta():
    """Happy path: round-trip a real JPEG through the decoder."""
    jpeg = _make_jpeg(80, 60)
    fields = _frame_fields(jpeg, sid="s", fid=7, fts=12345, w=80, h=60)
    bgr, fid, fts = _decode_frame_message(fields)
    assert bgr.shape == (60, 80, 3)
    assert fid == 7
    assert fts == 12345


def test_decode_frame_message_raises_on_bad_jpeg():
    """Malformed JPEG must raise (not silently return ``None``).

    The caller relies on the exception to keep the entry in the PEL rather
    than silently dropping the frame.
    """
    fields = _frame_fields(b"not a jpeg", sid="s", fid=1, fts=1, w=1, h=1)
    with pytest.raises(ValueError):
        _decode_frame_message(fields)


async def test_ensure_group_creates_group_idempotently(r):
    """Second ``_ensure_group`` call must not raise (BUSYGROUP swallowed).

    Reconcile calls this every cycle for every active session; a non-idempotent
    implementation would crash the supervisor as soon as a session entered the
    second tick.
    """
    await _ensure_group(r, "abc")
    await _ensure_group(r, "abc")
    groups = await r.xinfo_groups("frames:abc")
    # ``xinfo_groups`` returns str-keyed dicts even under
    # ``decode_responses=False`` (the *values* are bytes; the keys are
    # documented Redis field names). Match on the bytes-encoded group name.
    assert any(g["name"] == CONSUMER_GROUP.encode() for g in groups)


async def test_consume_session_processes_frame_and_publishes_event(r):
    """End-to-end happy path: frame in → event out + ack.

    Drives one frame through a mocked pipeline (a real pipeline would need
    OpenVINO model weights). Asserts both halves of the success contract:
    the detection event is published, and the source frame is acked (so the
    consumer group's PEL is empty).
    """
    sid = "live"
    jpeg = _make_jpeg(64, 64)
    await r.xadd(f"frames:{sid}", _frame_fields(jpeg, sid=sid, fid=1, fts=1000, w=64, h=64))

    pipeline = MagicMock()
    pipeline.process_frame = MagicMock(
        return_value={"session_id": sid, "frame_id": 1, "tracks": []}
    )

    stop = asyncio.Event()

    async def stop_soon():
        # Long enough for at least one XREADGROUP round-trip to complete in
        # fakeredis; short enough that the test stays fast.
        await asyncio.sleep(0.3)
        stop.set()

    await asyncio.gather(
        consume_session(r, sid, pipeline, consumer_name="t1", stop_event=stop),
        stop_soon(),
    )

    # Detection event published on detections:{sid}
    entries = await r.xrange(f"detections:{sid}")
    assert len(entries) == 1
    _id, fields = entries[0]
    event = json.loads(fields[b"event"])
    assert event["frame_id"] == 1
    assert event["session_id"] == sid

    # Frame acked — PEL empty.
    pending = await r.xpending(f"frames:{sid}", CONSUMER_GROUP)
    assert pending["pending"] == 0

    # Pipeline saw the exact frame_id / frame_ts / session_id from the wire.
    call = pipeline.process_frame.call_args
    assert call.args[1] == 1  # frame_no
    assert call.kwargs["session_id"] == sid
    assert call.kwargs["frame_ts"] == 1000


async def test_consume_session_does_not_ack_on_pipeline_failure(r):
    """Pipeline exception → no event published, entry stays in PEL.

    This is the operator-visibility contract: a recurring failure leaves a
    growing PEL that monitoring can alert on. Silent drops would erase that
    signal.
    """
    sid = "fail"
    jpeg = _make_jpeg()
    await r.xadd(f"frames:{sid}", _frame_fields(jpeg, sid=sid, fid=1, fts=1, w=64, h=64))

    pipeline = MagicMock()
    pipeline.process_frame = MagicMock(side_effect=RuntimeError("kaboom"))

    stop = asyncio.Event()

    async def stop_soon():
        await asyncio.sleep(0.3)
        stop.set()

    await asyncio.gather(
        consume_session(r, sid, pipeline, consumer_name="t1", stop_event=stop),
        stop_soon(),
    )

    # No detection event was published.
    assert await r.xlen(f"detections:{sid}") == 0
    # Frame still in the consumer group's Pending Entries List.
    pending = await r.xpending(f"frames:{sid}", CONSUMER_GROUP)
    assert pending["pending"] == 1


async def test_consumer_reconciles_sessions_from_active_set(r):
    """Supervisor adds tasks when ``sessions:active`` grows; removes them when it shrinks.

    Uses a short ``poll_s`` so the test doesn't have to wait the full 1s
    discovery interval. The accesses to private ``_tasks`` are a deliberate
    test-only peek at supervisor state — the alternative (asserting via Redis
    side effects) would require firing frames through the pipeline and is
    covered by the per-session test above.
    """
    pipeline = MagicMock()
    pipeline.process_frame = MagicMock(
        return_value={"session_id": "x", "frame_id": 0, "tracks": []}
    )

    consumer = Consumer(r, pipeline, consumer_name="t1", poll_s=0.05)

    await r.sadd("sessions:active", "s1")

    supervisor = asyncio.create_task(consumer.run())
    try:
        # Wait for at least one reconcile cycle to pick up s1.
        await asyncio.sleep(0.2)
        assert "s1" in consumer._tasks

        # Remove from the active set; reconcile should tear the task down.
        await r.srem("sessions:active", "s1")
        await asyncio.sleep(0.2)
        assert "s1" not in consumer._tasks
    finally:
        consumer.stop()
        await supervisor
