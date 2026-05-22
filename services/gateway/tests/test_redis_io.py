"""Tests for ``app.redis_io`` — gateway-side wire format for Redis Streams.

These tests pin the byte-level shape of the ``frames:{session_id}`` and
``detections:{session_id}`` streams against ``fakeredis`` so a single
integration check would catch regressions in either direction (gateway
publishing frames, worker publishing detection events back).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import fakeredis.aioredis
import pytest_asyncio

from app.redis_io import (
    FRAMES_MAXLEN,
    consume_detections,
    publish_detection_event,
    publish_frame,
)


@pytest_asyncio.fixture
async def r() -> AsyncIterator[fakeredis.aioredis.FakeRedis]:
    """Fresh, byte-clean fakeredis instance per test."""
    client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    try:
        yield client
    finally:
        await client.aclose()


async def test_publish_frame_writes_to_frames_stream(r):
    """publish_frame XADDs every field with the documented byte shape."""
    entry_id = await publish_frame(
        r,
        session_id="abc",
        jpeg_bytes=b"\xff\xd8\xff\xe0fakejpeg",
        frame_id=1,
        frame_ts=12345,
        width=640,
        height=480,
    )
    assert entry_id  # non-empty entry id
    length = await r.xlen("frames:abc")
    assert length == 1
    entries = await r.xrange("frames:abc", count=1)
    _, fields = entries[0]
    assert fields[b"session_id"] == b"abc"
    assert fields[b"frame_id"] == b"1"
    assert fields[b"frame_ts"] == b"12345"
    assert fields[b"width"] == b"640"
    assert fields[b"height"] == b"480"
    assert fields[b"jpeg"] == b"\xff\xd8\xff\xe0fakejpeg"


async def test_publish_frame_trims_to_maxlen(r):
    """MAXLEN ~ N keeps the stream from growing unbounded under backpressure."""
    for i in range(FRAMES_MAXLEN + 20):
        await publish_frame(
            r,
            session_id="trim",
            jpeg_bytes=b"x",
            frame_id=i,
            frame_ts=i,
            width=1,
            height=1,
        )
    length = await r.xlen("frames:trim")
    # MAXLEN ~ 30 is approximate but should be close to 30, not 50+
    assert length <= FRAMES_MAXLEN + 5


async def test_publish_detection_event_writes_to_detections_stream(r):
    """Detection events round-trip as a single JSON-encoded ``event`` field."""
    event = {"session_id": "x", "frame_id": 1, "tracks": []}
    entry_id = await publish_detection_event(r, session_id="x", event=event)
    assert entry_id
    entries = await r.xrange("detections:x", count=1)
    _, fields = entries[0]
    assert json.loads(fields[b"event"]) == event


async def test_consume_detections_yields_events(r):
    """Publish 5 events on one task; consume them in order on another."""
    session_id = "rt"
    received: list[dict] = []

    async def consumer():
        async for ev in consume_detections(r, session_id, block_ms=50, last_id=b"0-0"):
            received.append(ev)
            if len(received) >= 5:
                break

    async def producer():
        # Brief sleep so the consumer is blocking on XREAD before we XADD.
        await asyncio.sleep(0.05)
        for i in range(5):
            await publish_detection_event(
                r, session_id=session_id, event={"frame_id": i, "tracks": []}
            )

    await asyncio.gather(consumer(), producer())

    assert len(received) == 5
    assert [e["frame_id"] for e in received] == [0, 1, 2, 3, 4]


async def test_consume_detections_default_skips_old_events(r):
    """With default last_id='$', events present BEFORE the consumer are skipped.

    ``block_ms`` is generous (250) because fakeredis polls the change-callback
    at coarse granularity — if the producer's XADD lands too close to the
    block deadline, the consumer can race past it and re-issue XREAD with a
    refreshed ``$`` cursor that's already past the new entry. Real Redis
    doesn't suffer from this because BLOCK is event-driven, not polled.
    """
    await publish_detection_event(r, session_id="skip", event={"old": True})
    await publish_detection_event(r, session_id="skip", event={"old": True})

    received: list[dict] = []

    async def consumer():
        async for ev in consume_detections(r, "skip", block_ms=250):
            received.append(ev)
            if len(received) >= 1:
                break

    async def producer():
        await asyncio.sleep(0.05)
        await publish_detection_event(r, session_id="skip", event={"new": True})

    await asyncio.gather(consumer(), producer())

    assert len(received) == 1
    assert received[0] == {"new": True}
