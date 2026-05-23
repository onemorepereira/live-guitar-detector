"""Integration test harness: synthetic frame producer + assert subscriber.

DESIGN.md §9.3 calls for two roles, but a clean two-container design has a
chicken-and-egg coordination problem (both sides need to agree on a single
``session_id`` chosen at runtime). We collapse the roles into a single
process that runs producer and subscriber as concurrent asyncio tasks so
they trivially share the same generated UUID.

Behaviour:
- ``POST /api/session`` to register the session on the gateway.
- Open the detection WebSocket for that session.
- ``XADD`` fixture JPEGs onto ``frames:{sid}`` at the configured FPS.
- Assert that detection events flow back on the WS.

Exit codes:
- ``0`` — at least ``--min-events`` detection events were received in time.
- ``1`` — setup failure (gateway never became ready, no fixtures, etc.).
- ``2`` — wire path was healthy but not enough events arrived before the
  deadline.

The synthetic placeholder fixtures (Task 1.6) do NOT trigger YOLO detections,
which is OK: this harness proves the wire path works end-to-end (gateway
``POST /api/session`` -> WS accept -> worker ``XREADGROUP`` -> worker
``XADD detections`` -> gateway forwards JSON -> client receives it). The
worker emits detection events for every frame, even when ``tracks`` is empty,
so the harness still observes events.

If real fixture images are dropped in alongside ``REAL.txt`` the tighter
DESIGN.md §9.3 assertion (a stable label appears within 1 s) becomes
meaningful — that's a future tightening, gated on real fixtures landing.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import sys
import time
import uuid
from pathlib import Path

import cv2
import httpx
import numpy as np
import redis.asyncio as redis_async
import websockets


async def _wait_for_gateway(
    client: httpx.AsyncClient, gateway_http: str, attempts: int = 60
) -> bool:
    """Poll ``/readyz`` until the gateway reports ready or attempts exhaust."""
    for _ in range(attempts):
        try:
            resp = await client.get(f"{gateway_http}/readyz")
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        await asyncio.sleep(0.5)
    return False


async def producer(
    r: redis_async.Redis,
    session_id: str,
    frames: list[bytes],
    sizes: list[tuple[int, int]],
    duration_s: float,
    fps: float,
    stop_event: asyncio.Event,
) -> int:
    """Publish synthetic frames to ``frames:{session_id}`` at ``fps`` until done.

    Returns the number of frames sent. Exits early if ``stop_event`` fires
    (the subscriber sets this once it has collected enough detection events).
    """
    interval = 1.0 / max(0.1, fps)
    end_at = time.time() + duration_s
    frame_id = 0
    next_send_at = time.time()
    while time.time() < end_at and not stop_event.is_set():
        jpeg = frames[frame_id % len(frames)]
        w, h = sizes[frame_id % len(sizes)]
        now_ms = int(time.time() * 1000)
        await r.xadd(
            f"frames:{session_id}",
            {
                b"session_id": session_id.encode(),
                b"frame_id": str(frame_id).encode(),
                b"frame_ts": str(now_ms).encode(),
                b"width": str(w).encode(),
                b"height": str(h).encode(),
                b"jpeg": jpeg,
            },
            maxlen=30,
            approximate=True,
        )
        frame_id += 1
        next_send_at += interval
        sleep_for = max(0.0, next_send_at - time.time())
        await asyncio.sleep(sleep_for)
    print(f"producer: sent {frame_id} frames", flush=True)
    return frame_id


async def subscriber(
    ws_url_base: str,
    session_id: str,
    stop_event: asyncio.Event,
    received: list[dict],
    enough: int,
) -> None:
    """Read detection events off the WS into ``received`` until enough or stop.

    Treats ``{"type":"pong"}`` as a keepalive (not a detection event) so it
    doesn't inflate the count. Anything else with valid JSON is appended.
    """
    url = f"{ws_url_base}/ws?session_id={session_id}"
    async with websockets.connect(url) as ws:
        print(f"subscriber: connected to {url}", flush=True)
        while not stop_event.is_set():
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
            except TimeoutError:
                continue
            except websockets.ConnectionClosed:
                print("subscriber: ws closed by server", flush=True)
                return
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict) and event.get("type") == "pong":
                continue
            received.append(event)
            if len(received) >= enough:
                stop_event.set()
                return


def _load_fixtures(fixtures_dir: Path) -> tuple[list[bytes], list[tuple[int, int]]]:
    """Read every ``*.jpg`` under ``fixtures_dir`` and decode to learn its size.

    Raises ``SystemExit(1)`` if no fixtures or any image fails to decode —
    the producer can't ``XADD`` a frame without ``width``/``height``.
    """
    jpegs = sorted(fixtures_dir.glob("*.jpg"))
    if not jpegs:
        print(f"no fixture jpgs in {fixtures_dir}", file=sys.stderr)
        raise SystemExit(1)
    frames: list[bytes] = []
    sizes: list[tuple[int, int]] = []
    for p in jpegs:
        b = p.read_bytes()
        img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"could not decode fixture {p}", file=sys.stderr)
            raise SystemExit(1)
        h, w = img.shape[:2]
        frames.append(b)
        sizes.append((w, h))
    return frames, sizes


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway-http", default="http://gateway:8000")
    parser.add_argument("--gateway-ws", default="ws://gateway:8000")
    parser.add_argument("--redis", default="redis://redis:6379/0")
    parser.add_argument("--fixtures-dir", default="/fixtures")
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--min-events", type=int, default=1)
    args = parser.parse_args()

    frames, sizes = _load_fixtures(Path(args.fixtures_dir))

    session_id = str(uuid.uuid4())
    print(
        f"harness: session={session_id} fixtures={len(frames)} fps={args.fps} "
        f"duration={args.duration_s}s min_events={args.min_events}",
        flush=True,
    )

    # 1. Wait for the gateway to come up, then register the session.
    async with httpx.AsyncClient(timeout=5.0) as client:
        ready = await _wait_for_gateway(client, args.gateway_http)
        if not ready:
            print("gateway never became ready", file=sys.stderr)
            return 1
        resp = await client.post(
            f"{args.gateway_http}/api/session",
            json={"session_id": session_id},
        )
        if resp.status_code != 200:
            print(f"POST /api/session failed: {resp.status_code} {resp.text}", file=sys.stderr)
            return 1
    print("harness: session created", flush=True)

    # 2. Run producer + subscriber concurrently; collect detection events.
    r = redis_async.from_url(args.redis, decode_responses=False)
    received: list[dict] = []
    stop = asyncio.Event()

    try:
        results = await asyncio.gather(
            producer(r, session_id, frames, sizes, args.duration_s, args.fps, stop),
            subscriber(args.gateway_ws, session_id, stop, received, enough=5),
            return_exceptions=True,
        )
        for res in results:
            if isinstance(res, BaseException) and not isinstance(res, asyncio.CancelledError):
                print(f"harness: task failed: {res!r}", file=sys.stderr)
    finally:
        await r.aclose()
        async with httpx.AsyncClient(timeout=5.0) as client, contextlib.suppress(Exception):
            await client.delete(f"{args.gateway_http}/api/session/{session_id}")

    # 3. Assert.
    print(f"harness: received {len(received)} detection events", flush=True)
    for ev in received[:3]:
        if isinstance(ev, dict):
            print(
                f"  ev: frame_id={ev.get('frame_id')} tracks={len(ev.get('tracks', []) or [])}",
                flush=True,
            )
    if len(received) >= args.min_events:
        print("PASS", flush=True)
        return 0
    print(f"FAIL: wanted >= {args.min_events} events, got {len(received)}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
