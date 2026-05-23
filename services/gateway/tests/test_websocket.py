"""Tests for ``app.websocket`` — the per-connection WS detection forwarder.

Strategy: spin up a tiny FastAPI app inside each test that exposes
:func:`app.websocket.forward_detections` on ``/ws`` and drive it via
:class:`fastapi.testclient.TestClient`. The forwarder reads from
fakeredis backed by a SHARED :class:`fakeredis.FakeServer` so the
TestClient's event loop and the producer thread (which runs its own
loop) share the same in-memory dataset — fakeredis client objects
themselves are bound to a loop, but the underlying server is not.
"""

from __future__ import annotations

import asyncio
import json
import threading

import fakeredis
import fakeredis.aioredis
import pytest
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient

from app.redis_io import publish_detection_event
from app.websocket import forward_detections


@pytest.fixture
def fake_server() -> fakeredis.FakeServer:
    """Shared in-memory Redis server (loop-agnostic).

    Async clients are bound to whichever event loop opened them, but the
    underlying server state survives loop boundaries. Each test gets a
    fresh server so streams don't leak between tests.
    """
    return fakeredis.FakeServer()


def _make_app(fake_server: fakeredis.FakeServer) -> FastAPI:
    """Tiny FastAPI app that exposes :func:`forward_detections` on ``/ws``.

    A fresh async fakeredis client is created INSIDE the route on the
    server's event loop so it binds correctly. The shared
    :class:`fakeredis.FakeServer` keeps state consistent across loops.
    """
    app = FastAPI()

    @app.websocket("/ws")
    async def ws_route(websocket: WebSocket, session_id: str) -> None:
        await websocket.accept()
        r = fakeredis.aioredis.FakeRedis(server=fake_server, decode_responses=False)
        try:
            await forward_detections(websocket, r, session_id)
        finally:
            await r.aclose()

    return app


def test_ping_pong_roundtrip(fake_server: fakeredis.FakeServer) -> None:
    """Client sends ``{"type":"ping"}``; server replies ``{"type":"pong"}``."""
    app = _make_app(fake_server)
    with (
        TestClient(app) as client,
        client.websocket_connect("/ws?session_id=ping_test") as ws,
    ):
        ws.send_text(json.dumps({"type": "ping"}))
        data = json.loads(ws.receive_text())
        assert data == {"type": "pong"}


def test_detection_event_forwarded(fake_server: fakeredis.FakeServer) -> None:
    """Publishing to ``detections:{sid}`` pushes the event to the WS as JSON.

    The forwarder defaults to ``last_id=b"$"`` (only events arriving AFTER
    the consumer subscribes), so we publish AFTER the WS connects. Since
    :class:`TestClient` proxies the WS synchronously on the main thread, the
    producer runs in a separate thread with its own event loop (sharing the
    in-memory :class:`FakeServer`) and a brief delay so the forwarder's
    XREAD is already blocking when the XADD lands.
    """
    app = _make_app(fake_server)
    event = {"session_id": "evt_test", "frame_id": 42, "tracks": []}

    # Stop signalled from the main thread once the WS receives an event.
    stop = threading.Event()

    async def publish_loop() -> None:
        producer_r = fakeredis.aioredis.FakeRedis(server=fake_server, decode_responses=False)
        try:
            # consume_detections uses block_ms=100 with last_id=b"$"; after
            # an empty XREAD returns, the next call re-resolves "$" to the
            # current tail, so any XADD that landed in that gap is silently
            # missed (this is a real Redis semantic, not a fakeredis quirk).
            # Publishing on a short loop until the consumer confirms receipt
            # closes the race window deterministically.
            while not stop.is_set():
                await publish_detection_event(producer_r, session_id="evt_test", event=event)
                await asyncio.sleep(0.05)
        finally:
            await producer_r.aclose()

    def producer() -> None:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(publish_loop())
        finally:
            loop.close()

    with (
        TestClient(app) as client,
        client.websocket_connect("/ws?session_id=evt_test") as ws,
    ):
        t = threading.Thread(target=producer)
        t.start()
        try:
            data = json.loads(ws.receive_text())
        finally:
            stop.set()
            t.join()
        assert data == event


def test_malformed_client_message_ignored(
    fake_server: fakeredis.FakeServer,
) -> None:
    """A non-JSON client message doesn't crash the forwarder.

    Follow-up ping must still be answered — proves the receive loop kept
    running after the malformed frame.
    """
    app = _make_app(fake_server)
    with (
        TestClient(app) as client,
        client.websocket_connect("/ws?session_id=junk") as ws,
    ):
        ws.send_text("not json")
        ws.send_text(json.dumps({"type": "ping"}))
        data = json.loads(ws.receive_text())
        assert data == {"type": "pong"}
