"""WebSocket detection forwarder for the gateway.

One forwarder task per WebSocket connection. Reads detection events from
``detections:{session_id}`` via :func:`app.redis_io.consume_detections`,
pushes them to the client as JSON, and responds to ``{"type":"ping"}`` with
``{"type":"pong"}``.

The forwarder exits cleanly when:

- the client disconnects (FastAPI raises :class:`WebSocketDisconnect`),
- the consumer cancels the forwarder externally (idle-session sweep),
- an unrecoverable error fires (caught + logged, task exits).

The send and receive paths must run concurrently — clients should be able to
ping while detection events stream — so we spawn one task per direction and
join with :func:`asyncio.wait` using ``FIRST_COMPLETED``. Whichever side
terminates first triggers cancellation of the other; we drain those
cancellations so no background tasks leak past the route handler's return.
"""

from __future__ import annotations

import asyncio
import json

import redis.asyncio as redis_async
from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger

from app.redis_io import consume_detections


async def forward_detections(
    websocket: WebSocket,
    redis: redis_async.Redis,
    session_id: str,
) -> None:
    """Pump detection events from Redis to a WebSocket; handle ping/pong.

    The WebSocket must already be accepted by the caller (FastAPI route
    handler). This coroutine returns when the client disconnects, either
    direction errors out, or the surrounding task is cancelled.
    """
    sender = asyncio.create_task(_send_loop(websocket, redis, session_id), name="ws-send")
    receiver = asyncio.create_task(_recv_loop(websocket), name="ws-recv")

    done: set[asyncio.Task[None]] = set()
    try:
        done, _pending = await asyncio.wait(
            {sender, receiver},
            return_when=asyncio.FIRST_COMPLETED,
        )
    except asyncio.CancelledError:
        # Outer cancellation (e.g. TestClient teardown, idle-sweep). Fall
        # through into the finally block to clean both tasks up.
        pass
    finally:
        sender.cancel()
        receiver.cancel()
        # ``asyncio.shield`` keeps the drain alive even if our caller has
        # already cancelled us — without it, the shielded tasks would leak
        # and the route handler would unwind with a dangling sender task
        # still inside :func:`consume_detections`'s ``xread`` block.
        await asyncio.shield(asyncio.gather(sender, receiver, return_exceptions=True))

    # Surface any unexpected exceptions from the completed task (best-effort
    # logging only — the WS is already torn down by the time we get here).
    for task in done:
        if task.cancelled():
            continue
        exc = task.exception()
        if exc and not isinstance(exc, WebSocketDisconnect | asyncio.CancelledError):
            logger.warning(
                "session={} ws task {} failed: {}",
                session_id,
                task.get_name(),
                exc,
            )


async def _send_loop(
    websocket: WebSocket,
    redis: redis_async.Redis,
    session_id: str,
) -> None:
    """Forward detection events from Redis to the WebSocket as JSON."""
    try:
        async for event in consume_detections(redis, session_id):
            await websocket.send_text(json.dumps(event))
    except WebSocketDisconnect:
        # Normal termination — client closed.
        pass
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("session={} send_loop aborted: {}", session_id, exc)


async def _recv_loop(websocket: WebSocket) -> None:
    """Receive client messages; reply to ping with pong.

    Malformed (non-JSON) client frames are silently ignored — we'd rather
    keep the WS alive than tear it down on bad input. The only frame the
    server actually acts on today is ``{"type":"ping"}``.
    """
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(msg, dict) and msg.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        # Normal termination.
        pass
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning("recv_loop aborted: {}", exc)
