"""Gateway FastAPI app.

Wires session lifecycle, WebRTC negotiation, and the detection-event
WebSocket forwarder onto the HTTP/WS surface described in DESIGN.md §5.2.

Lifespan owns:
- a single shared async Redis client (byte-clean — no implicit utf-8 decode),
- a :class:`SessionManager` over that client,
- a :class:`WebRTCManager` whose ``on_close`` hook deletes the session, and
- a background sweep task that tears down idle sessions every 2 seconds.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
from pathlib import Path

import redis.asyncio as redis_async
from fastapi import APIRouter, FastAPI, HTTPException, Request, Response, WebSocket
from fastapi.staticfiles import StaticFiles
from loguru import logger

from app.config import Settings
from app.models import (
    SessionCreateRequest,
    SessionCreateResponse,
    WebRTCAnswerResponse,
    WebRTCOfferRequest,
)
from app.session import SessionAlreadyExists, SessionManager
from app.webrtc import WebRTCManager
from app.websocket import forward_detections

# How often the idle-sweep task wakes up. Independent of
# ``SESSION_IDLE_TIMEOUT_S`` — the timeout sets the staleness threshold,
# this cadence controls how soon after staleness we react.
_IDLE_SWEEP_INTERVAL_S = 2.0


async def sweep_idle_sessions(
    session_manager: SessionManager,
    webrtc_manager: WebRTCManager,
    timeout_s: int,
) -> list[str]:
    """Run one iteration of the idle sweep.

    Extracted from the background loop so it can be unit-tested without
    spinning up the FastAPI lifespan or sleeping for the loop interval.
    Returns the list of session IDs that were torn down — useful both for
    log lines and for test assertions.
    """
    stale = await session_manager.idle_sessions(timeout_s)
    for sid in stale:
        await webrtc_manager.close(sid)
        await session_manager.delete(sid)
    return stale


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise/teardown gateway-wide resources.

    Creates the shared Redis client, the session and WebRTC managers, and
    starts the idle-sweep background task. On shutdown the sweep task is
    cancelled and drained before the Redis client is closed so no
    in-flight Redis call outlives the connection it's running on.
    """
    settings = Settings()
    app.state.settings = settings
    app.state.redis = redis_async.from_url(settings.REDIS_URL, decode_responses=False)
    app.state.session_manager = SessionManager(app.state.redis)
    app.state.webrtc_manager = WebRTCManager(
        r=app.state.redis,
        settings=settings,
        on_close=app.state.session_manager.delete,
        on_frame=app.state.session_manager.touch,
    )

    async def _idle_sweep_loop() -> None:
        """Background loop: periodically tear down idle sessions.

        Exceptions inside one iteration are logged and swallowed so a
        transient Redis hiccup doesn't kill the whole sweeper —
        :class:`asyncio.CancelledError` always propagates so shutdown
        works correctly.
        """
        while True:
            try:
                stale = await sweep_idle_sessions(
                    app.state.session_manager,
                    app.state.webrtc_manager,
                    settings.SESSION_IDLE_TIMEOUT_S,
                )
                for sid in stale:
                    logger.info("session={} idle-sweep tearing down", sid)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("idle_sweep iteration failed: {}", exc)
            await asyncio.sleep(_IDLE_SWEEP_INTERVAL_S)

    sweep_task = asyncio.create_task(_idle_sweep_loop(), name="idle-sweep")

    logger.info("gateway starting; redis={}", settings.REDIS_URL)
    try:
        yield
    finally:
        sweep_task.cancel()
        with suppress(asyncio.CancelledError):
            await sweep_task
        await app.state.redis.aclose()
        logger.info("gateway shutting down")


app = FastAPI(lifespan=lifespan)

api = APIRouter(prefix="/api")


@api.get("/config")
async def config(request: Request) -> dict:
    """Return browser-consumable config. Currently just ICE servers so
    the client's RTCPeerConnection uses the same TURN relay we do."""
    s: Settings = request.app.state.settings
    if not s.TURN_URL:
        return {"iceServers": []}
    return {
        "iceServers": [
            {
                "urls": [s.TURN_URL],
                "username": s.TURN_USERNAME,
                "credential": s.TURN_PASSWORD,
            }
        ]
    }


@api.post("/session", response_model=SessionCreateResponse, status_code=200)
async def create_session(body: SessionCreateRequest, request: Request) -> SessionCreateResponse:
    """Create a new session. 409 if the id is already live."""
    sm: SessionManager = request.app.state.session_manager
    try:
        await sm.create(body.session_id)
    except SessionAlreadyExists as exc:
        raise HTTPException(
            status_code=409,
            detail=f"session {body.session_id} already exists",
        ) from exc
    return SessionCreateResponse(ok=True)


@api.delete("/session/{session_id}", status_code=204)
async def delete_session(session_id: str, request: Request) -> Response:
    """Tear down a session and its WebRTC peer. No-op if absent (still 204)."""
    sm: SessionManager = request.app.state.session_manager
    wm: WebRTCManager = request.app.state.webrtc_manager
    await wm.close(session_id)
    await sm.delete(session_id)
    return Response(status_code=204)


@api.post("/webrtc/offer", response_model=WebRTCAnswerResponse)
async def webrtc_offer(body: WebRTCOfferRequest, request: Request) -> dict[str, str]:
    """Negotiate WebRTC for an existing session; return the SDP answer."""
    sm: SessionManager = request.app.state.session_manager
    wm: WebRTCManager = request.app.state.webrtc_manager
    if not await sm.exists(body.session_id):
        raise HTTPException(
            status_code=404,
            detail=f"session {body.session_id} not found; create it first",
        )
    return await wm.handle_offer(body.session_id, body.sdp, body.type)


app.include_router(api)


@app.websocket("/ws")
async def ws_route(websocket: WebSocket, session_id: str) -> None:
    """Push detection events for ``session_id`` to the connected client.

    Closes immediately with WS code 4404 if the session doesn't exist —
    the client should ``POST /api/session`` before opening the WS.
    """
    sm: SessionManager = websocket.app.state.session_manager
    if not await sm.exists(session_id):
        await websocket.close(code=4404, reason="session not found")
        return
    await websocket.accept()
    await forward_detections(websocket, websocket.app.state.redis, session_id)


@app.get("/healthz")
async def healthz() -> dict:
    """Liveness probe — process is up and serving HTTP."""
    return {"ok": True}


@app.get("/readyz")
async def readyz(response: Response) -> dict:
    """Readiness probe — gateway can reach its Redis dependency.

    Returns 503 if `redis.ping()` raises so an orchestrator can hold
    traffic off until the dependency recovers.
    """
    try:
        await app.state.redis.ping()
        return {"ok": True}
    except Exception as e:
        response.status_code = 503
        return {"ok": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Static SPA mount.
#
# Mounted LAST so every route registered above (the /api router, /ws, the
# health probes) takes precedence — FastAPI route matching is order-
# sensitive, and a "/" mount with html=True would otherwise swallow them.
#
# The mount is conditional on the directory existing because the static
# bundle is baked in only by the production Dockerfile (Task 3.2). In
# dev (pytest, local uvicorn without docker) /app/static is absent and
# we want main.py to import cleanly anyway.
#
# html=True turns the mount into an SPA-aware server: GET / serves
# index.html and unknown sub-paths fall back to index.html so client-
# side routing works.
# ---------------------------------------------------------------------------
# The prod Dockerfile COPYs the SPA to /app/static; the __file__-relative
# fallback covers editable installs (dev). First existing path wins.
_STATIC_DIR = next(
    (
        p
        for p in (
            Path("/app/static"),
            Path(__file__).resolve().parent.parent / "static",
        )
        if p.is_dir()
    ),
    None,
)
if _STATIC_DIR is not None:
    app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")
