"""WebRTC peer management for the gateway.

One :class:`aiortc.RTCPeerConnection` per session. When a video track
arrives we spawn an asyncio task that:

- ``recv()`` :class:`av.VideoFrame` objects in a loop
- decodes them to numpy BGR via ``to_ndarray(format="bgr24")``
- rate-limits to ``settings.MAX_INGEST_FPS`` (drops over-budget frames)
- JPEG-encodes at ``settings.JPEG_QUALITY``
- ``XADD``s to ``frames:{session_id}`` via :mod:`app.redis_io`

Peer-connection state transitions to ``closed`` / ``failed`` /
``disconnected`` trigger a teardown callback supplied by the caller
(typically :py:meth:`app.session.SessionManager.delete`).

The 30 FPS ingest cap is the canonical Phase 1 backpressure mechanism
(DESIGN.md §5.5): the client can push at whatever cadence its camera
produces, but anything exceeding the budget is silently dropped on the
gateway side before it ever hits Redis. The decision is factored into
:func:`_should_drop` so it can be unit-tested without aiortc machinery.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import cv2
import redis.asyncio as redis_async
from aiortc import (
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)
from aiortc.contrib.media import MediaBlackhole
from loguru import logger

from app.config import Settings
from app.redis_io import publish_frame


@dataclass
class _PeerEntry:
    """Bookkeeping for a single live peer connection.

    ``last_publish_ms`` is the unix-millisecond timestamp of the most
    recent frame we *accepted* (i.e. didn't drop for rate-limit); it
    starts at 0 so the first frame is always passed through.

    ``frame_counter`` is a per-session monotonically-increasing integer
    stamped into the ``frame_id`` field of each published frame so the
    worker can detect gaps caused by drops.
    """

    peer: RTCPeerConnection
    task: asyncio.Task | None = None  # the frame-receive loop
    last_publish_ms: int = 0
    frame_counter: int = 0


def _should_drop(entry: _PeerEntry, now_ms: int, min_interval_ms: int) -> bool:
    """Return True if a frame arriving at ``now_ms`` violates the FPS cap.

    Pure function — no aiortc, no Redis. The strict ``<`` comparison
    matters: a frame arriving exactly at the budget boundary is
    accepted, which keeps source cadences locked to ``MAX_INGEST_FPS``
    from starving.
    """
    return now_ms - entry.last_publish_ms < min_interval_ms


def _build_ice_config(settings: Settings) -> RTCConfiguration | None:
    """Construct an :class:`RTCConfiguration` from TURN_* settings.

    Returns ``None`` when no TURN URL is configured so aiortc falls back
    to its default (no ICE servers). When TURN is set, the same server
    list is exposed to the browser via ``GET /api/config`` so both ends
    of the call use the relay.
    """
    if not settings.TURN_URL:
        return None
    return RTCConfiguration(
        iceServers=[
            RTCIceServer(
                urls=[settings.TURN_URL],
                username=settings.TURN_USERNAME or None,
                credential=settings.TURN_PASSWORD or None,
            )
        ]
    )


class WebRTCManager:
    """Holds one :class:`RTCPeerConnection` per session; bridges video to Redis.

    The caller (Task 2.6 API layer) constructs ``WebRTCManager`` once at
    startup with a shared Redis client, the gateway :class:`Settings`,
    and a teardown callback that deletes the session (typically
    :py:meth:`SessionManager.delete`).
    """

    def __init__(
        self,
        r: redis_async.Redis,
        settings: Settings,
        on_close: Callable[[str], Awaitable[None]],
        on_frame: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self._r = r
        self._settings = settings
        self._on_close = on_close
        # Called once per accepted frame so the caller can refresh the
        # session's idle timer. Typically wired to SessionManager.touch.
        self._on_frame = on_frame
        self._peers: dict[str, _PeerEntry] = {}
        # Strong refs to "fire and forget" tasks (audio blackhole start,
        # teardown coroutines scheduled from state callbacks). Without
        # this set the event loop holds only a weak reference and the
        # task can be GC'd mid-flight (per CPython asyncio docs / RUF006).
        self._bg_tasks: set[asyncio.Task] = set()

    async def handle_offer(self, session_id: str, sdp: str, sdp_type: str) -> dict[str, str]:
        """Negotiate WebRTC for ``session_id``; return the answer SDP/type.

        Idempotent on re-offer: if a peer already exists for this session
        we tear it down first so we don't leak the previous connection.
        In normal operation the client should ``DELETE`` the session
        before sending a second offer, but defending against this here
        keeps us safe against client bugs.
        """
        if session_id in self._peers:
            await self.close(session_id)

        peer = RTCPeerConnection(configuration=_build_ice_config(self._settings))
        entry = _PeerEntry(peer=peer)
        self._peers[session_id] = entry

        # Explicitly declare a recvonly video transceiver BEFORE
        # setRemoteDescription. Without this, aiortc's auto-negotiation
        # from the client's offer is unreliable across versions — the
        # transceiver may end up inactive and `on_track` never fires.
        peer.addTransceiver("video", direction="recvonly")

        @peer.on("track")
        def on_track(track) -> None:
            logger.info("session={} on_track kind={}", session_id, track.kind)
            if track.kind != "video":
                # We don't currently use audio; sink it to /dev/null so the
                # remote isn't backpressured by buffered packets we never
                # drain. MediaBlackhole is the canonical aiortc helper.
                blackhole = MediaBlackhole()
                blackhole.addTrack(track)
                self._spawn_bg(blackhole.start())
                return
            entry.task = asyncio.create_task(self._consume_video(session_id, track))

        @peer.on("connectionstatechange")
        async def on_state() -> None:
            state = peer.connectionState
            logger.debug("session={} peer state={}", session_id, state)
            if state in ("closed", "failed", "disconnected"):
                # Schedule teardown rather than awaiting it here — the
                # state callback shouldn't block aiortc's internal loop.
                self._spawn_bg(self._teardown(session_id))

        # Dump candidates from both SDPs — when TURN is in play and ICE
        # is failing, seeing each side's candidate list is the only way to
        # confirm what's actually being advertised.
        offer_cands = [ln for ln in sdp.splitlines() if "candidate" in ln.lower()]
        logger.info("session={} OFFER candidates: {}", session_id, offer_cands or "none yet")

        await peer.setRemoteDescription(RTCSessionDescription(sdp=sdp, type=sdp_type))
        answer = await peer.createAnswer()
        await peer.setLocalDescription(answer)

        answer_sdp = peer.localDescription.sdp
        answer_cands = [ln for ln in answer_sdp.splitlines() if "candidate" in ln.lower()]
        logger.info("session={} ANSWER candidates: {}", session_id, answer_cands or "none gathered")

        return {"sdp": answer_sdp, "type": peer.localDescription.type}

    def _spawn_bg(self, coro) -> asyncio.Task:
        """Schedule a fire-and-forget coroutine, retaining a strong ref.

        ``asyncio`` only holds weak references to tasks created via
        :func:`asyncio.create_task` / :func:`asyncio.ensure_future`, so
        an un-awaited handle can be garbage-collected mid-flight (Python
        bug #91887). Stash the task here and drop it on completion.
        """
        task = asyncio.ensure_future(coro)
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)
        return task

    async def close(self, session_id: str) -> None:
        """Close the peer for ``session_id`` and cancel its ingest task.

        Does NOT fire ``on_close`` — that's the :meth:`_teardown` path,
        which composes ``close`` with the caller's session-deletion hook.
        Callers that want both behaviours should call :meth:`_teardown`
        (or it'll fire automatically on a real peer state transition).
        """
        entry = self._peers.pop(session_id, None)
        if entry is None:
            return
        if entry.task is not None and not entry.task.done():
            entry.task.cancel()
        try:
            await entry.peer.close()
        except Exception as exc:
            logger.warning("error closing peer for session={}: {}", session_id, exc)

    async def _teardown(self, session_id: str) -> None:
        """Close the peer and notify the caller's ``on_close`` hook.

        Used by the ``connectionstatechange`` handler when the peer goes
        to ``closed`` / ``failed`` / ``disconnected``. Errors from the
        caller's hook are logged but not re-raised — we've done our
        local cleanup and don't want a hook failure to crash the task.
        """
        await self.close(session_id)
        try:
            await self._on_close(session_id)
        except Exception as exc:
            logger.warning("on_close raised for session={}: {}", session_id, exc)

    async def _consume_video(self, session_id: str, track) -> None:
        """Receive frames, rate-limit, JPEG-encode, publish to Redis.

        Cancellation is the expected stop signal — :meth:`close` cancels
        this task. We re-raise :class:`asyncio.CancelledError` so the
        runtime can complete teardown cleanly; other exceptions are
        logged and the loop exits (the peer is already going away).
        """
        entry = self._peers.get(session_id)
        if entry is None:
            return

        min_interval_ms = int(1000 / max(1, self._settings.MAX_INGEST_FPS))
        jpeg_quality = int(self._settings.JPEG_QUALITY)

        logger.info("session={} _consume_video started", session_id)
        try:
            while True:
                frame = await track.recv()
                now_ms = int(time.time() * 1000)
                if _should_drop(entry, now_ms, min_interval_ms):
                    continue
                entry.last_publish_ms = now_ms
                if entry.frame_counter == 0:
                    logger.info(
                        "session={} first frame received {}x{}",
                        session_id,
                        frame.width,
                        frame.height,
                    )

                # av.VideoFrame → numpy BGR. ``to_ndarray(format="bgr24")``
                # returns a contiguous H x W x 3 uint8 array suitable for
                # direct cv2 consumption.
                bgr = frame.to_ndarray(format="bgr24")
                height, width = bgr.shape[:2]
                ok, jpeg = cv2.imencode(
                    ".jpg",
                    bgr,
                    (cv2.IMWRITE_JPEG_QUALITY, jpeg_quality),
                )
                if not ok:
                    logger.warning("session={} JPEG encode failed", session_id)
                    continue

                entry.frame_counter += 1
                await publish_frame(
                    self._r,
                    session_id=session_id,
                    jpeg_bytes=bytes(jpeg),
                    frame_id=entry.frame_counter,
                    frame_ts=now_ms,
                    width=width,
                    height=height,
                )
                if self._on_frame is not None:
                    try:
                        await self._on_frame(session_id)
                    except Exception as exc:
                        logger.warning("on_frame raised for session={}: {}", session_id, exc)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("session={} frame ingest aborted: {}", session_id, exc)
