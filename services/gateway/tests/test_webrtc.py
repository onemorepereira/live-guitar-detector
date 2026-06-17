"""Tests for ``app.webrtc`` — WebRTC peer wrangling + frame ingest.

Two tiers:

* **Always-on, pure-logic tests** exercise the rate-limit decision
  function and basic ``WebRTCManager`` bookkeeping without ever
  instantiating a real ``RTCPeerConnection``. These are the regression
  net we care about most for CI.

* **``requires_aiortc_peer`` integration tests** stand up a real
  ``RTCPeerConnection`` against ours and exchange SDP / data channels.
  These need the libsrtp/libvpx stack and are gated behind
  ``RUN_AIORTC_TESTS=1`` (see ``conftest.py``).
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import fakeredis.aioredis
import pytest
import pytest_asyncio
from aiortc import RTCPeerConnection

from app.config import Settings
from app.webrtc import WebRTCManager, _PeerEntry, _should_drop


@pytest_asyncio.fixture
async def r() -> AsyncIterator[fakeredis.aioredis.FakeRedis]:
    """Byte-clean fakeredis instance — JPEG bytes must survive untouched."""
    client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    try:
        yield client
    finally:
        await client.aclose()


@pytest_asyncio.fixture
async def manager(r) -> AsyncIterator[WebRTCManager]:
    """``WebRTCManager`` with a recording ``on_close`` callback.

    The list of closed session ids is attached to the manager as
    ``_closed_log`` so assertions can inspect what teardown saw.
    """
    closed: list[str] = []

    async def on_close(sid: str) -> None:
        closed.append(sid)

    mgr = WebRTCManager(r=r, settings=Settings(_env_file=None), on_close=on_close)
    mgr._closed_log = closed  # type: ignore[attr-defined]
    try:
        yield mgr
    finally:
        # Best-effort teardown of any peers left lying around.
        for sid in list(mgr._peers):
            await mgr.close(sid)


# ---------------------------------------------------------------------------
# Pure-logic tests — these run on every CI invocation.
# ---------------------------------------------------------------------------


def test_should_drop_within_interval() -> None:
    """Frames that arrive before ``min_interval_ms`` has elapsed get dropped."""
    entry = _PeerEntry(peer=None)  # type: ignore[arg-type]
    entry.last_publish_ms = 1000
    assert _should_drop(entry, now_ms=1010, min_interval_ms=33)


def test_should_drop_outside_interval() -> None:
    """Once the budget has elapsed the next frame is allowed through."""
    entry = _PeerEntry(peer=None)  # type: ignore[arg-type]
    entry.last_publish_ms = 1000
    assert not _should_drop(entry, now_ms=1050, min_interval_ms=33)


def test_should_drop_first_frame_passes() -> None:
    """``last_publish_ms == 0`` (no frame yet) must never be dropped."""
    entry = _PeerEntry(peer=None)  # type: ignore[arg-type]
    assert not _should_drop(entry, now_ms=100, min_interval_ms=33)


def test_should_drop_boundary_exact_interval() -> None:
    """A frame arriving exactly at ``min_interval_ms`` is NOT dropped.

    Decision is strict ``<``, so the equality case passes — important to
    avoid steady-state starvation when the source is locked to the same
    cadence as the ingest cap.
    """
    entry = _PeerEntry(peer=None)  # type: ignore[arg-type]
    entry.last_publish_ms = 1000
    assert not _should_drop(entry, now_ms=1033, min_interval_ms=33)


async def test_manager_construction(r) -> None:
    """``WebRTCManager`` constructs cleanly with the documented args."""

    async def noop(_sid: str) -> None:
        return None

    mgr = WebRTCManager(r=r, settings=Settings(_env_file=None), on_close=noop)
    assert mgr._peers == {}


async def test_close_unknown_session_is_noop(manager: WebRTCManager) -> None:
    """Closing a session that was never opened must not raise."""
    await manager.close("never-existed")
    assert manager._peers == {}


async def test_teardown_unknown_session_still_fires_on_close(
    manager: WebRTCManager,
) -> None:
    """``_teardown`` runs the on_close callback even with no peer entry.

    This matches the production path where the connectionstatechange
    handler schedules a teardown; the entry may already be gone by the
    time the teardown coroutine runs, but the caller's cleanup hook still
    has to fire so the session is removed from Redis.
    """
    await manager._teardown("ghost")
    assert manager._closed_log == ["ghost"]  # type: ignore[attr-defined]


async def test_close_all_closes_every_peer(manager: WebRTCManager) -> None:
    """``close_all`` tears down every live peer and empties the registry.

    Used by the app-shutdown path so peer connections (and their ingest
    tasks) don't leak when the process stops.
    """
    closed: list[int] = []

    class _FakePeer:
        async def close(self) -> None:
            closed.append(id(self))

    p1, p2 = _FakePeer(), _FakePeer()
    manager._peers["a"] = _PeerEntry(peer=p1)  # type: ignore[arg-type]
    manager._peers["b"] = _PeerEntry(peer=p2)  # type: ignore[arg-type]

    await manager.close_all()

    assert manager._peers == {}
    assert len(closed) == 2


# ---------------------------------------------------------------------------
# Integration tests — only run when RUN_AIORTC_TESTS=1.
# ---------------------------------------------------------------------------


@pytest.mark.requires_aiortc_peer
async def test_handle_offer_returns_valid_answer(manager: WebRTCManager) -> None:
    """Negotiate against a client RTCPeerConnection; answer SDP must be valid.

    We use a data channel (not a media track) on the client side — the
    SDP shape is what matters here, not real media transport. Avoids the
    libvpx / V4L2 surface area entirely.
    """
    client = RTCPeerConnection()
    try:
        client.createDataChannel("control")
        offer = await client.createOffer()
        await client.setLocalDescription(offer)

        answer = await manager.handle_offer(
            "s1",
            client.localDescription.sdp,
            client.localDescription.type,
        )
        assert answer["type"] == "answer"
        assert "v=0" in answer["sdp"]
        assert "s1" in manager._peers
    finally:
        await client.close()


@pytest.mark.requires_aiortc_peer
async def test_reoffer_replaces_previous_peer(manager: WebRTCManager) -> None:
    """A second offer for the same session id tears down the previous peer."""
    client_a = RTCPeerConnection()
    client_b = RTCPeerConnection()
    try:
        client_a.createDataChannel("ctl")
        offer_a = await client_a.createOffer()
        await client_a.setLocalDescription(offer_a)
        await manager.handle_offer(
            "s2", client_a.localDescription.sdp, client_a.localDescription.type
        )
        first_peer = manager._peers["s2"].peer

        client_b.createDataChannel("ctl")
        offer_b = await client_b.createOffer()
        await client_b.setLocalDescription(offer_b)
        await manager.handle_offer(
            "s2", client_b.localDescription.sdp, client_b.localDescription.type
        )
        second_peer = manager._peers["s2"].peer
        assert second_peer is not first_peer
    finally:
        await client_a.close()
        await client_b.close()
