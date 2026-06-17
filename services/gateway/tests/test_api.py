"""HTTP API endpoint tests for the gateway.

Strategy: inject a fakeredis-backed :class:`SessionManager` and
:class:`WebRTCManager` onto ``app.state`` BEFORE the ASGITransport client
connects, bypassing the real :func:`app.main.lifespan`. The lifespan does
real Redis connect + spawns a background idle-sweep task which we don't
want firing during unit tests; manual state injection sidesteps both.

The WebSocket route is covered by ``tests/test_websocket.py`` which spins
up its own tiny FastAPI app around :func:`forward_detections`. Adding a
full WS-route integration test here that exercises both the session check
AND the forwarder would be redundant — the wiring is one line and the
forwarder semantics are already exhaustively tested.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import fakeredis.aioredis
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.config import Settings
from app.main import app, sweep_idle_sessions
from app.session import SessionManager
from app.webrtc import WebRTCManager


@pytest_asyncio.fixture
async def client_with_fakeredis() -> (
    AsyncIterator[tuple[AsyncClient, fakeredis.aioredis.FakeRedis, SessionManager, WebRTCManager]]
):
    """Inject fakeredis-backed managers onto ``app.state`` and yield a client.

    We DON'T enter ``app.router.lifespan_context`` — the real lifespan
    would overwrite our manually-installed managers and start a 2-second
    background sweep loop. The handlers only read ``app.state.*``, so
    pre-populating the state attributes is sufficient.
    """
    server = fakeredis.aioredis.FakeServer()
    fake = fakeredis.aioredis.FakeRedis(server=server, decode_responses=False)
    settings = Settings(_env_file=None)
    sm = SessionManager(fake)
    wm = WebRTCManager(r=fake, settings=settings, on_close=sm.delete)

    app.state.redis = fake
    app.state.session_manager = sm
    app.state.webrtc_manager = wm
    app.state.settings = settings

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac, fake, sm, wm

    await fake.aclose()


async def test_create_session_returns_200_ok(client_with_fakeredis) -> None:
    """``POST /api/session`` with a fresh id returns ``{"ok": true}``."""
    ac, _r, sm, _wm = client_with_fakeredis
    sid = str(uuid.uuid4())
    resp = await ac.post("/api/session", json={"session_id": sid})
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
    assert await sm.exists(sid)


async def test_create_session_over_active_cap_returns_429(client_with_fakeredis) -> None:
    """A create beyond the active-session cap is rejected with 429."""
    ac, fake, _sm, _wm = client_with_fakeredis
    # Swap in a capped manager (the fixture installs an uncapped one).
    app.state.session_manager = SessionManager(fake, max_active_sessions=1)
    first = await ac.post("/api/session", json={"session_id": str(uuid.uuid4())})
    assert first.status_code == 200
    second = await ac.post("/api/session", json={"session_id": str(uuid.uuid4())})
    assert second.status_code == 429


async def test_create_session_duplicate_returns_409(client_with_fakeredis) -> None:
    """Second create with the same id is rejected with 409."""
    ac, _r, _sm, _wm = client_with_fakeredis
    sid = str(uuid.uuid4())
    first = await ac.post("/api/session", json={"session_id": sid})
    assert first.status_code == 200
    dup = await ac.post("/api/session", json={"session_id": sid})
    assert dup.status_code == 409
    assert sid in dup.json()["detail"]


async def test_create_session_empty_body_returns_422(client_with_fakeredis) -> None:
    """Missing ``session_id`` field surfaces as 422 from pydantic validation."""
    ac, *_ = client_with_fakeredis
    resp = await ac.post("/api/session", json={})
    assert resp.status_code == 422


async def test_create_session_empty_session_id_returns_422(
    client_with_fakeredis,
) -> None:
    """Empty-string ``session_id`` violates the ``NonEmptyStr`` constraint."""
    ac, *_ = client_with_fakeredis
    resp = await ac.post("/api/session", json={"session_id": ""})
    assert resp.status_code == 422


async def test_create_session_extra_field_returns_422(
    client_with_fakeredis,
) -> None:
    """``extra="forbid"`` rejects unknown fields with 422."""
    ac, *_ = client_with_fakeredis
    resp = await ac.post("/api/session", json={"session_id": "x", "extra": "nope"})
    assert resp.status_code == 422


async def test_delete_session_returns_204(client_with_fakeredis) -> None:
    """``DELETE /api/session/{id}`` removes the session and returns 204."""
    ac, _r, sm, _wm = client_with_fakeredis
    sid = str(uuid.uuid4())
    await ac.post("/api/session", json={"session_id": sid})
    resp = await ac.delete(f"/api/session/{sid}")
    assert resp.status_code == 204
    assert not await sm.exists(sid)


async def test_delete_session_missing_returns_204(client_with_fakeredis) -> None:
    """Deleting a non-existent session is a no-op (still 204)."""
    ac, *_ = client_with_fakeredis
    resp = await ac.delete(f"/api/session/{uuid.uuid4()}")
    assert resp.status_code == 204


async def test_webrtc_offer_missing_session_returns_404(
    client_with_fakeredis,
) -> None:
    """``POST /api/webrtc/offer`` for an unknown session is rejected with 404."""
    ac, *_ = client_with_fakeredis
    sid = str(uuid.uuid4())
    resp = await ac.post(
        "/api/webrtc/offer",
        json={"session_id": sid, "sdp": "v=0\n", "type": "offer"},
    )
    assert resp.status_code == 404
    assert sid in resp.json()["detail"]


async def test_webrtc_offer_invalid_type_returns_422(
    client_with_fakeredis,
) -> None:
    """``type`` field must be the literal ``"offer"`` — anything else is 422."""
    ac, *_ = client_with_fakeredis
    sid = str(uuid.uuid4())
    await ac.post("/api/session", json={"session_id": sid})
    resp = await ac.post(
        "/api/webrtc/offer",
        json={"session_id": sid, "sdp": "v=0\n", "type": "answer"},
    )
    assert resp.status_code == 422


async def test_webrtc_offer_empty_sdp_returns_422(client_with_fakeredis) -> None:
    """Empty SDP violates the ``NonEmptyStr`` constraint on ``sdp``."""
    ac, *_ = client_with_fakeredis
    sid = str(uuid.uuid4())
    await ac.post("/api/session", json={"session_id": sid})
    resp = await ac.post(
        "/api/webrtc/offer",
        json={"session_id": sid, "sdp": "", "type": "offer"},
    )
    assert resp.status_code == 422


async def test_webrtc_offer_oversized_sdp_returns_422(client_with_fakeredis) -> None:
    """An absurdly large SDP is rejected before it ever reaches aiortc.

    Real SDPs are a few KB; the unauthenticated offer endpoint must bound
    the body so a client can't feed an arbitrarily large blob to the parser.
    """
    ac, *_ = client_with_fakeredis
    sid = str(uuid.uuid4())
    await ac.post("/api/session", json={"session_id": sid})
    huge_sdp = "v=0\n" + ("a=x\n" * 100_000)  # ~400 KB
    resp = await ac.post(
        "/api/webrtc/offer",
        json={"session_id": sid, "sdp": huge_sdp, "type": "offer"},
    )
    assert resp.status_code == 422


# --------------------------------------------------------------------------
# Idle-sweep helper tests
#
# The lifespan-attached background loop is hard to test deterministically
# (it sleeps 2s between iterations). Instead we test the extracted
# single-iteration helper directly — same logic, no timing dependency.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sweep_idle_sessions_tears_down_idle() -> None:
    """A session whose metadata key is gone is reported and removed."""
    server = fakeredis.aioredis.FakeServer()
    fake = fakeredis.aioredis.FakeRedis(server=server, decode_responses=False)
    settings = Settings(_env_file=None)
    sm = SessionManager(fake)
    wm = WebRTCManager(r=fake, settings=settings, on_close=sm.delete)

    sid = "idle-session"
    await sm.create(sid)
    # Force "idle" by dropping the metadata key while leaving the active-set
    # entry — :meth:`SessionManager.idle_sessions` treats absent metadata as
    # definitively stale (key TTL'd out).
    await fake.delete(SessionManager._key(sid))

    stale = await sweep_idle_sessions(sm, wm, timeout_s=settings.SESSION_IDLE_TIMEOUT_S)
    assert stale == [sid]
    # Active-set entry should now be gone too — delete() removes it.
    assert not await sm.exists(sid)
    members = await fake.smembers(SessionManager.ACTIVE_SET)
    assert sid.encode() not in members

    await fake.aclose()


@pytest.mark.asyncio
async def test_sweep_idle_sessions_skips_fresh() -> None:
    """A session that was just created is NOT swept (well under the timeout)."""
    server = fakeredis.aioredis.FakeServer()
    fake = fakeredis.aioredis.FakeRedis(server=server, decode_responses=False)
    settings = Settings(_env_file=None)
    sm = SessionManager(fake)
    wm = WebRTCManager(r=fake, settings=settings, on_close=sm.delete)

    sid = "fresh-session"
    await sm.create(sid)

    stale = await sweep_idle_sessions(sm, wm, timeout_s=settings.SESSION_IDLE_TIMEOUT_S)
    assert stale == []
    assert await sm.exists(sid)

    await fake.aclose()
