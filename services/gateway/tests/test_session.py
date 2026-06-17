"""Session lifecycle tests for the gateway.

Uses ``fakeredis.aioredis.FakeRedis`` as an in-process stand-in for Redis;
all the commands we rely on (``SET NX EX``, ``SADD``, ``SREM``, ``SMEMBERS``,
``GET``, ``DEL``, ``EXISTS``) are supported there.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from unittest.mock import patch

import fakeredis.aioredis
import pytest
import pytest_asyncio

from app.session import SessionAlreadyExists, SessionLimitReached, SessionManager


@pytest_asyncio.fixture
async def manager() -> AsyncIterator[SessionManager]:
    """Fresh SessionManager backed by an isolated fakeredis instance per test."""
    r = fakeredis.aioredis.FakeRedis()
    try:
        yield SessionManager(r)
    finally:
        await r.aclose()


async def test_create_then_exists(manager: SessionManager) -> None:
    sid = "abc"
    state = await manager.create(sid)
    assert state.session_id == sid
    assert state.created_ts == state.last_frame_ts
    assert await manager.exists(sid)


async def test_double_create_raises(manager: SessionManager) -> None:
    await manager.create("dup")
    with pytest.raises(SessionAlreadyExists):
        await manager.create("dup")


async def test_delete_removes_metadata_and_active_entry(manager: SessionManager) -> None:
    sid = "del"
    await manager.create(sid)
    assert await manager.exists(sid)
    # Confirm presence in the active set before deletion.
    members_before = await manager._r.smembers(SessionManager.ACTIVE_SET)
    assert any(m.decode() == sid for m in members_before)

    await manager.delete(sid)
    assert not await manager.exists(sid)
    members_after = await manager._r.smembers(SessionManager.ACTIVE_SET)
    assert all(m.decode() != sid for m in members_after)

    # Re-creating after delete must succeed (no stale guard).
    await manager.create(sid)


async def test_delete_clears_streams(manager: SessionManager) -> None:
    """Stream keys for the session must be removed on delete()."""
    sid = "streamy"
    await manager.create(sid)
    # Plant something in both streams so we can verify deletion.
    await manager._r.xadd(f"frames:{sid}", {"f": b"0"})
    await manager._r.xadd(f"detections:{sid}", {"d": b"0"})
    assert await manager._r.exists(f"frames:{sid}") == 1
    assert await manager._r.exists(f"detections:{sid}") == 1

    await manager.delete(sid)
    assert await manager._r.exists(f"frames:{sid}") == 0
    assert await manager._r.exists(f"detections:{sid}") == 0


async def test_touch_updates_last_frame_ts(manager: SessionManager) -> None:
    sid = "touch"
    await manager.create(sid)
    # Patch _now_ms to force a deterministic future timestamp.
    with patch.object(SessionManager, "_now_ms", return_value=999_999_999):
        await manager.touch(sid)
    raw = await manager._r.get(SessionManager._key(sid))
    assert raw is not None
    data = json.loads(raw)
    assert data["last_frame_ts"] == 999_999_999


async def test_create_rejects_when_at_active_session_cap() -> None:
    """With ``max_active_sessions`` set, a create beyond the cap is rejected.

    The worker shares one ByteTrack tracker across sessions, so two concurrent
    sessions corrupt each other's tracks. The gateway caps active sessions to
    keep that from happening.
    """
    r = fakeredis.aioredis.FakeRedis()
    try:
        mgr = SessionManager(r, max_active_sessions=1)
        await mgr.create("first")
        with pytest.raises(SessionLimitReached):
            await mgr.create("second")
        # After the first is gone, a new session fits again.
        await mgr.delete("first")
        await mgr.create("third")
    finally:
        await r.aclose()


async def test_create_unlimited_by_default() -> None:
    """No cap configured → any number of sessions can be created."""
    r = fakeredis.aioredis.FakeRedis()
    try:
        mgr = SessionManager(r)  # no max_active_sessions
        await mgr.create("a")
        await mgr.create("b")
        await mgr.create("c")
        assert await mgr.exists("c")
    finally:
        await r.aclose()


async def test_touch_does_not_resurrect_concurrently_deleted_session(
    manager: SessionManager,
) -> None:
    """A delete landing between touch()'s GET and SET must not recreate the key.

    Otherwise we leak a ``session:`` key that is no longer in ``sessions:active``
    — a zombie the idle sweep can never reap.
    """
    sid = "racy"
    await manager.create(sid)

    real_get = manager._r.get

    async def get_then_delete(key: str):
        # Read the live value, then simulate a concurrent teardown landing
        # before touch() issues its SET.
        raw = await real_get(key)
        await manager.delete(sid)
        return raw

    with patch.object(manager._r, "get", side_effect=get_then_delete):
        await manager.touch(sid)

    assert not await manager.exists(sid)
    members = await manager._r.smembers(SessionManager.ACTIVE_SET)
    assert all(m.decode() != sid for m in members)


async def test_idle_sessions_returns_old_ones(manager: SessionManager) -> None:
    # Create a session whose last_frame_ts sits far in the past.
    with patch.object(SessionManager, "_now_ms", return_value=0):
        await manager.create("stale")
    with patch.object(SessionManager, "_now_ms", return_value=100_000):
        idle = await manager.idle_sessions(timeout_s=10)
    assert "stale" in idle


async def test_idle_sessions_excludes_recently_touched(manager: SessionManager) -> None:
    with patch.object(SessionManager, "_now_ms", return_value=0):
        await manager.create("fresh")
    with patch.object(SessionManager, "_now_ms", return_value=5_000):
        await manager.touch("fresh")
    with patch.object(SessionManager, "_now_ms", return_value=10_000):
        idle = await manager.idle_sessions(timeout_s=10)
    assert "fresh" not in idle


async def test_concurrent_create_only_one_wins(manager: SessionManager) -> None:
    """Two coroutines racing create() for the same id: exactly one wins."""
    sid = "race"

    async def try_create() -> str:
        try:
            await manager.create(sid)
            return "ok"
        except SessionAlreadyExists:
            return "err"

    results = await asyncio.gather(try_create(), try_create())
    assert results.count("ok") == 1
    assert results.count("err") == 1
