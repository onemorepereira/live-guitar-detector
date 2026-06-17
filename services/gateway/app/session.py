"""Session lifecycle for the gateway.

A "session" is one browser tab's connection: WebRTC peer + WebSocket +
Redis streams. This module owns:

- ``session:{id}``     — JSON-blob key with ``created_ts`` + ``last_frame_ts``,
                         60s sliding TTL (refreshed via :meth:`touch`).
- ``sessions:active``  — Redis Set of currently-active session IDs.
- ``frames:{id}``      — Stream stub; populated by Task 2.3 (XADD MAXLEN ~30).
- ``detections:{id}``  — Stream stub; populated by Task 2.3 (XADD MAXLEN ~100).

Stream creation note: streams are NOT pre-created here. The first ``XADD``
naturally creates the stream key, so explicit stream initialization would
require a dummy entry + trim. We sidestep that — the only stream lifecycle
this module owns is teardown (:meth:`delete` removes them).

``create()`` is atomic via ``SET ... NX EX 60``. Two coroutines racing on
the same ``session_id``: exactly one wins; the other gets
:class:`SessionAlreadyExists`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import redis.asyncio as redis_async


class SessionAlreadyExists(Exception):
    """Raised when :meth:`SessionManager.create` is called for an existing id."""


class SessionLimitReached(Exception):
    """Raised when :meth:`SessionManager.create` would exceed the active cap.

    The inference worker shares one ByteTrack tracker (and vote registry)
    across all sessions, so two concurrent sessions interleave frames into the
    same tracker and corrupt each other's track ids. Until that's isolated
    per-session, the gateway caps the number of simultaneously-active sessions.
    """


@dataclass(frozen=True)
class SessionState:
    """Immutable snapshot returned by :meth:`SessionManager.create`."""

    session_id: str
    created_ts: int  # unix milliseconds
    last_frame_ts: int  # unix milliseconds


class SessionManager:
    """Manages session metadata in Redis. Async-only."""

    SESSION_KEY = "session:{session_id}"
    ACTIVE_SET = "sessions:active"
    SESSION_TTL_S = 60  # sliding TTL refreshed by touch()

    def __init__(self, r: redis_async.Redis, max_active_sessions: int | None = None) -> None:
        self._r = r
        # ``None`` = unlimited. When set, :meth:`create` rejects a new session
        # once ``sessions:active`` is already at the cap.
        self._max_active_sessions = max_active_sessions

    @classmethod
    def _now_ms(cls) -> int:
        return int(time.time() * 1000)

    @classmethod
    def _key(cls, session_id: str) -> str:
        return cls.SESSION_KEY.format(session_id=session_id)

    async def create(self, session_id: str) -> SessionState:
        """Create a session. Atomic via ``SET NX``.

        Raises :class:`SessionAlreadyExists` if the id is already live. We use
        a single JSON-blob value (under one key) rather than ``HSET`` so the
        whole metadata write is atomic in one round-trip. Streams are not
        pre-created — the first ``XADD`` (Task 2.3) will create them.
        """
        if self._max_active_sessions is not None:
            # Best-effort cap check. The SCARD→SET window is racy under truly
            # concurrent creates, but the realistic deployment is single-user;
            # a rare over-admit is harmless next to the cost of a Lua/WATCH
            # transaction. SET NX below still guarantees per-id atomicity.
            active = await self._r.scard(self.ACTIVE_SET)
            if active >= self._max_active_sessions:
                raise SessionLimitReached(
                    f"active session limit reached ({self._max_active_sessions})"
                )
        now = self._now_ms()
        payload = json.dumps({"created_ts": now, "last_frame_ts": now})
        ok = await self._r.set(
            self._key(session_id),
            payload,
            nx=True,
            ex=self.SESSION_TTL_S,
        )
        if not ok:
            raise SessionAlreadyExists(session_id)
        await self._r.sadd(self.ACTIVE_SET, session_id)
        return SessionState(session_id=session_id, created_ts=now, last_frame_ts=now)

    async def delete(self, session_id: str) -> None:
        """Remove session metadata, both stream keys, and the active-set entry."""
        await self._r.delete(
            self._key(session_id),
            f"frames:{session_id}",
            f"detections:{session_id}",
        )
        await self._r.srem(self.ACTIVE_SET, session_id)

    async def touch(self, session_id: str) -> None:
        """Bump ``last_frame_ts`` on this session; resets the 60s TTL.

        No-op if the session metadata has already expired or been deleted —
        the caller has nothing to recover, and we avoid resurrecting a key
        that another coroutine might have just torn down.
        """
        key = self._key(session_id)
        raw = await self._r.get(key)
        if raw is None:
            return
        data = json.loads(raw)
        data["last_frame_ts"] = self._now_ms()
        # ``xx=True``: only refresh if the key still exists. Without it, a
        # delete landing between the GET above and this SET would recreate the
        # ``session:`` key with a fresh TTL — orphaned from ``sessions:active``
        # and unreapable by the idle sweep.
        await self._r.set(key, json.dumps(data), xx=True, ex=self.SESSION_TTL_S)

    async def idle_sessions(self, timeout_s: int) -> list[str]:
        """Return IDs whose ``last_frame_ts`` is at least ``timeout_s`` old.

        Sessions whose metadata key has expired (the natural 60s TTL elapsed
        without a :meth:`touch`) are also reported — they're definitively idle
        and need active-set cleanup.
        """
        now = self._now_ms()
        threshold_ms = timeout_s * 1000
        out: list[str] = []
        active = await self._r.smembers(self.ACTIVE_SET)
        for sid_raw in active:
            sid = sid_raw.decode() if isinstance(sid_raw, bytes) else sid_raw
            raw = await self._r.get(self._key(sid))
            if raw is None:
                out.append(sid)
                continue
            data = json.loads(raw)
            if now - int(data.get("last_frame_ts", 0)) >= threshold_ms:
                out.append(sid)
        return out

    async def exists(self, session_id: str) -> bool:
        """Whether the session metadata key is still live in Redis."""
        return await self._r.exists(self._key(session_id)) > 0
