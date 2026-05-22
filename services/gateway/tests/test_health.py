"""Health endpoint tests for the gateway."""

from __future__ import annotations

from unittest.mock import AsyncMock

import fakeredis.aioredis
import httpx

from app.main import app as gateway_app


async def test_healthz_returns_200(client: httpx.AsyncClient) -> None:
    """/healthz is a pure liveness probe — no dependencies required."""
    response = await client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"ok": True}


async def test_readyz_returns_200_when_redis_reachable(
    client: httpx.AsyncClient,
    fake_redis: fakeredis.aioredis.FakeRedis,
) -> None:
    """/readyz reports ready when Redis ping succeeds."""
    gateway_app.state.redis = fake_redis
    response = await client.get("/readyz")
    assert response.status_code == 200
    assert response.json() == {"ok": True}


async def test_readyz_returns_503_when_redis_unreachable(
    client: httpx.AsyncClient,
) -> None:
    """/readyz reports 503 + error string when Redis ping raises."""
    broken = AsyncMock()
    broken.ping = AsyncMock(side_effect=ConnectionError("redis down"))
    gateway_app.state.redis = broken

    response = await client.get("/readyz")
    assert response.status_code == 503
    body = response.json()
    assert body["ok"] is False
    assert "redis down" in body["error"]
