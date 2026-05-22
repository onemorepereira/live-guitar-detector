"""Shared pytest fixtures for the gateway test suite."""

from __future__ import annotations

from collections.abc import AsyncIterator

import fakeredis.aioredis
import httpx
import pytest

from app.main import app as gateway_app


@pytest.fixture
async def fake_redis() -> AsyncIterator[fakeredis.aioredis.FakeRedis]:
    """In-memory Redis stand-in.

    Tests that need a working `redis.ping()` swap ``app.state.redis`` with
    this fixture; teardown closes the client so each test gets a fresh
    instance.
    """
    client = fakeredis.aioredis.FakeRedis(decode_responses=False)
    try:
        yield client
    finally:
        await client.aclose()


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    """ASGI-transport HTTP client driving the FastAPI app in-process.

    Wrapping the app with ``httpx.ASGITransport`` exercises the real
    lifespan handlers (so ``app.state.redis`` is initialised against
    whatever ``REDIS_URL`` is set to) without binding a network socket.
    Tests that want a hermetic Redis should override
    ``gateway_app.state.redis`` with the ``fake_redis`` fixture.
    """
    transport = httpx.ASGITransport(app=gateway_app)
    # Force lifespan startup so app.state.redis exists before tests
    # mutate it; combined with the httpx client context per SIM117.
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://test") as ac,
        gateway_app.router.lifespan_context(gateway_app),
    ):
        yield ac
