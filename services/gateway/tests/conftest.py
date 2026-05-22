"""Shared pytest fixtures for the gateway test suite."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

import fakeredis.aioredis
import httpx
import pytest

from app.main import app as gateway_app


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests gated on the ``requires_aiortc_peer`` marker.

    Spinning up a real aiortc peer pair pulls in libsrtp/libvpx and can be
    flaky in minimal/CI environments. For Phase 1 we always skip unless the
    developer opts in via ``RUN_AIORTC_TESTS=1``; Phase 4 K3s smoke tests
    will exercise the real path. The env-var gate keeps the opt-in cheap
    (no marker file to maintain).
    """
    if os.environ.get("RUN_AIORTC_TESTS") == "1":
        return
    skip_aiortc = pytest.mark.skip(
        reason="set RUN_AIORTC_TESTS=1 to enable aiortc peer integration tests"
    )
    for item in items:
        if "requires_aiortc_peer" in item.keywords:
            item.add_marker(skip_aiortc)


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
