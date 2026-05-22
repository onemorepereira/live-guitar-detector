"""Gateway FastAPI app — Phase 2.1 skeleton.

Currently only health endpoints are wired; session/WebRTC/WebSocket
endpoints land in tasks 2.2-2.6.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import redis.asyncio as redis_async
from fastapi import FastAPI, Response
from loguru import logger

from app.config import Settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise/teardown gateway-wide resources.

    A single async Redis client is created at startup and closed at
    shutdown. `decode_responses=False` keeps the client byte-clean so
    later phases can stream raw JPEG bytes through Redis without
    surprise utf-8 decoding.
    """
    settings = Settings()
    app.state.settings = settings
    app.state.redis = redis_async.from_url(settings.REDIS_URL, decode_responses=False)
    logger.info("gateway starting; redis={}", settings.REDIS_URL)
    try:
        yield
    finally:
        await app.state.redis.aclose()
        logger.info("gateway shutting down")


app = FastAPI(lifespan=lifespan)


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
