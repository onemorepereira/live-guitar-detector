"""Wire models for the gateway HTTP API.

Mirrors DESIGN.md §5.1 / §5.2 request/response shapes. Pydantic v2 with
strict field constraints — invalid bodies surface as 422 from FastAPI.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, StringConstraints

# Session ids are client-chosen (typically a UUID); 256 chars is a generous
# bound that rejects abuse while accommodating any sane identifier scheme.
SessionId = Annotated[str, StringConstraints(min_length=1, max_length=256, strip_whitespace=True)]

# Real WebRTC offers are a few KB even with a full ICE candidate list. Cap the
# body at 64 KB so the unauthenticated offer endpoint can't hand an arbitrarily
# large blob to aiortc's SDP parser.
Sdp = Annotated[str, StringConstraints(min_length=1, max_length=64_000)]


class SessionCreateRequest(BaseModel):
    """Body for ``POST /api/session``."""

    model_config = ConfigDict(extra="forbid")
    session_id: SessionId


class SessionCreateResponse(BaseModel):
    """200 response body for ``POST /api/session``."""

    ok: bool = True


class WebRTCOfferRequest(BaseModel):
    """Body for ``POST /api/webrtc/offer``."""

    model_config = ConfigDict(extra="forbid")
    session_id: SessionId
    sdp: Sdp
    type: Literal["offer"]


class WebRTCAnswerResponse(BaseModel):
    """200 response body for ``POST /api/webrtc/offer``."""

    sdp: str
    type: Literal["answer"]
