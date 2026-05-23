"""Wire models for the gateway HTTP API.

Mirrors DESIGN.md §5.1 / §5.2 request/response shapes. Pydantic v2 with
strict field constraints — invalid bodies surface as 422 from FastAPI.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, StringConstraints

NonEmptyStr = Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]


class SessionCreateRequest(BaseModel):
    """Body for ``POST /api/session``."""

    model_config = ConfigDict(extra="forbid")
    session_id: NonEmptyStr


class SessionCreateResponse(BaseModel):
    """200 response body for ``POST /api/session``."""

    ok: bool = True


class WebRTCOfferRequest(BaseModel):
    """Body for ``POST /api/webrtc/offer``."""

    model_config = ConfigDict(extra="forbid")
    session_id: NonEmptyStr
    sdp: NonEmptyStr
    type: Literal["offer"]


class WebRTCAnswerResponse(BaseModel):
    """200 response body for ``POST /api/webrtc/offer``."""

    sdp: str
    type: Literal["answer"]
