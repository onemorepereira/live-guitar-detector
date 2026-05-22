"""Gateway configuration.

All fields and defaults are the authoritative mirror of DESIGN.md §5.7
(Gateway env-var table). Values are read from environment variables (and an
optional `.env` file) via pydantic-settings.

Field names use SCREAMING_SNAKE_CASE to match the environment variable names
directly — pydantic-settings will read each field from the env var of the
same name when `env_prefix=""`.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Gateway runtime configuration.

    See DESIGN.md §5.7 for the canonical description of every field.
    """

    # Redis
    REDIS_URL: str = "redis://redis:6379/0"

    # Ingest pipeline
    MAX_INGEST_FPS: int = Field(30, ge=1, le=120)
    JPEG_QUALITY: int = Field(75, ge=1, le=100)
    SESSION_IDLE_TIMEOUT_S: int = Field(10, ge=1)

    # Logging
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", env_prefix="")
