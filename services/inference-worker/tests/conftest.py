"""Shared pytest fixtures for the inference worker test suite."""

from pathlib import Path

import pytest

from app.config import Settings

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def settings() -> Settings:
    """Hermetic Settings instance pointing path defaults at tests/fixtures/.

    Passing `_env_file=None` prevents pydantic-settings from reading the
    developer's local `.env`, so the fixture is reproducible in CI and on dev
    machines alike.
    """
    return Settings(
        MODELS_DIR=FIXTURES_DIR,
        PROMPTS_FILE=FIXTURES_DIR / "prompts.yaml",
        _env_file=None,
    )
