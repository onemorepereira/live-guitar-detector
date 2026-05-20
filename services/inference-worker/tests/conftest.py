"""Shared pytest fixtures and collection hooks for the inference worker test suite."""

from pathlib import Path

import pytest

from app.config import Settings

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Repo paths used by collection hooks. The worker package lives at
# ``services/inference-worker/`` so ``parents[1]`` from this file is the
# worker root and ``parents[1] / "app" / "models"`` is where
# ``scripts/download_models.py`` writes its OpenVINO IR output.
_WORKER_ROOT = Path(__file__).resolve().parents[1]
_MODELS_ROOT = _WORKER_ROOT / "app" / "models"
_YOLO_IR_XML = _MODELS_ROOT / "yolov8n-oiv7-fp32" / "yolov8n-oiv7.xml"
_CLIP_IMAGE_IR_XML = _MODELS_ROOT / "mobileclip-image-fp16" / "image.xml"
_CLIP_TEXT_IR_XML = _MODELS_ROOT / "mobileclip-text-fp16" / "text.xml"
_REQUIRED_IRS = (_YOLO_IR_XML, _CLIP_IMAGE_IR_XML, _CLIP_TEXT_IR_XML)

# Marker file the developer touches after dropping real fixture JPEGs into
# ``tests/fixtures/images/``. Cheap, env-var-free, and explicit: opting in
# requires a deliberate ``touch``, so synthetic-placeholder runs stay green.
_REAL_FIXTURES_MARKER = FIXTURES_DIR / "images" / "REAL.txt"


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


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests that need optional, large, or human-curated assets.

    Two markers are handled here:

    * ``requires_model`` — skipped unless the YOLO + MobileCLIP OpenVINO IRs
      all exist on disk. The runtime cost of downloading + exporting the
      models (see ``scripts/download_models.py``) is too high to require
      unconditionally. We gate the marker on all three IRs together because
      they ship from the same ``download_models.py all`` invocation; partial
      installs are a developer-error case, not a normal mode.

    * ``requires_real_fixtures`` — skipped unless
      ``tests/fixtures/images/REAL.txt`` exists. The committed fixture JPEGs
      are synthetic placeholders that the detector will (correctly) not fire
      on; meaningful end-to-end assertions need real guitar photos which
      can't be redistributed in the repo. The developer signals "I've
      replaced the placeholders" by touching the marker file.
    """
    skip_model = None
    missing = [p for p in _REQUIRED_IRS if not p.exists()]
    if missing:
        missing_list = ", ".join(str(p) for p in missing)
        skip_model = pytest.mark.skip(
            reason=(
                f"Required model IR(s) not found: {missing_list}; "
                "run scripts/download_models.py all"
            )
        )

    skip_real = None
    if not _REAL_FIXTURES_MARKER.exists():
        skip_real = pytest.mark.skip(
            reason=(
                "No real fixture images present; replace synthetic placeholders in "
                "tests/fixtures/images/ and `touch tests/fixtures/images/REAL.txt`"
            )
        )

    for item in items:
        if skip_model is not None and "requires_model" in item.keywords:
            item.add_marker(skip_model)
        if skip_real is not None and "requires_real_fixtures" in item.keywords:
            item.add_marker(skip_real)
