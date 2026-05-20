"""Tests for the MobileCLIP zero-shot classifier wrapper (DESIGN.md §5.4).

The classifier loads two OpenVINO IRs (image tower + text tower), precomputes
text features at construction time, and returns a ``{brand, model, confidence}``
dict for each incoming BGR image.

Test stratification mirrors ``test_detector.py``:

* ``requires_model`` — the OpenVINO IRs are large; skip if absent.
* ``requires_real_fixtures`` — the accuracy tests need real photographs;
  the committed synthetic placeholders will not classify meaningfully.

Tests 1-3 (shape / contract tests) run against synthetic images because they
only assert the *output shape* of ``classify()`` — not the correctness of any
specific label. Tests 4-5 (accuracy + acoustic rejection) require real photos
and will auto-skip on a fresh clone.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from app.classifier import Classifier
from app.prompts import load_prompts

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "images"
MODELS_DIR = Path(__file__).resolve().parents[1] / "app" / "models"
PROMPTS_FILE = Path(__file__).resolve().parents[2].parent / "docs" / "prompts.md"


def _load_fixture(name: str) -> np.ndarray:
    """Read a fixture JPEG as a BGR uint8 numpy array."""
    path = FIXTURES_DIR / name
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert img is not None, f"failed to read fixture {path}"
    return img


def _prompts():
    return load_prompts(PROMPTS_FILE)


# ---------------------------------------------------------------------------
# Shape / contract tests — need the model but not real fixtures.
# ---------------------------------------------------------------------------


@pytest.mark.requires_model
def test_classifier_constructs_and_precomputes_text_features():
    """Constructor must load both towers and cache an (N, D) text-feature tensor.

    The cache is exposed as ``_text_features`` (name-mangling-friendly single
    leading underscore) so test code can assert on its shape without re-running
    the text tower. DESIGN.md §5.4 mandates that text features are computed
    once at startup, never per-frame.
    """
    prompts = _prompts()
    clf = Classifier(MODELS_DIR, prompts, input_size=224)

    assert hasattr(clf, "_text_features"), "text features cache missing"
    feats = clf._text_features
    assert isinstance(feats, np.ndarray), f"expected ndarray, got {type(feats)}"
    assert feats.ndim == 2, f"expected 2-D (N, D), got shape {feats.shape}"
    assert feats.shape[0] == len(
        prompts
    ), f"text-feature row count {feats.shape[0]} != prompt count {len(prompts)}"
    assert feats.shape[1] > 0, "feature dimension must be positive"

    # Cosine similarity requires unit-length features; assert each row's
    # L2 norm is ~1.0. Use a generous tolerance because the IR is FP16.
    norms = np.linalg.norm(feats, axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-2, atol=1e-2)


@pytest.mark.requires_model
def test_classify_returns_well_formed_result():
    """classify() returns a dict with the documented schema on any image.

    Uses a synthetic fixture so this test does not need real photos. We only
    check the output shape and types — not that the predicted label is the
    "right" one for the placeholder content.
    """
    prompts = _prompts()
    clf = Classifier(MODELS_DIR, prompts, input_size=224)

    img = _load_fixture("strat_01.jpg")
    result = clf.classify(img)

    assert isinstance(result, dict)
    assert set(result.keys()) == {"brand", "model", "confidence"}
    assert isinstance(result["brand"], str) and result["brand"], "brand empty"
    assert isinstance(result["model"], str) and result["model"], "model empty"
    assert isinstance(result["confidence"], float)
    assert 0.0 <= result["confidence"] <= 1.0, f"confidence {result['confidence']} out of [0, 1]"


@pytest.mark.requires_model
def test_classify_returns_known_brand_or_unknown():
    """The predicted brand must be one of the labels declared in prompts.md.

    There are only three distinct brand values in the prompts file
    (``Gibson``, ``Fender``, ``Unknown``); whatever the image happens to be,
    the classifier must land on one of them.
    """
    prompts = _prompts()
    clf = Classifier(MODELS_DIR, prompts, input_size=224)

    img = _load_fixture("strat_01.jpg")
    result = clf.classify(img)

    valid_brands = {p.brand for p in prompts}
    assert (
        result["brand"] in valid_brands
    ), f"brand {result['brand']!r} not in declared prompt brands {valid_brands}"


# ---------------------------------------------------------------------------
# Accuracy tests — need both the model AND real fixture photographs.
# ---------------------------------------------------------------------------


# The fixtures the test_detector suite documents: 2x Strat, 2x Les Paul.
# DESIGN.md §5.4 mandates ≥5/6 on six fixtures; we only have 4 brand+model
# images committed today, so we relax to "majority correct" — 3/4 — while
# still asserting on real photos when REAL.txt is present.
_BRAND_MODEL_FIXTURES = [
    ("strat_01.jpg", "Fender", "Stratocaster"),
    ("strat_02.jpg", "Fender", "Stratocaster"),
    ("lp_01.jpg", "Gibson", "Les Paul"),
    ("lp_02.jpg", "Gibson", "Les Paul"),
]


@pytest.mark.requires_model
@pytest.mark.requires_real_fixtures
def test_classify_accuracy_on_brand_model_fixtures():
    """Top-1 (brand, model) must match on the majority of real fixture photos.

    With 4 fixtures we require ≥ 3 correct. If/when more real fixtures land
    (Explorer, Flying V, SG, Telecaster), bump ``_BRAND_MODEL_FIXTURES`` and
    raise the threshold to ≥ 5/6 to match the DESIGN.md target exactly.
    """
    prompts = _prompts()
    clf = Classifier(MODELS_DIR, prompts, input_size=224)

    correct = 0
    failures: list[str] = []
    for name, expected_brand, expected_model in _BRAND_MODEL_FIXTURES:
        img = _load_fixture(name)
        result = clf.classify(img)
        if result["brand"] == expected_brand and result["model"] == expected_model:
            correct += 1
        else:
            failures.append(
                f"{name}: expected ({expected_brand}, {expected_model}), "
                f"got ({result['brand']}, {result['model']}) @ {result['confidence']:.3f}"
            )

    threshold = 3  # ≥ 3 of 4; majority-correct, leaves room for one miss.
    assert correct >= threshold, (
        f"classifier accuracy {correct}/{len(_BRAND_MODEL_FIXTURES)} below "
        f"threshold {threshold}/{len(_BRAND_MODEL_FIXTURES)}. Failures:\n  " + "\n  ".join(failures)
    )


@pytest.mark.requires_model
@pytest.mark.requires_real_fixtures
def test_classify_acoustic_is_unknown():
    """An acoustic-guitar photo must hit a rejection prompt (brand or model == 'Unknown').

    The prompts file includes ``"a photograph of an acoustic guitar" ->
    (Unknown, Unknown)`` precisely so that the six known brand/model
    combinations don't claim acoustic photos. If this test fails, the rejection
    prompt's score is being beaten by one of the electric-guitar prompts —
    iterate on the wording in ``docs/prompts.md`` rather than the classifier.
    """
    prompts = _prompts()
    clf = Classifier(MODELS_DIR, prompts, input_size=224)

    img = _load_fixture("acoustic_01.jpg")
    result = clf.classify(img)
    assert result["brand"] == "Unknown" or result["model"] == "Unknown", (
        f"expected acoustic to be Unknown, got "
        f"({result['brand']}, {result['model']}) @ {result['confidence']:.3f}"
    )
