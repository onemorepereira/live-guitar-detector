"""Tests for the MobileCLIP linear-probe classifier.

The probe wraps two artifacts:
  - the MobileCLIP image tower IR (shared with the zero-shot classifier)
  - a small `.npz` head produced by ``scripts/train_probe.py``

The shape/contract tests below build a *random* probe head with the
correct dimensionality so they can run without a real training set —
they verify the loader, the inference plumbing, and the
``{brand, model, confidence}`` output schema.

All tests are gated on ``requires_model`` because constructing the
classifier needs the OpenVINO image tower on disk.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.probe_classifier import ProbeClassifier

MODELS_DIR = Path(__file__).resolve().parents[1] / "app" / "models"

_LABELS = [
    ("Gibson", "Les Paul"),
    ("Gibson", "SG"),
    ("Gibson", "Explorer"),
    ("Gibson", "Flying V"),
    ("Fender", "Stratocaster"),
    ("Fender", "Telecaster"),
    ("Unknown", "Unknown"),
]


def _make_random_probe(path: Path, n_classes: int = 7, feature_dim: int = 512) -> None:
    """Write a probe artifact with random weights for shape testing."""
    rng = np.random.default_rng(0)
    W = rng.standard_normal((n_classes, feature_dim)).astype(np.float32)
    b = rng.standard_normal((n_classes,)).astype(np.float32)
    labels = np.array([list(pair) for pair in _LABELS[:n_classes]], dtype="<U64")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, W=W, b=b, labels=labels)


def _solid_image(width: int = 320, height: int = 240) -> np.ndarray:
    """A flat mid-grey BGR uint8 image — valid input for the preprocessor."""
    return np.full((height, width, 3), 128, dtype=np.uint8)


@pytest.mark.requires_model
def test_probe_classifier_loads_random_head(tmp_path: Path) -> None:
    probe = tmp_path / "probe.npz"
    _make_random_probe(probe)
    clf = ProbeClassifier(MODELS_DIR, probe, input_size=224)
    assert clf.labels == _LABELS
    assert clf.feature_dim == 512


@pytest.mark.requires_model
def test_probe_classify_returns_well_formed_result(tmp_path: Path) -> None:
    probe = tmp_path / "probe.npz"
    _make_random_probe(probe)
    clf = ProbeClassifier(MODELS_DIR, probe, input_size=224)

    result = clf.classify(_solid_image())

    assert isinstance(result, dict)
    assert set(result.keys()) == {"brand", "model", "confidence"}
    assert isinstance(result["brand"], str) and result["brand"]
    assert isinstance(result["model"], str) and result["model"]
    assert 0.0 <= result["confidence"] <= 1.0
    assert (result["brand"], result["model"]) in _LABELS


@pytest.mark.requires_model
def test_probe_rejects_non_uint8_image(tmp_path: Path) -> None:
    probe = tmp_path / "probe.npz"
    _make_random_probe(probe)
    clf = ProbeClassifier(MODELS_DIR, probe, input_size=224)

    bad = np.full((240, 320, 3), 0.5, dtype=np.float32)
    with pytest.raises(ValueError, match="uint8"):
        clf.classify(bad)


@pytest.mark.requires_model
def test_probe_handles_non_square_via_padding(tmp_path: Path) -> None:
    """Tall-and-narrow input should classify without raising (square-pad path)."""
    probe = tmp_path / "probe.npz"
    _make_random_probe(probe)
    clf = ProbeClassifier(MODELS_DIR, probe, input_size=224)

    tall = _solid_image(width=120, height=480)
    result = clf.classify(tall)
    assert (result["brand"], result["model"]) in _LABELS


@pytest.mark.requires_model
def test_probe_rejects_shape_mismatch_in_artifact(tmp_path: Path) -> None:
    """Loader complains if W/b/labels disagree on row count."""
    probe = tmp_path / "bad_probe.npz"
    W = np.zeros((7, 512), dtype=np.float32)
    b = np.zeros((6,), dtype=np.float32)  # off-by-one
    labels = np.array([list(p) for p in _LABELS], dtype="<U64")
    np.savez(probe, W=W, b=b, labels=labels)

    with pytest.raises(ValueError, match="probe bias shape"):
        ProbeClassifier(MODELS_DIR, probe, input_size=224)


def test_probe_missing_artifact_raises_file_not_found(tmp_path: Path) -> None:
    """Pre-tower check — runs without the model, fails fast on missing probe."""
    with pytest.raises(FileNotFoundError, match="image tower IR"):
        ProbeClassifier(
            tmp_path / "no-such-models",
            tmp_path / "no-such-probe.npz",
            input_size=224,
        )
