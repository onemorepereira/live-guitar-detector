"""Tests for the YOLO + ByteTrack detector wrapper (DESIGN.md §5.4).

Most tests in this module are gated behind two markers so the suite stays
green on machines that don't have the model and/or real fixture photos:

* ``requires_model`` — auto-skipped if the OpenVINO YOLO IR isn't on disk
  (see ``conftest.py``).
* ``requires_real_fixtures`` — auto-skipped unless the developer has touched
  ``tests/fixtures/images/REAL.txt`` to indicate the committed JPEGs are real
  guitar photos and not the synthetic placeholders that ship in the repo.

The pure-pydantic tests at the top of the file run unconditionally — they
exercise the value-object contract without loading any model.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from app.detector import Detection, Detector

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "images"
YOLO_MODEL_DIR = Path(__file__).resolve().parents[1] / "app" / "models" / "yolov8n-oiv7-fp32"


# ---------------------------------------------------------------------------
# Detection pydantic model — no model weights required.
# ---------------------------------------------------------------------------


def test_detection_model_accepts_full_payload():
    d = Detection(
        track_id=7,
        bbox_xyxy=(10.0, 20.0, 100.0, 200.0),
        confidence=0.91,
        class_name="Guitar",
    )
    assert d.track_id == 7
    assert d.bbox_xyxy == (10.0, 20.0, 100.0, 200.0)
    assert d.confidence == pytest.approx(0.91)
    assert d.class_name == "Guitar"


def test_detection_model_allows_none_track_id():
    """Untracked single-frame detections set track_id=None."""
    d = Detection(
        track_id=None,
        bbox_xyxy=(0.0, 0.0, 10.0, 10.0),
        confidence=0.5,
        class_name="Guitar",
    )
    assert d.track_id is None


def test_detection_model_is_frozen():
    d = Detection(
        track_id=None,
        bbox_xyxy=(0.0, 0.0, 1.0, 1.0),
        confidence=0.5,
        class_name="Guitar",
    )
    with pytest.raises(ValidationError):
        d.confidence = 0.9  # type: ignore[misc]


def test_detection_model_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        Detection(
            track_id=None,
            bbox_xyxy=(0.0, 0.0, 1.0, 1.0),
            confidence=0.5,
            class_name="Guitar",
            extra="nope",  # type: ignore[call-arg]
        )


def test_detection_model_rejects_negative_confidence():
    with pytest.raises(ValidationError):
        Detection(
            track_id=None,
            bbox_xyxy=(0.0, 0.0, 1.0, 1.0),
            confidence=-0.1,
            class_name="Guitar",
        )


def test_detection_model_rejects_confidence_above_one():
    with pytest.raises(ValidationError):
        Detection(
            track_id=None,
            bbox_xyxy=(0.0, 0.0, 1.0, 1.0),
            confidence=1.5,
            class_name="Guitar",
        )


def test_detection_model_rejects_bbox_with_wrong_arity():
    with pytest.raises(ValidationError):
        Detection(
            track_id=None,
            bbox_xyxy=(0.0, 0.0, 1.0),  # type: ignore[arg-type]
            confidence=0.5,
            class_name="Guitar",
        )


def test_detection_model_rejects_empty_class_name():
    with pytest.raises(ValidationError):
        Detection(
            track_id=None,
            bbox_xyxy=(0.0, 0.0, 1.0, 1.0),
            confidence=0.5,
            class_name="",
        )


# ---------------------------------------------------------------------------
# Detector construction — requires the model on disk, but no real fixtures.
# ---------------------------------------------------------------------------


@pytest.mark.requires_model
def test_detector_constructs_with_default_settings():
    """Load the OpenVINO IR and verify the Guitar class is registered."""
    det = Detector(YOLO_MODEL_DIR)
    # The wrapper must learn which class id maps to "Guitar" at load time.
    # Exposing it as a public attribute keeps the test cheap (no introspection
    # of model.names) and lets downstream tooling assert against it.
    assert det.guitar_class_id is not None
    assert det.conf == pytest.approx(0.35)
    assert det.iou == pytest.approx(0.5)
    assert det.imgsz == 416


@pytest.mark.requires_model
def test_detector_constructs_with_custom_settings():
    det = Detector(YOLO_MODEL_DIR, conf=0.5, iou=0.4, imgsz=320)
    assert det.conf == pytest.approx(0.5)
    assert det.iou == pytest.approx(0.4)
    assert det.imgsz == 320


@pytest.mark.requires_model
def test_detector_raises_if_guitar_class_missing(monkeypatch):
    """Fail-fast (DESIGN.md §5.4) when loaded weights don't have a Guitar class.

    We let the real loader run to produce a real model, then monkeypatch the
    ``names`` dict to simulate the wrong-weights scenario. Construction must
    raise ``RuntimeError`` — silently picking class id 0 would let bad
    weights ship to prod and produce garbage detections.
    """
    import app.detector as detector_mod

    real_yolo = detector_mod.YOLO

    class _FakeYOLO:
        def __init__(self, *args, **kwargs):
            self._inner = real_yolo(*args, **kwargs)
            # Replace the class table with something that doesn't have Guitar.
            self.names = {0: "Person", 1: "Car"}

        def __getattr__(self, item):
            return getattr(self._inner, item)

    monkeypatch.setattr(detector_mod, "YOLO", _FakeYOLO)

    with pytest.raises(RuntimeError, match="Guitar"):
        Detector(YOLO_MODEL_DIR)


# ---------------------------------------------------------------------------
# Detection behaviour — needs both the model AND real photographs.
# ---------------------------------------------------------------------------


def _load_fixture(name: str) -> np.ndarray:
    """Read a fixture JPEG as a BGR uint8 numpy array, matching OpenCV's
    `cv2.imdecode` convention used everywhere else in the worker."""
    path = FIXTURES_DIR / name
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert img is not None, f"failed to read fixture {path}"
    return img


@pytest.mark.requires_model
@pytest.mark.requires_real_fixtures
def test_detect_returns_empty_for_image_without_guitar():
    det = Detector(YOLO_MODEL_DIR)
    frame = _load_fixture("street_scene.jpg")
    detections = det.detect(frame)
    assert detections == []


@pytest.mark.requires_model
@pytest.mark.requires_real_fixtures
def test_detect_returns_guitar_detection_for_clear_photo():
    det = Detector(YOLO_MODEL_DIR)
    frame = _load_fixture("lp_01.jpg")
    detections = det.detect(frame)
    assert len(detections) >= 1
    for d in detections:
        assert isinstance(d, Detection)
        assert d.class_name == "Guitar"
        assert d.confidence >= det.conf
        assert d.track_id is None  # no tracking in plain detect()


@pytest.mark.requires_model
@pytest.mark.requires_real_fixtures
def test_detect_bbox_within_frame_bounds():
    det = Detector(YOLO_MODEL_DIR)
    frame = _load_fixture("lp_01.jpg")
    h, w = frame.shape[:2]
    detections = det.detect(frame)
    assert detections, "fixture expected to produce at least one detection"
    for d in detections:
        x1, y1, x2, y2 = d.bbox_xyxy
        assert 0 <= x1 < x2 <= w, f"x bounds violated: ({x1}, {x2}) for width {w}"
        assert 0 <= y1 < y2 <= h, f"y bounds violated: ({y1}, {y2}) for height {h}"


@pytest.mark.requires_model
@pytest.mark.requires_real_fixtures
def test_detect_and_track_assigns_persistent_track_ids():
    """Feed the same image 10 times; ByteTrack should reuse at least one id.

    ByteTrack assigns integer ids starting from 1 once a detection is
    confirmed (typically within ~3 frames). With ``persist=True`` the
    tracker state survives across calls on the same Detector instance, so
    a stationary object's id should stabilize and repeat.
    """
    det = Detector(YOLO_MODEL_DIR)
    frame = _load_fixture("lp_01.jpg")

    ids_per_call: list[set[int]] = []
    for _ in range(10):
        dets = det.detect_and_track(frame)
        ids = {d.track_id for d in dets if d.track_id is not None}
        ids_per_call.append(ids)

    # Drop the warm-up calls before ByteTrack confirms the track.
    confirmed = [s for s in ids_per_call if s]
    assert confirmed, "tracker never produced a confirmed track id over 10 frames"

    persistent = set.intersection(*confirmed)
    assert (
        persistent
    ), f"no track id persisted across confirmed calls; got per-call sets: {confirmed}"


@pytest.mark.requires_model
@pytest.mark.requires_real_fixtures
def test_detect_returns_list_of_pydantic_detection_objects():
    """Concrete type check on the live model path."""
    det = Detector(YOLO_MODEL_DIR)
    frame = _load_fixture("lp_01.jpg")
    detections = det.detect(frame)
    assert isinstance(detections, list)
    for d in detections:
        assert isinstance(d, Detection)
