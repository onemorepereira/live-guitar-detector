"""Tests for the end-to-end inference pipeline orchestrator (DESIGN.md §5.4).

The pipeline glues :class:`Detector`, :class:`Classifier`, :class:`TrackRegistry`,
and per-track :class:`TrackVote` instances together into a single
``process_frame(frame, frame_no)`` entry point that returns a
:data:`DetectionEvent`-shaped ``dict`` as described in DESIGN.md §5.1.

Tests are split into two layers:

* Unit tests (1-6) use ``unittest.mock.Mock(spec=...)`` for Detector + Classifier
  so the pipeline can be exercised in milliseconds without the OpenVINO models
  on disk. These run unconditionally on every CI run.

* The integration test at the bottom needs both ``requires_model`` (OpenVINO
  IRs on disk) and ``requires_real_fixtures`` (a touched
  ``tests/fixtures/images/REAL.txt`` indicating committed JPEGs are real
  guitar photos). It feeds 20 copies of ``lp_01.jpg`` to the real detector +
  classifier and asserts the track stabilizes onto ``(Gibson, Les Paul)`` by
  frame 15 (the default vote window). This is the closest test to a real-world
  end-to-end run that does not require a Redis/gateway stack.
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from app.classifier import Classifier
from app.config import Settings
from app.detector import Detection, Detector
from app.pipeline import Pipeline
from app.prompts import load_prompts

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "images"
YOLO_MODEL_DIR = Path(__file__).resolve().parents[1] / "app" / "models" / "yolov8n-oiv7-fp32"
MODELS_DIR = Path(__file__).resolve().parents[1] / "app" / "models"
PROMPTS_FILE = Path(__file__).resolve().parents[2].parent / "docs" / "prompts.md"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings() -> Settings:
    """Settings with the worker defaults (no env file lookup)."""
    return Settings(_env_file=None)


def _make_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """A solid-grey BGR uint8 frame — the pipeline never inspects pixel
    content since both the detector and classifier are mocked in these tests."""
    return np.full((height, width, 3), 127, dtype=np.uint8)


def _det(
    *,
    track_id: int | None = 1,
    bbox: tuple[float, float, float, float] = (100.0, 100.0, 200.0, 200.0),
    conf: float = 0.9,
) -> Detection:
    return Detection(
        track_id=track_id,
        bbox_xyxy=bbox,
        confidence=conf,
        class_name="Guitar",
    )


def _mock_detector(return_value: list[Detection]) -> Mock:
    """Mock Detector whose ``detect_and_track`` returns ``return_value``.

    Using ``spec=Detector`` means any call to an attribute the real Detector
    doesn't expose will raise — keeps the pipeline implementation honest about
    the surface it actually uses.
    """
    det = Mock(spec=Detector)
    det.detect_and_track.return_value = return_value
    return det


def _mock_classifier(return_value: dict | None = None) -> Mock:
    """Mock Classifier whose ``classify`` returns ``return_value``.

    Defaults to a high-confidence Gibson Les Paul so vote convergence is the
    common case and individual tests only override when they need to."""
    if return_value is None:
        return_value = {"brand": "Gibson", "model": "Les Paul", "confidence": 0.95}
    clf = Mock(spec=Classifier)
    clf.classify.return_value = return_value
    return clf


# ---------------------------------------------------------------------------
# 1. No detections → empty tracks but a fully-populated event.
# ---------------------------------------------------------------------------


def test_process_frame_with_no_detections_returns_empty_tracks():
    detector = _mock_detector([])
    classifier = _mock_classifier()
    pipeline = Pipeline(detector, classifier, _make_settings())

    event = pipeline.process_frame(_make_frame(), frame_no=0, session_id="s1", frame_ts=42)

    assert event["session_id"] == "s1"
    assert event["frame_id"] == 0
    assert event["frame_ts"] == 42
    assert isinstance(event["inference_ts"], int)
    assert event["inference_ts"] > 0
    assert event["tracks"] == []
    classifier.classify.assert_not_called()


# ---------------------------------------------------------------------------
# 2. Bbox normalization happens in the pipeline (pixels → 0..1).
# ---------------------------------------------------------------------------


def test_process_frame_normalizes_bbox_to_zero_one():
    # 640x480 frame, bbox at pixels (100, 100, 200, 200). The pipeline must
    # divide x's by width and y's by height; that's the only place this
    # normalization happens (Detector still emits pixels per its docstring).
    frame = _make_frame(width=640, height=480)
    detector = _mock_detector([_det(bbox=(100.0, 100.0, 200.0, 200.0))])
    classifier = _mock_classifier()
    pipeline = Pipeline(detector, classifier, _make_settings())

    event = pipeline.process_frame(frame, frame_no=0)

    assert len(event["tracks"]) == 1
    bbox = event["tracks"][0]["bbox"]
    assert bbox[0] == pytest.approx(100 / 640)
    assert bbox[1] == pytest.approx(100 / 480)
    assert bbox[2] == pytest.approx(200 / 640)
    assert bbox[3] == pytest.approx(200 / 480)
    for v in bbox:
        assert 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# 3. Timestamps + frame_id wiring.
# ---------------------------------------------------------------------------


def test_process_frame_sets_timestamps():
    detector = _mock_detector([])
    classifier = _mock_classifier()
    pipeline = Pipeline(detector, classifier, _make_settings())

    before = int(time.time() * 1000)
    event = pipeline.process_frame(_make_frame(), frame_no=17)
    after = int(time.time() * 1000)

    # frame_id must be exactly the frame_no the caller passed; the pipeline
    # does not renumber.
    assert event["frame_id"] == 17
    # frame_ts defaults to "around now" when the caller doesn't pass one
    # (real-time webcam case).
    assert before <= event["frame_ts"] <= after
    # inference_ts is set by the pipeline at emit time, so it must be at
    # least as recent as the start of this call.
    assert before <= event["inference_ts"] <= after
    # inference_ts is set *after* the detector/classifier run — so it is
    # always >= frame_ts (when frame_ts defaults). They can be equal at
    # millisecond resolution on a fast path; allow that.
    assert event["inference_ts"] >= event["frame_ts"]


# ---------------------------------------------------------------------------
# 4. Skipping classification: tiny bbox still emits a track, but no classify().
# ---------------------------------------------------------------------------


def test_skip_classification_emits_track_without_label_update():
    # bbox area 10x10 = 100 px² on a 640x480 frame = 100/307200 ≈ 0.0003 <
    # MIN_BBOX_AREA_FRACTION (0.005). The registry's should_classify must
    # therefore return False — even during warm-up.
    frame = _make_frame(width=640, height=480)
    detector = _mock_detector([_det(bbox=(0.0, 0.0, 10.0, 10.0))])
    classifier = _mock_classifier()
    pipeline = Pipeline(detector, classifier, _make_settings())

    event = pipeline.process_frame(frame, frame_no=0)

    classifier.classify.assert_not_called()
    assert len(event["tracks"]) == 1
    track = event["tracks"][0]
    assert track["track_id"] == 1
    assert track["label"] is None  # vote was never updated
    assert track["stable"] is False
    assert track["age_frames"] == 0


# ---------------------------------------------------------------------------
# 5. Detections with track_id=None are skipped entirely.
# ---------------------------------------------------------------------------


def test_track_id_none_is_skipped():
    """ByteTrack returns track_id=None for unconfirmed warm-up boxes.

    The pipeline elects to drop these entirely rather than emit a TrackDetection
    with no stable id — downstream consumers (the gateway, the frontend) rely
    on track_id as a join key, so a None there would have no business value
    and would only create per-frame ghost tracks in the UI.
    """
    detector = _mock_detector([_det(track_id=None)])
    classifier = _mock_classifier()
    pipeline = Pipeline(detector, classifier, _make_settings())

    event = pipeline.process_frame(_make_frame(), frame_no=0)

    assert event["tracks"] == []
    # And no per-track work was done.
    classifier.classify.assert_not_called()


# ---------------------------------------------------------------------------
# 6. Pruning a stale track drops its vote state.
# ---------------------------------------------------------------------------


def test_pruned_tracks_drop_vote_state():
    """A track unseen for ≥ PRUNE_AFTER_FRAMES (90) must lose its vote state.

    Without this, ``self._votes`` would grow without bound over long sessions
    as ByteTrack assigns new ids to objects re-entering frame.
    """
    detector = _mock_detector([_det(track_id=7)])
    classifier = _mock_classifier()
    pipeline = Pipeline(detector, classifier, _make_settings())

    # Frame 0: observe track 7 (and run classify since it's warm-up + big bbox).
    pipeline.process_frame(_make_frame(), frame_no=0)
    assert 7 in pipeline._votes

    # Frame 100: no detections; prune should sweep track 7 (gap 100 ≥ 90).
    detector.detect_and_track.return_value = []
    pipeline.process_frame(_make_frame(), frame_no=100)
    assert 7 not in pipeline._votes


# ---------------------------------------------------------------------------
# 7. Integration: real models + real fixtures → vote stabilizes by frame 15.
# ---------------------------------------------------------------------------


@pytest.mark.requires_model
@pytest.mark.requires_real_fixtures
def test_track_stabilizes_within_vote_window():
    """Feed 20 frames of the same Les Paul photo; by frame 15 the vote
    must be ``stable=True`` with the right label.

    Frame 15 is the configured ``VOTE_WINDOW``; the rolling deque is full at
    that point and (for a consistent input) the stable flag must flip on.
    """
    frame = cv2.imread(str(FIXTURES_DIR / "lp_01.jpg"), cv2.IMREAD_COLOR)
    assert frame is not None, "fixture lp_01.jpg missing"

    settings = _make_settings()
    detector = Detector(YOLO_MODEL_DIR)
    classifier = Classifier(MODELS_DIR, load_prompts(PROMPTS_FILE), settings.CLIP_INPUT_SIZE)
    pipeline = Pipeline(detector, classifier, settings)

    # Track ids the same object collects across the run. ByteTrack normally
    # converges on a single id within ~3 confirmation frames, but the spec
    # doesn't guarantee one id for the whole loop. Index events by track_id
    # so the assertion is robust to the first id getting dropped and a new
    # one assigned (would be a regression worth investigating, but not this
    # test's bug).
    events_by_track: dict[int, list[dict]] = {}
    for frame_no in range(20):
        event = pipeline.process_frame(frame, frame_no=frame_no)
        for t in event["tracks"]:
            events_by_track.setdefault(t["track_id"], []).append((frame_no, t))

    assert events_by_track, "tracker never produced a confirmed track over 20 frames"

    # The "winning" track is the one with the most observations — that's the
    # stable id ByteTrack settled on for the guitar in the photo.
    winning_id = max(events_by_track, key=lambda k: len(events_by_track[k]))
    appearances = events_by_track[winning_id]

    # By frame 15 (index 15 ≡ the 16th call), the vote window is full and the
    # track should have been stable on at least one sample by then.
    by_frame_15 = [t for (fno, t) in appearances if fno <= 15]
    stable_hits = [t for t in by_frame_15 if t["stable"]]
    assert stable_hits, (
        f"track {winning_id} never reached stable by frame 15; "
        f"appearances up to frame 15: {[(t['label'], t['stable']) for t in by_frame_15]}"
    )

    # And when stable, the label must be the right (brand, model).
    first_stable = stable_hits[0]
    assert first_stable["label"] is not None
    assert first_stable["label"]["brand"] == "Gibson"
    assert first_stable["label"]["model"] == "Les Paul"
