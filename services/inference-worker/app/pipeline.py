"""End-to-end inference pipeline orchestrator (DESIGN.md §5.4).

Glues four single-responsibility modules into one ``process_frame`` entry
point:

* :class:`app.detector.Detector` — YOLO + ByteTrack, returns pixel-space boxes.
* :class:`app.classifier.Classifier` — zero-shot MobileCLIP brand/model labels.
* :class:`app.tracks.TrackRegistry` — first/last-seen bookkeeping and the
  "should I run CLIP this frame?" policy from §5.4.
* :class:`app.voting.TrackVote` — per-track rolling-window weighted vote.

The orchestrator is the only place that:

1. Normalizes detector bboxes from pixel coordinates to ``[0, 1]`` for emission
   (the detector deliberately stays in pixel space so it can crop without
   re-multiplying by frame size — see ``Detector.bbox_xyxy`` docstring).
2. Owns the ``dict[track_id, TrackVote]`` lifecycle: lazily allocate per
   track, drop in lock-step with ``TrackRegistry.prune``.
3. Stamps ``inference_ts`` onto the outgoing event.

Phase 1 has no Redis loop yet — the gateway/Redis wiring lives in a later
task. For now, ``session_id`` and ``frame_ts`` are caller-supplied arguments
to ``process_frame`` so the same orchestrator works for a webcam-driven smoke
test today and a Redis-driven worker loop later.
"""

from __future__ import annotations

import time
from collections import Counter
from typing import TypedDict

import numpy as np

from app.classifier import Classifier
from app.config import Settings
from app.detector import Detection, Detector
from app.tracks import TrackRegistry
from app.voting import SmoothedLabel, TrackVote


class ClassificationLabelDict(TypedDict):
    """Wire form of DESIGN.md §5.1 ``ClassificationLabel``.

    Same field names as the TypeScript counterpart so JSON encode/decode is
    a pass-through on both ends.
    """

    brand: str
    model: str
    confidence: float


class TrackDetectionDict(TypedDict):
    """Wire form of DESIGN.md §5.1 ``TrackDetection``.

    ``label`` is ``None`` until the per-track vote produces a non-Unknown
    winner. ``stable`` flips on once the rolling vote window is full and the
    smoothed confidence clears the threshold; consumers should treat
    ``stable=False`` labels as preview only.
    """

    track_id: int
    bbox: tuple[float, float, float, float]
    detection_confidence: float
    label: ClassificationLabelDict | None
    stable: bool
    age_frames: int


class DetectionEventDict(TypedDict):
    """Wire form of DESIGN.md §5.1 ``DetectionEvent``."""

    session_id: str
    frame_id: int
    frame_ts: int
    inference_ts: int
    tracks: list[TrackDetectionDict]


def _smoothed_to_label(s: SmoothedLabel) -> ClassificationLabelDict | None:
    """Convert a :class:`SmoothedLabel` into the API-facing ``ClassificationLabel``.

    ``SmoothedLabel.label`` is either ``None`` (vote winning Unknown or empty
    buffer) or ``{"brand": str, "model": str}`` without a confidence; the
    smoothed confidence lives on ``SmoothedLabel.confidence``. The API shape
    in DESIGN.md §5.1 collapses these two fields into one object.
    """
    if s.label is None:
        return None
    return {
        "brand": s.label["brand"],
        "model": s.label["model"],
        "confidence": s.confidence,
    }


def _normalize_bbox(
    bbox_xyxy: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    """Pixel-space xyxy → ``[0, 1]`` xyxy.

    Done here rather than in :class:`Detector` because the detector's
    consumer (this module) also needs pixel coords for the classifier crop;
    keeping the detector pixel-native means we don't multiply-then-divide on
    every frame.
    """
    return (
        bbox_xyxy[0] / width,
        bbox_xyxy[1] / height,
        bbox_xyxy[2] / width,
        bbox_xyxy[3] / height,
    )


def _crop(
    frame: np.ndarray,
    bbox_xyxy: tuple[float, float, float, float],
) -> np.ndarray:
    """Crop ``frame`` to the (possibly fractional) ``bbox_xyxy``.

    Detector bboxes are in pixel coordinates but the YOLO/ByteTrack pipeline
    can emit sub-pixel floats; we round to the nearest integer and clip to
    the frame bounds so the slice can never be empty for a valid bbox
    (validated upstream: ``x1 < x2`` and ``y1 < y2``).
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    xi1 = max(0, min(w - 1, int(round(x1))))
    yi1 = max(0, min(h - 1, int(round(y1))))
    xi2 = max(xi1 + 1, min(w, int(round(x2))))
    yi2 = max(yi1 + 1, min(h, int(round(y2))))
    return frame[yi1:yi2, xi1:xi2]


class Pipeline:
    """Frame-in, ``DetectionEvent``-out orchestrator.

    Owns the per-track vote state and the track registry. Stateful across
    frames: callers must reuse the same :class:`Pipeline` instance for the
    lifetime of a session, otherwise ByteTrack ids, vote history, and the
    "should I classify this frame?" cadence all reset.
    """

    def __init__(
        self,
        detector: Detector,
        classifier: Classifier,
        settings: Settings,
    ) -> None:
        self._detector = detector
        self._classifier = classifier
        self._settings = settings
        self._tracks = TrackRegistry()
        self._votes: dict[int, TrackVote] = {}
        # Counter exposed for diagnostics: if this dominates total detections
        # in a long session, ByteTrack is misconfigured (persist=False,
        # confirmation threshold too high, etc.) and the pipeline is silently
        # discarding every box. Read by scripts/benchmark.py.
        self.unconfirmed_skips: int = 0
        # Total YOLO detections (any track_id, including None) — useful for
        # distinguishing "YOLO sees nothing" from "ByteTrack rejected everything".
        self.raw_detections_total: int = 0
        # Per-(brand, model) tally of raw classifier outputs since the last
        # reset. Lets an operator distinguish "CLIP returns Unknown for every
        # crop" from "CLIP labels but vote can't converge" — the former means
        # the rejection prompts are winning, the latter means flapping.
        self.classifier_label_counts: Counter[tuple[str, str]] = Counter()

    def process_frame(
        self,
        frame: np.ndarray,
        frame_no: int,
        *,
        session_id: str = "local",
        frame_ts: int | None = None,
    ) -> DetectionEventDict:
        """Run detection + classification on one frame; return a DetectionEvent.

        ``frame_no`` is the monotonic per-session frame index — used as
        ``DetectionEvent.frame_id`` on the wire and as the time axis for the
        track registry (``observe``, ``should_classify``, ``prune``).

        ``frame_ts`` is the unix-ms wallclock the gateway recorded when it
        ingested this frame; if the caller doesn't have one (webcam smoke
        test), we default to ``now`` so downstream charts don't have to
        special-case ``None``.

        Detections with ``track_id=None`` (ByteTrack still warming up on a
        new box) are dropped on the floor — see ``test_track_id_none_is_skipped``
        for the rationale.
        """
        if frame_ts is None:
            frame_ts = int(time.time() * 1000)

        height, width = frame.shape[:2]
        frame_area = float(width * height)

        detections: list[Detection] = self._detector.detect_and_track(frame)
        self.raw_detections_total += len(detections)
        tracks_out: list[TrackDetectionDict] = []

        for det in detections:
            # ByteTrack hasn't confirmed this box yet; skip emission entirely
            # rather than ship a TrackDetection with a None track_id which
            # downstream consumers cannot join on across frames.
            if det.track_id is None:
                self.unconfirmed_skips += 1
                continue

            tid: int = det.track_id
            self._tracks.observe(tid, frame_no)

            x1, y1, x2, y2 = det.bbox_xyxy
            bbox_pixel_area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
            bbox_area_fraction = bbox_pixel_area / frame_area if frame_area > 0 else 0.0

            vote = self._votes.setdefault(
                tid,
                TrackVote(
                    window=self._settings.VOTE_WINDOW,
                    stable_min=self._settings.VOTE_STABLE_MIN,
                    stable_conf=self._settings.VOTE_STABLE_CONF,
                ),
            )

            # Stability check must happen *before* should_classify so the
            # registry can switch to the cheap STABLE_CLASSIFY_INTERVAL
            # cadence as soon as the vote converges.
            smoothed = vote.current()

            if self._tracks.should_classify(
                tid,
                frame_no,
                stable=smoothed.stable,
                bbox_area_fraction=bbox_area_fraction,
            ):
                crop = _crop(frame, det.bbox_xyxy)
                raw = self._classifier.classify(crop)
                self.classifier_label_counts[(raw["brand"], raw["model"])] += 1
                vote.update(raw)
                # Refresh after the update so the emitted label reflects this
                # frame's contribution instead of the previous snapshot.
                smoothed = vote.current()

            tracks_out.append(
                TrackDetectionDict(
                    track_id=tid,
                    bbox=_normalize_bbox(det.bbox_xyxy, width, height),
                    detection_confidence=det.confidence,
                    label=_smoothed_to_label(smoothed),
                    stable=smoothed.stable,
                    age_frames=self._tracks.age(tid, frame_no),
                )
            )

        # Sweep tracks that have aged out, and drop their vote state in
        # lock-step. Without this, ``self._votes`` would grow without bound
        # as ByteTrack assigns new ids to re-entering objects.
        pruned = self._tracks.prune(frame_no)
        for tid in pruned:
            self._votes.pop(tid, None)

        return DetectionEventDict(
            session_id=session_id,
            frame_id=frame_no,
            frame_ts=frame_ts,
            inference_ts=int(time.time() * 1000),
            tracks=tracks_out,
        )
