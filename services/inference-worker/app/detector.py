"""YOLO + ByteTrack detector wrapper (DESIGN.md §5.4).

Wraps Ultralytics' OpenVINO IR backend with two narrow entry points
(:meth:`Detector.detect`, :meth:`Detector.detect_and_track`) that return
plain :class:`Detection` pydantic models. Tracking state is owned by the
Ultralytics predictor and persisted across :meth:`detect_and_track` calls
on the same instance (``persist=True``), so callers should keep the
:class:`Detector` alive for the lifetime of a track.

The class table from the underlying weights is consulted once at load
time: if the loaded model does not expose a ``"Guitar"`` class we raise
:class:`RuntimeError`. This is the fail-fast contract from DESIGN.md §5.4
— silently picking a wrong class id would let bad weights ship to prod and
produce nonsense detections.

The OpenVINO IR is loaded via Ultralytics' ``YOLO`` entrypoint. Ultralytics
auto-detects the backend by directory-name suffix (``*_openvino_model``);
our canonical export layout (see ``scripts/download_models.py``) writes
``yolov8n-oiv7-fp32/`` instead, so the constructor sets up a small symlink
into a temp directory at load time when needed. This is an implementation
detail callers shouldn't have to know about.
"""

from __future__ import annotations

import atexit
import shutil
import tempfile
from pathlib import Path
from typing import Annotated

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from ultralytics import YOLO

# Reuse the prompts module's convention for non-empty strings so that
# ``class_name=""`` raises a validation error instead of silently passing.
_NonEmptyStr = Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]

# Directory-name suffix Ultralytics' autobackend uses to identify a
# "this is an OpenVINO IR directory" path. Anything else makes ``YOLO(...)``
# raise ``TypeError`` before we ever get to inference.
_OPENVINO_DIR_SUFFIX = "_openvino_model"


class Detection(BaseModel):
    """A single object detection emitted by :class:`Detector`.

    Fields:
        track_id: ByteTrack id when produced by :meth:`Detector.detect_and_track`,
            ``None`` when produced by :meth:`Detector.detect` (no tracker)
            or when the tracker hasn't confirmed this box yet.
        bbox_xyxy: ``(x1, y1, x2, y2)`` in **pixel** coordinates of the
            input frame. Normalization to ``[0, 1]`` happens at the
            pipeline-orchestration layer; keeping pixels here means
            downstream code that wants to crop the original frame
            doesn't have to know its dimensions twice.
        confidence: YOLO confidence in ``[0, 1]``.
        class_name: Human-readable class name from the underlying weights
            (``"Guitar"`` in production, but the model isn't hard-coded to
            that string so unit tests with synthetic class tables work).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    track_id: int | None
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float = Field(ge=0.0, le=1.0)
    class_name: _NonEmptyStr


def _resolve_openvino_path(model_dir: Path) -> Path:
    """Return a path that Ultralytics' autobackend will recognize as OpenVINO.

    Ultralytics keys backend detection off the directory's *name* suffix —
    it must contain ``_openvino_model`` (see ``export_formats()`` in
    ``ultralytics.engine.exporter``). Our exporter writes to a tidier name
    like ``yolov8n-oiv7-fp32/``; rather than rename the canonical layout
    we transparently expose a suffixed symlink from a temp directory when
    needed and clean it up at interpreter exit.
    """
    if not model_dir.exists():
        raise FileNotFoundError(f"detector model_dir does not exist: {model_dir}")
    if not model_dir.is_dir():
        raise NotADirectoryError(f"detector model_dir is not a directory: {model_dir}")

    xmls = list(model_dir.glob("*.xml"))
    if not xmls:
        raise FileNotFoundError(
            f"no .xml file found in {model_dir}; run scripts/download_models.py"
        )

    # If the directory name already advertises itself as OpenVINO, let
    # Ultralytics use it directly — no symlink, no temp files, simpler
    # error messages on failure.
    if _OPENVINO_DIR_SUFFIX in model_dir.name:
        return model_dir

    # Otherwise build a stable symlink in a temp dir. We use a per-process
    # tempdir scoped to this function so multiple Detector instances in
    # the same process don't stomp on each other.
    tmp_root = Path(tempfile.mkdtemp(prefix="guitar-detect-yolo-"))
    atexit.register(shutil.rmtree, tmp_root, ignore_errors=True)

    # The Ultralytics check is on `name.endswith(...)` in some code paths
    # and `<suffix> in name` in others — use the model directory's stem
    # plus the suffix for a name that matches both. Resolve to absolute
    # so the symlink survives any later chdir.
    link = tmp_root / f"{model_dir.name}{_OPENVINO_DIR_SUFFIX}"
    link.symlink_to(model_dir.resolve(), target_is_directory=True)
    return link


class Detector:
    """OpenVINO YOLOv8 detector with optional ByteTrack tracking.

    Parameters mirror DESIGN.md §5.7's worker env vars; callers should
    pass values from :class:`app.config.Settings`.
    """

    def __init__(
        self,
        model_dir: Path,
        conf: float = 0.35,
        iou: float = 0.5,
        imgsz: int = 416,
    ) -> None:
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self._model_dir = Path(model_dir)

        backend_path = _resolve_openvino_path(self._model_dir)
        # task='detect' avoids Ultralytics' task-autodetection heuristics,
        # which read metadata.yaml and can be flaky on hand-built IR dirs.
        self._model = YOLO(str(backend_path), task="detect")

        names = getattr(self._model, "names", None)
        if not isinstance(names, dict):
            raise RuntimeError(
                f"YOLO model at {model_dir} did not expose a class table "
                f"(names={names!r}); cannot determine the Guitar class id."
            )
        guitar_ids = [cid for cid, label in names.items() if str(label) == "Guitar"]
        if not guitar_ids:
            raise RuntimeError(
                f"YOLO model at {model_dir} does not have a 'Guitar' class — wrong weights?"
            )
        # OIv7 has exactly one Guitar class; if a future weights file has
        # multiple we'd want the caller to know rather than silently picking
        # the first.
        if len(guitar_ids) > 1:
            raise RuntimeError(
                f"YOLO model at {model_dir} exposes multiple 'Guitar' class ids "
                f"({guitar_ids}); ambiguous, cannot proceed."
            )
        self._guitar_id: int = int(guitar_ids[0])
        self._class_names: dict[int, str] = {int(k): str(v) for k, v in names.items()}

    # ------------------------------------------------------------------
    # Public read-only attributes
    # ------------------------------------------------------------------

    @property
    def guitar_class_id(self) -> int:
        """The class id that maps to ``"Guitar"`` in the loaded weights."""
        return self._guitar_id

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run a single detection pass on ``frame`` (BGR uint8 HxWx3).

        Returns a list of :class:`Detection` with ``track_id=None``. The
        underlying predictor is invoked with ``classes=[guitar_class_id]``
        so we never see non-guitar predictions cross the boundary, even if
        the weights cover hundreds of classes.
        """
        results = self._model.predict(
            frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            classes=[self._guitar_id],
            verbose=False,
        )
        return self._parse_results(results, with_track_ids=False)

    def detect_and_track(self, frame: np.ndarray) -> list[Detection]:
        """Run detection + ByteTrack on ``frame``.

        ``persist=True`` means the tracker carries its state across
        invocations of this method on the same Detector instance — essential
        for ByteTrack id continuity. Unconfirmed boxes (ByteTrack hasn't
        assigned them an id yet) come through with ``track_id=None`` so
        callers can choose to ignore them.
        """
        results = self._model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            classes=[self._guitar_id],
            verbose=False,
        )
        return self._parse_results(results, with_track_ids=True)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _parse_results(self, results, *, with_track_ids: bool) -> list[Detection]:
        """Convert an Ultralytics ``Results`` list into our pydantic objects.

        Ultralytics returns a list with one entry per input image; we
        always pass a single frame, so we read ``results[0]``. The
        ``Boxes`` object exposes ``xyxy``, ``conf``, ``cls`` and optionally
        ``id`` as torch tensors — we move them to CPU floats before
        instantiating :class:`Detection`.
        """
        if not results:
            return []
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy()

        track_ids: list[int | None]
        if with_track_ids and getattr(boxes, "id", None) is not None:
            # ``boxes.id`` is a 1-D tensor of floats (Ultralytics' choice).
            # Cast to int; ByteTrack uses 1-based ids so a 0 is meaningful
            # iff it ever appears (it shouldn't, but we don't filter).
            track_ids = [int(t) for t in boxes.id.cpu().numpy().tolist()]
        else:
            track_ids = [None] * len(xyxy)

        out: list[Detection] = []
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = (float(v) for v in xyxy[i])
            class_id = int(cls[i])
            out.append(
                Detection(
                    track_id=track_ids[i],
                    bbox_xyxy=(x1, y1, x2, y2),
                    confidence=float(conf[i]),
                    class_name=self._class_names.get(class_id, str(class_id)),
                )
            )
        return out
