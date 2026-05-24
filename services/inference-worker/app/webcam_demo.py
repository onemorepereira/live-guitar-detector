"""Local webcam demo for the Phase 1 manual gate.

Opens an OpenCV ``VideoCapture``, feeds each frame through the inference
:class:`~app.pipeline.Pipeline`, overlays detection boxes + brand/model
labels, and prints per-second average + p95 latency to stdout. Press ``q``
to quit.

This is the Phase 1 done-when artifact (DESIGN.md §8 Phase 1): it proves
the detector + tracker + classifier + voting integrate end-to-end on a
real camera feed before any web/Redis plumbing is built.

Why this lives under ``app/`` and not ``scripts/``: keeping the real logic
inside the Python package means ``app.main`` can import it cleanly without
having to make ``scripts/`` an installable package or add it to
``sys.path``. ``scripts/webcam_demo.py`` is a thin shim that delegates here.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

from app.classifier import Classifier
from app.config import Settings
from app.detector import Detector
from app.pipeline import DetectionEventDict, Pipeline
from app.prompts import load_prompts

# Brand → BGR colour map. DESIGN.md §5.6 specifies the HUD palette in HTML
# hex (``#RRGGBB``); OpenCV draws in BGR so each tuple below is the byte
# reversal of the corresponding hex:
#
#   Gibson  #C8A45C  → RGB(200, 164,  92) → BGR( 92, 164, 200)  (warm gold)
#   Fender  #F5F5F5  → RGB(245, 245, 245) → BGR(245, 245, 245)  (off-white)
#   Unknown #888888  → RGB(136, 136, 136) → BGR(136, 136, 136)  (mid-grey)
#
# These intentionally match the eventual web HUD so that a printed-photo
# demo here looks the same as the production canvas overlay.
BRAND_COLORS: dict[str, tuple[int, int, int]] = {
    "Gibson": (92, 164, 200),
    "Fender": (245, 245, 245),
    "Unknown": (136, 136, 136),
}

_WINDOW_TITLE = "guitar-detect (q to quit)"


def _resolve_prompts_path(settings: Settings) -> Path | None:
    """Return a usable prompts path or ``None`` if neither default exists.

    The container default is ``/config/prompts.yaml``; on a developer box
    that path doesn't exist, so we fall back to the canonical
    ``docs/prompts.md`` in the repo root. Both are first-class — the
    loader handles ``.md`` files by extracting the embedded YAML block.
    """
    configured = settings.PROMPTS_FILE
    if configured.is_file():
        return configured
    repo_root = Path(__file__).resolve().parents[3]
    fallback = repo_root / "docs" / "prompts.md"
    if fallback.is_file():
        return fallback
    return None


def _resolve_models_dir(settings: Settings) -> Path | None:
    """Return a usable models directory or ``None`` if neither default exists.

    Mirrors ``_resolve_prompts_path``: prefers the container path
    (``/models``), falls back to the in-repo ``app/models/`` that
    ``scripts/download_models.py`` populates for local development.
    """
    configured = settings.MODELS_DIR
    if (configured / "yolov8n-oiv7-fp32").is_dir():
        return configured
    worker_root = Path(__file__).resolve().parents[1]
    local = worker_root / "app" / "models"
    if (local / "yolov8n-oiv7-fp32").is_dir():
        return local
    return None


def draw_overlay(frame: np.ndarray, event: DetectionEventDict) -> None:
    """Draw detection bboxes + labels onto ``frame`` in place.

    ``event["tracks"][i]["bbox"]`` is in normalized ``[0, 1]`` coordinates
    (see ``Pipeline._normalize_bbox``); we de-normalize back to pixel space
    here. The classifier label may be ``None`` (track still warming up); in
    that case we render a neutral "Analyzing..." placeholder so the user
    can still see the box.
    """
    h, w = frame.shape[:2]
    for t in event["tracks"]:
        nx1, ny1, nx2, ny2 = t["bbox"]
        x1, y1 = int(nx1 * w), int(ny1 * h)
        x2, y2 = int(nx2 * w), int(ny2 * h)

        label = t["label"]
        brand = label["brand"] if label else "Unknown"
        color = BRAND_COLORS.get(brand, BRAND_COLORS["Unknown"])

        # 1-px black inner stroke gives the coloured box contrast against
        # both bright (white guitar body) and dark (stage) backgrounds.
        cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (0, 0, 0), 1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        if t["stable"] and label is not None:
            conf_pct = int(round(label["confidence"] * 100))
            label_text = f"#{t['track_id']} {label['brand']} {label['model']} · {conf_pct}%"
        else:
            label_text = f"#{t['track_id']} Analyzing..."

        # Text with black background pill. Placement: above the box, unless
        # the box hugs the top of the frame in which case flip to inside-top.
        font = cv2.FONT_HERSHEY_DUPLEX
        scale = 0.5
        thickness = 1
        (tw, th), _baseline = cv2.getTextSize(label_text, font, scale, thickness)
        text_y = y1 - 6 if (y1 - 6 - th) >= 0 else y2 + th + 6
        cv2.rectangle(
            frame,
            (x1, text_y - th - 4),
            (x1 + tw + 8, text_y + 4),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            frame,
            label_text,
            (x1 + 4, text_y),
            font,
            scale,
            (255, 255, 255),
            thickness,
            lineType=cv2.LINE_AA,
        )


def _report_latency(frame_no: int, latencies: list[float]) -> None:
    """Print ``frames/s``, ``avg`` and ``p95`` for the last accumulator window.

    Called once per wall-clock second from the main loop; ``latencies`` is
    cleared by the caller afterwards. ``p95`` uses nearest-rank on a sorted
    list — exact enough for a one-second sample, and avoids pulling in numpy
    for what would be a single percentile call.
    """
    if not latencies:
        return
    count = len(latencies)
    avg_ms = 1000.0 * sum(latencies) / count
    p95_idx = max(0, int(round(0.95 * (count - 1))))
    p95_ms = 1000.0 * sorted(latencies)[p95_idx]
    print(
        f"frame {frame_no}: {count} fps, avg={avg_ms:.1f}ms p95={p95_ms:.1f}ms",
        flush=True,
    )


def run_webcam(cam_index: int, settings: Settings) -> int:
    """Open the webcam, loop the pipeline, render the HUD.

    Returns ``0`` on clean user exit (``q`` pressed), non-zero on init
    failures (camera not openable, model directory missing, prompts file
    missing). Designed to be wrapped by :func:`app.main.main` and by the
    standalone ``scripts/webcam_demo.py`` shim — both pass a fresh
    :class:`Settings` so env-var overrides still work.
    """
    # Only the zero-shot classifier needs prompts.md; probe modes embed
    # their label space in the trained `.npz` artifact itself.
    prompts_path = None
    if settings.CLASSIFIER_MODE == "zero_shot":
        prompts_path = _resolve_prompts_path(settings)
        if prompts_path is None:
            print(
                f"prompts file not found (checked {settings.PROMPTS_FILE} and "
                "<repo>/docs/prompts.md); set PROMPTS_FILE or run from a repo checkout.",
                file=sys.stderr,
            )
            return 1

    models_dir = _resolve_models_dir(settings)
    if models_dir is None:
        print(
            f"models directory not found (checked {settings.MODELS_DIR} and "
            "<worker>/app/models/); run scripts/download_models.py all first.",
            file=sys.stderr,
        )
        return 1

    print(f"loading models from {models_dir}", flush=True)
    detector = Detector(
        models_dir / "yolov8n-oiv7-fp32",
        conf=settings.DETECT_CONF,
        iou=settings.DETECT_IOU,
        imgsz=settings.DETECT_IMGSZ,
    )

    if settings.CLASSIFIER_MODE == "probe":
        from app.probe_classifier import ProbeClassifier

        configured = settings.PROBE_PATH
        probe_path = configured if configured.is_file() else None
        if probe_path is None:
            local = models_dir / "classifier-probe" / "probe.npz"
            if local.is_file():
                probe_path = local
        if probe_path is None:
            print(
                f"probe head not found at {configured} or {models_dir}/classifier-probe/; "
                "train one with scripts/train_probe.py or set CLASSIFIER_MODE=zero_shot.",
                file=sys.stderr,
            )
            return 1
        print(f"loading probe head from {probe_path}", flush=True)
        classifier = ProbeClassifier(models_dir, probe_path, input_size=settings.CLIP_INPUT_SIZE)
    elif settings.CLASSIFIER_MODE == "siglip_probe":
        from app.siglip_probe_classifier import SigLIPProbeClassifier

        configured = settings.SIGLIP_PROBE_PATH
        siglip_probe_path = configured if configured.is_file() else None
        if siglip_probe_path is None:
            local = models_dir / "classifier-probe" / "probe_siglip.npz"
            if local.is_file():
                siglip_probe_path = local
        if siglip_probe_path is None:
            print(
                f"siglip probe head not found at {configured} or {models_dir}/classifier-probe/; "
                "train one with scripts/train_probe.py --backend siglip.",
                file=sys.stderr,
            )
            return 1
        print(
            f"loading SigLIP probe head from {siglip_probe_path}"
            f" (encoder: {settings.SIGLIP_MODEL_ID})",
            flush=True,
        )
        classifier = SigLIPProbeClassifier(siglip_probe_path, model_id=settings.SIGLIP_MODEL_ID)
    else:
        print(f"loading prompts from {prompts_path}", flush=True)
        prompts = load_prompts(prompts_path)
        classifier = Classifier(models_dir, prompts, input_size=settings.CLIP_INPUT_SIZE)

    pipeline = Pipeline(detector, classifier, settings)

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"could not open webcam at index {cam_index}", file=sys.stderr)
        return 1

    print(f"webcam {cam_index} opened. Press 'q' in the window to quit.", flush=True)

    frame_no = 0
    latencies: list[float] = []
    last_report = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("webcam read failed; exiting.", file=sys.stderr)
                return 1

            t0 = time.perf_counter()
            event = pipeline.process_frame(frame, frame_no)
            t1 = time.perf_counter()
            latencies.append(t1 - t0)

            draw_overlay(frame, event)
            cv2.imshow(_WINDOW_TITLE, frame)

            now = time.time()
            if now - last_report >= 1.0:
                _report_latency(frame_no, latencies)
                latencies.clear()
                last_report = now

            # waitKey(1) is mandatory: it both pumps the GUI event loop
            # (without it the window stays blank on most platforms) and
            # returns the key the user pressed. Mask to 0xFF to discard
            # modifier bits some OpenCV builds set on the high byte.
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

            frame_no += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0
