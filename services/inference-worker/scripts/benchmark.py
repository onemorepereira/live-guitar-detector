"""Phase 1 benchmark — measures end-to-end Pipeline latency on cycled fixtures.

Reports p50 / p95 / p99 of:

  - detect_ms      (Detector.detect_and_track only)
  - classify_ms    (Classifier.classify on one bbox crop, sampled per-frame)
  - total_ms       (Pipeline.process_frame end-to-end)

The Phase 1 done-when gate (DESIGN.md §8) is ``p50 total_ms < 50ms`` on the
target hardware. DESIGN.md targets a Ryzen 7 specifically; this script
records the host details (``platform.processor()`` / ``platform.platform()``)
in its output so a fast dev box and the eventual deployment host can be
compared apples-to-apples in ``docs/BENCHMARKS.md``.

The per-stage detector + classifier timings are sampled outside the
:class:`~app.pipeline.Pipeline` instance so we don't have to instrument the
orchestrator. The ``total_ms`` number IS the gate metric; the per-stage
numbers are diagnostics so we can tell which stage is hot when total is
borderline.
"""

from __future__ import annotations

import argparse
import platform
import statistics
import sys
import time
from pathlib import Path

import cv2

from app.classifier import Classifier
from app.config import Settings
from app.detector import Detector
from app.pipeline import Pipeline
from app.prompts import load_prompts


def percentile(xs: list[float], p: float) -> float:
    """Nearest-rank percentile of ``xs`` at fraction ``p`` (0..1).

    Avoids pulling numpy onto the hot path of a benchmark; for N=600 the
    nearest-rank approximation is well within the noise of CPU jitter.
    """
    if not xs:
        return 0.0
    xs_sorted = sorted(xs)
    k = max(0, min(len(xs_sorted) - 1, round(p * (len(xs_sorted) - 1))))
    return xs_sorted[k]


def _resolve_paths() -> tuple[Path, Path, Path]:
    """Return ``(prompts_path, models_dir, fixtures_dir)`` for a local checkout.

    We're always run from a developer checkout (the benchmark isn't packaged
    into the container image), so the in-repo defaults are the only paths
    we have to support — no need to plumb Settings.MODELS_DIR / PROMPTS_FILE.
    """
    repo_root = Path(__file__).resolve().parents[3]
    worker_root = Path(__file__).resolve().parents[1]
    prompts_path = repo_root / "docs" / "prompts.md"
    models_dir = worker_root / "app" / "models"
    fixtures_dir = worker_root / "tests" / "fixtures" / "images"
    return prompts_path, models_dir, fixtures_dir


def run(frames: int, target_fps: float, settings: Settings) -> int:
    """Execute the benchmark loop and print the report. Returns shell exit code.

    Exit codes:
      ``0`` — Phase 1 gate passed (p50 total < 50ms).
      ``1`` — Setup error (missing fixtures/models/prompts).
      ``2`` — Gate failed; see DESIGN.md §10.5 tuning checklist.
    """
    prompts_path, models_dir, fixtures_dir = _resolve_paths()

    if not prompts_path.is_file():
        print(f"prompts file not found at {prompts_path}", file=sys.stderr)
        return 1
    if not (models_dir / "yolov8n-oiv7-fp32").is_dir():
        print(
            f"models not found at {models_dir}; run scripts/download_models.py first",
            file=sys.stderr,
        )
        return 1

    fixture_files = sorted(fixtures_dir.glob("*.jpg"))
    if not fixture_files:
        print(f"no fixture .jpg files in {fixtures_dir}", file=sys.stderr)
        return 1

    print(f"loading prompts from {prompts_path}", flush=True)
    prompts = load_prompts(prompts_path)

    print(f"loading models from {models_dir}", flush=True)
    detector = Detector(
        models_dir / "yolov8n-oiv7-fp32",
        conf=settings.DETECT_CONF,
        iou=settings.DETECT_IOU,
        imgsz=settings.DETECT_IMGSZ,
    )
    classifier = Classifier(models_dir, prompts, input_size=settings.CLIP_INPUT_SIZE)
    pipeline = Pipeline(detector, classifier, settings)

    fixture_frames = [cv2.imread(str(p), cv2.IMREAD_COLOR) for p in fixture_files]
    # cv2.imread returns None on a corrupt / unreadable file; surface that
    # now so we don't NPE deep inside detect_and_track.
    for path, frame in zip(fixture_files, fixture_frames, strict=True):
        if frame is None:
            print(f"failed to decode fixture {path}", file=sys.stderr)
            return 1

    print(
        f"loaded {len(fixture_files)} fixture frames; cycling for {frames} "
        f"iterations at target {target_fps} fps",
        flush=True,
    )

    detect_ms: list[float] = []
    classify_ms: list[float] = []
    total_ms: list[float] = []
    dropped = 0
    target_period_ms = 1000.0 / target_fps

    # Warm-up: first calls into OpenVINO compile kernels and prime caches.
    # 10 frames is enough to stabilize after the first inference. Numbers
    # from warm-up are discarded so they don't skew percentiles.
    warmup_n = min(10, len(fixture_frames))
    for i in range(warmup_n):
        pipeline.process_frame(fixture_frames[i % len(fixture_frames)], i)

    # Reset the pipeline so warm-up frames don't leave residual track state
    # (ByteTrack ids, vote history, registry first-seen) bleeding into the
    # measured loop. Detector/classifier are stateless across calls; only
    # the orchestrator carries cross-frame state.
    pipeline = Pipeline(detector, classifier, settings)

    for i in range(frames):
        frame = fixture_frames[i % len(fixture_frames)]

        # Time detect separately by calling the detector directly. This
        # double-runs the detector (once here, once inside process_frame
        # below), but the goal is shape of per-stage cost — not summing to
        # total — so the duplication is harmless.
        t_det0 = time.perf_counter()
        dets = detector.detect_and_track(frame)
        t_det1 = time.perf_counter()
        detect_ms.append((t_det1 - t_det0) * 1000.0)

        # Time classify on the first detected box when present. Synthetic
        # fixtures may produce zero detections; in that case classify_ms
        # has fewer samples than detect_ms — handled in the report below.
        if dets:
            x1, y1, x2, y2 = dets[0].bbox_xyxy
            h, w = frame.shape[:2]
            xi1 = max(0, min(w - 1, round(x1)))
            yi1 = max(0, min(h - 1, round(y1)))
            xi2 = max(xi1 + 1, min(w, round(x2)))
            yi2 = max(yi1 + 1, min(h, round(y2)))
            crop = frame[yi1:yi2, xi1:xi2]
            if crop.size > 0:
                t_cls0 = time.perf_counter()
                classifier.classify(crop)
                t_cls1 = time.perf_counter()
                classify_ms.append((t_cls1 - t_cls0) * 1000.0)

        # Total: the actual gate metric. Pipeline.process_frame runs its
        # own detect + classify + track + vote — totally independent of the
        # per-stage instrumentation above.
        t0 = time.perf_counter()
        pipeline.process_frame(frame, i)
        t1 = time.perf_counter()
        ms = (t1 - t0) * 1000.0
        total_ms.append(ms)

        if ms > target_period_ms:
            dropped += 1

    def line(name: str, xs: list[float]) -> str:
        if not xs:
            return f"  {name:12s} N=   0  (no samples)"
        return (
            f"  {name:12s} N={len(xs):4d}  "
            f"p50={percentile(xs, 0.5):6.2f}ms  "
            f"p95={percentile(xs, 0.95):6.2f}ms  "
            f"p99={percentile(xs, 0.99):6.2f}ms  "
            f"mean={statistics.fmean(xs):6.2f}ms"
        )

    print()
    print(f"Benchmark: {frames} frames at target {target_fps} fps")
    print(f"Host:    {platform.processor() or 'unknown CPU'}")
    print(f"Platform: {platform.platform()}")
    print(f"Python:  {platform.python_version()}")
    print()
    print(line("detect_ms", detect_ms))
    print(line("classify_ms", classify_ms))
    print(line("total_ms", total_ms))
    print()
    print(
        f"Dropped (total > {target_period_ms:.1f}ms): "
        f"{dropped} / {frames} ({100.0 * dropped / frames:.1f}%)"
    )

    # Diagnostic: if every detection had track_id=None the pipeline emits
    # zero TrackDetections regardless of how fast it ran, which would make
    # total_ms unrepresentative of a real session. Surface this so a clean
    # gate pass with no tracked detections doesn't go unnoticed.
    print(f"Unconfirmed-track skips: {pipeline.unconfirmed_skips}")

    p50_total = percentile(total_ms, 0.5)
    print()
    if p50_total < 50.0:
        print(f"Phase 1 gate PASSED: p50 total = {p50_total:.2f}ms < 50ms target")
        return 0
    print(f"Phase 1 gate FAILED: p50 total = {p50_total:.2f}ms >= 50ms target")
    print("  See DESIGN.md §10.5 tuning checklist before proceeding to Phase 2.")
    return 2


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1 inference benchmark.")
    parser.add_argument(
        "--frames",
        type=int,
        default=600,
        help="Number of frames to benchmark (default: 600).",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=15.0,
        help="Target FPS for the dropped-frame budget (default: 15).",
    )
    args = parser.parse_args()
    return run(frames=args.frames, target_fps=args.target_fps, settings=Settings())


if __name__ == "__main__":
    sys.exit(main())
