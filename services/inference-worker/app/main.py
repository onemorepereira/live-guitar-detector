"""Inference worker CLI entry point.

Two modes:

* ``python -m app.main`` (no flags) — the production Redis Streams consumer
  mode: discovers active sessions from ``sessions:active``, joins the
  ``inference`` consumer group on each ``frames:{sid}`` stream, runs the
  Phase 1 pipeline, and publishes detection events back on ``detections:{sid}``.
  See ``app.consumer`` for the supervisor + per-session loop.

* ``python -m app.main --webcam IDX`` — the local OpenCV demo from Phase 1.
  Useful for proving the detector + classifier + voting integrate end-to-end
  on a real camera feed without standing up Redis. Preserved unchanged.

Each mode owns its imports (the consumer doesn't pull in OpenCV's window
machinery; the webcam demo doesn't pull in ``redis.asyncio``) so ``--help``
stays fast and a missing optional dep can't break the other mode.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from app.config import Settings


def _build_parser() -> argparse.ArgumentParser:
    """Construct the top-level CLI parser.

    Split into its own function so tests (and ``--help`` smoke tests) can
    exercise argument parsing without paying the heavy import cost of either
    runtime mode — :func:`main` only imports the consumer / webcam modules
    when the corresponding mode is actually requested.
    """
    parser = argparse.ArgumentParser(
        prog="app.main",
        description=(
            "Inference worker CLI. Default mode runs the Redis Streams "
            "consumer; --webcam runs the local OpenCV demo."
        ),
    )
    parser.add_argument(
        "--webcam",
        type=int,
        metavar="IDX",
        help="Run the local-webcam demo on the given OpenCV device index (e.g. 0).",
    )
    return parser


def _resolve_prompts_path(settings: Settings) -> Path | None:
    """Return a usable prompts path or ``None`` if neither default exists.

    Mirrors ``app.webcam_demo._resolve_prompts_path``: prefers the configured
    path (typically ``/config/prompts.yaml`` in the container), falls back to
    the in-repo ``docs/prompts.md`` so a developer can ``python -m app.main``
    on their laptop without mounting anything.
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
    """Return a usable models dir or ``None`` if neither default exists.

    Same shape as ``_resolve_prompts_path``: container default first, then the
    ``app/models/`` directory ``scripts/download_models.py`` populates locally.
    """
    configured = settings.MODELS_DIR
    if (configured / "yolov8n-oiv7-fp32").is_dir():
        return configured
    worker_root = Path(__file__).resolve().parents[1]
    local = worker_root / "app" / "models"
    if (local / "yolov8n-oiv7-fp32").is_dir():
        return local
    return None


def _resolve_probe_path(settings: Settings, models_dir: Path) -> Path | None:
    """Return a usable trained-probe path or ``None`` if neither exists.

    Same container-first / local-fallback shape as the other resolvers.
    ``models_dir`` is the already-resolved active models directory so we
    don't reach into ``settings.MODELS_DIR`` for the fallback (which may
    be ``/models`` and unavailable outside the container).
    """
    configured = settings.PROBE_PATH
    if configured.is_file():
        return configured
    local = models_dir / "classifier-probe" / "probe.npz"
    if local.is_file():
        return local
    return None


async def _run_consumer(settings: Settings) -> int:
    """Construct pipeline + Redis client and run the consumer supervisor.

    Imports are inline so the cold-start cost of ultralytics / openvino /
    open_clip lands in this mode only — ``--webcam`` doesn't pay for the
    Redis client and ``--help`` doesn't pay for either.

    Returns a process exit code: 0 on clean shutdown, 1 on a fatal
    initialisation failure (missing models or prompts).
    """
    import redis.asyncio as redis_async

    from app.classifier import Classifier
    from app.consumer import Consumer
    from app.detector import Detector
    from app.pipeline import Pipeline
    from app.prompts import load_prompts

    # Only the zero-shot classifier needs prompts.md; probe mode embeds
    # its label space in the trained `.npz` artifact itself.
    prompts_path: Path | None = None
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

        probe_path = _resolve_probe_path(settings, models_dir)
        if probe_path is None:
            print(
                f"probe head not found at {settings.PROBE_PATH} "
                f"(or under {models_dir}/classifier-probe/); train one with "
                "scripts/train_probe.py or set CLASSIFIER_MODE=zero_shot.",
                file=sys.stderr,
            )
            return 1
        print(f"loading probe head from {probe_path}", flush=True)
        classifier = ProbeClassifier(models_dir, probe_path, input_size=settings.CLIP_INPUT_SIZE)
    else:
        print(f"loading prompts from {prompts_path}", flush=True)
        prompts = load_prompts(prompts_path)
        classifier = Classifier(models_dir, prompts, input_size=settings.CLIP_INPUT_SIZE)

    pipeline = Pipeline(detector, classifier, settings)

    # ``decode_responses=False`` is mandatory: the frames stream carries raw
    # JPEG bytes, decoding to str would mangle them.
    r = redis_async.from_url(settings.REDIS_URL, decode_responses=False)
    # ``HOSTNAME`` is set by Kubernetes to the pod name — gives each replica
    # a unique consumer name within the ``inference`` group. Falls back to a
    # static name for local runs where multiple workers aren't expected.
    consumer = Consumer(r, pipeline, consumer_name=os.environ.get("HOSTNAME", "worker-1"))

    # K8s readiness probe (DESIGN.md §7.3): /tmp/ready is the file the probe
    # `cat`s. Best-effort — failure to create it shouldn't kill the worker
    # (e.g. a read-only /tmp in some test sandboxes).
    try:
        Path("/tmp/ready").touch()
    except OSError as exc:
        print(f"warning: could not touch /tmp/ready: {exc}", file=sys.stderr)

    print(f"consumer started as {consumer._consumer_name}", flush=True)
    try:
        await consumer.run()
    finally:
        await r.aclose()
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI dispatcher. Returns a process exit code.

    ``argv`` is exposed for testability — the production call site
    (``if __name__ == '__main__'``) defaults to ``sys.argv[1:]``.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    settings = Settings()

    if args.webcam is not None:
        # Lazy import: cv2 GUI + ultralytics + openvino + open_clip pull in
        # hundreds of MB of native code; we don't want ``--help`` or the
        # Redis consumer path to pay for that.
        from app.webcam_demo import run_webcam

        return run_webcam(cam_index=args.webcam, settings=settings)

    # Default: Redis consumer mode.
    return asyncio.run(_run_consumer(settings))


if __name__ == "__main__":
    sys.exit(main())
