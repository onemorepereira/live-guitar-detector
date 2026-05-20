"""Inference worker CLI entry point.

Phase 1 supports only ``python -m app.main --webcam IDX``: a local
OpenCV demo that proves the detector + classifier + voting pipeline
integrate end-to-end on a real camera feed (DESIGN.md §8 Phase 1 manual
gate).

Phase 2 will land the Redis Streams consumer loop here as the *default*
mode (no flag); for now invoking ``app.main`` without ``--webcam`` raises
``NotImplementedError`` so the missing path is obvious rather than silent.
"""

from __future__ import annotations

import argparse
import sys

from app.config import Settings


def _build_parser() -> argparse.ArgumentParser:
    """Construct the top-level CLI parser.

    Split into its own function so tests (and ``--help`` smoke tests) can
    exercise argument parsing without paying the heavy import cost of the
    webcam demo module — :func:`main` only imports
    :mod:`app.webcam_demo` when ``--webcam`` is actually requested.
    """
    parser = argparse.ArgumentParser(
        prog="app.main",
        description="Inference worker CLI. Phase 1: local webcam demo only.",
    )
    parser.add_argument(
        "--webcam",
        type=int,
        metavar="IDX",
        help="Run the local-webcam demo on the given OpenCV device index (e.g. 0).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI dispatcher. Returns a process exit code.

    ``argv`` is exposed for testability — the production call site
    (``if __name__ == '__main__'``) defaults to ``sys.argv[1:]``.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    settings = Settings()

    if args.webcam is not None:
        # Lazy import: cv2 + ultralytics + openvino + open_clip pull in
        # hundreds of MB of native code; we don't want ``--help`` to pay
        # for that.
        from app.webcam_demo import run_webcam

        return run_webcam(cam_index=args.webcam, settings=settings)

    raise NotImplementedError(
        "Redis consumer mode is a Phase 2 task. For Phase 1, run with --webcam IDX."
    )


if __name__ == "__main__":
    sys.exit(main())
