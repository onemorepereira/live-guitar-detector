"""Webcam demo CLI shim.

Standalone alternative to ``python -m app.main --webcam IDX``: same demo,
slightly different argument flavour (``--cam`` defaults to ``0`` for the
zero-arg case). All real logic lives in :mod:`app.webcam_demo`; this file
intentionally stays trivial so packaging stays clean (``scripts/`` is not
an installable package).
"""

from __future__ import annotations

import argparse
import sys

from app.config import Settings
from app.webcam_demo import run_webcam


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the local webcam Phase-1 demo.")
    parser.add_argument(
        "--cam",
        type=int,
        default=0,
        help="OpenCV VideoCapture index (default: 0).",
    )
    args = parser.parse_args()
    return run_webcam(cam_index=args.cam, settings=Settings())


if __name__ == "__main__":
    sys.exit(main())
