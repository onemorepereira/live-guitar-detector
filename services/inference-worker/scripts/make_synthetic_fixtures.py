"""Generate synthetic placeholder JPEGs for the detector test suite.

These are *not* real guitars — they are deterministic geometric patterns that
keep the test suite hermetic and the repo small. The actual detector accuracy
tests are gated behind the ``requires_real_fixtures`` marker (see
``tests/conftest.py``); the synthetic fixtures only exist so the file
discovery / I/O paths work without a marker indicating real photos.

Run once to regenerate; the produced JPEGs (a few KB each) are committed:

    python scripts/make_synthetic_fixtures.py

The output directory defaults to ``tests/fixtures/images/`` next to the
worker package.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

# Single source of truth for which placeholder filenames the test suite
# expects to find. The plan calls for two Strats, two Les Pauls, one acoustic,
# and one no-guitar street scene.
_GUITAR_NAMES = (
    "strat_01.jpg",
    "strat_02.jpg",
    "lp_01.jpg",
    "lp_02.jpg",
    "acoustic_01.jpg",
)
_NEGATIVE_NAMES = ("street_scene.jpg",)


def _draw_guitar_silhouette(img: np.ndarray, seed: int) -> None:
    """Draw a crude guitar shape: a body ellipse plus a neck rectangle.

    The ``seed`` controls placement and color so successive calls produce
    distinct-but-deterministic images (no test flakiness, no need to commit
    five copies of the same bytes).
    """
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]

    body_color = tuple(int(c) for c in rng.integers(40, 220, size=3))
    neck_color = tuple(int(c) for c in rng.integers(40, 220, size=3))

    body_cx = w // 2 + int(rng.integers(-40, 40))
    body_cy = int(h * 0.65) + int(rng.integers(-20, 20))
    body_axes = (int(w * 0.18), int(h * 0.22))
    cv2.ellipse(img, (body_cx, body_cy), body_axes, 0, 0, 360, body_color, thickness=-1)

    # Neck: a tall thin rectangle reaching up toward the top of the frame.
    neck_w = max(8, int(w * 0.04))
    neck_top = int(h * 0.08)
    neck_x1 = body_cx - neck_w // 2
    neck_x2 = body_cx + neck_w // 2
    cv2.rectangle(img, (neck_x1, neck_top), (neck_x2, body_cy), neck_color, thickness=-1)


def _draw_street_scene(img: np.ndarray) -> None:
    """Draw a few overlapping rectangles to suggest "buildings" — no curves.

    YOLO trained on OIv7 doesn't care about our crude shapes; we mostly want
    a deterministic non-empty image with no guitar-shaped silhouettes so the
    "negative" test case has *something* to look at if a real fixture isn't
    committed yet.
    """
    rng = np.random.default_rng(42)
    h, w = img.shape[:2]
    for _ in range(8):
        x1 = int(rng.integers(0, w - 50))
        y1 = int(rng.integers(0, h - 50))
        x2 = min(w, x1 + int(rng.integers(40, 200)))
        y2 = min(h, y1 + int(rng.integers(40, 200)))
        color = tuple(int(c) for c in rng.integers(20, 200, size=3))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=-1)


def _new_canvas(seed: int) -> np.ndarray:
    """Return a 640x480x3 BGR canvas tinted by a seeded base color.

    A solid color compresses to a couple of KB under JPEG, which is exactly
    what we want — these fixtures only need to exist on disk and be
    visually distinct, not photorealistic. Noise blows the file size up
    by an order of magnitude and buys nothing for these tests.
    """
    rng = np.random.default_rng(seed)
    base = rng.integers(40, 80, size=3, dtype=np.int32)
    return np.tile(base.astype(np.uint8), (480, 640, 1)).astype(np.uint8)


def generate(out_dir: Path) -> list[Path]:
    """Write the placeholder JPEGs into ``out_dir`` and return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for i, name in enumerate(_GUITAR_NAMES):
        img = _new_canvas(seed=100 + i)
        _draw_guitar_silhouette(img, seed=200 + i)
        path = out_dir / name
        # quality=90 keeps the JPEGs in the ~5-10KB range without obvious
        # blocking artifacts; we don't need perfection here.
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        written.append(path)

    for name in _NEGATIVE_NAMES:
        img = _new_canvas(seed=900)
        _draw_street_scene(img)
        path = out_dir / name
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        written.append(path)

    return written


def main() -> int:
    default_out = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "images"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=default_out, help="Output directory")
    args = parser.parse_args()

    paths = generate(args.out)
    for p in paths:
        size_kb = p.stat().st_size / 1024
        print(f"wrote {p} ({size_kb:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
