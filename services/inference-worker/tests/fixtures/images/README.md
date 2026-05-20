# Detector test fixtures

The JPEGs in this directory are **synthetic placeholders**, not photographs
of guitars. They are produced by
`services/inference-worker/scripts/make_synthetic_fixtures.py` and exist
only so file discovery / I/O paths in `tests/test_detector.py` work without
a marker indicating real photos.

Tests that assert real detector behaviour (non-empty results on a
guitar image, empty results on a street scene, bbox bounds, ByteTrack ID
persistence, etc.) are gated behind the `@pytest.mark.requires_real_fixtures`
marker. Those tests **auto-skip** until a developer replaces the placeholders
with real photos and signals opt-in with a marker file.

## Filenames the test suite expects

| File               | Subject                                                   |
| ------------------ | --------------------------------------------------------- |
| `strat_01.jpg`     | Fender Stratocaster (clear, single guitar)                |
| `strat_02.jpg`     | Fender Stratocaster (different angle/lighting)            |
| `lp_01.jpg`        | Gibson Les Paul (clear, single guitar)                    |
| `lp_02.jpg`        | Gibson Les Paul (different angle/lighting)                |
| `acoustic_01.jpg`  | Any acoustic guitar (steel- or nylon-string)              |
| `street_scene.jpg` | A scene with **no guitars** (cityscape, plate of food, …) |

Aspect ratio doesn't matter; the detector resizes internally. ~640x480 is a
reasonable target.

## How to swap in real photos

1. Source images you may legally redistribute. Good options:
   - [Wikimedia Commons](https://commons.wikimedia.org/) — search for
     "Gibson Les Paul", "Fender Stratocaster", etc. and filter to CC0 / PD.
   - [Pexels](https://www.pexels.com/) (CC0-equivalent license).
   - Self-photographed images you own.
2. Rename to the filenames in the table above, save as JPEG, drop into this
   directory (overwrite the synthetic versions).
3. `touch services/inference-worker/tests/fixtures/images/REAL.txt`
   to enable the `requires_real_fixtures` tests on this machine. The marker
   file is intentionally not committed — each developer/CI runner opts in
   independently after curating their own copies.
4. `pytest -q services/inference-worker/tests/test_detector.py` and confirm
   the previously-skipped tests now pass.

If you commit real photos back to the repo, make sure their license permits
redistribution and add an attribution note here (filename → source URL →
license).
