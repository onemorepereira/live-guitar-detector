# Benchmarks

Recorded benchmark runs of the inference pipeline against the documented
performance gates. See `services/inference-worker/scripts/benchmark.py` for
the harness. Each section pastes the verbatim script output plus the host
details, so a later re-run on a different machine is directly comparable.

## 2026-05-22 — Phase 1 gate (initial)

**Host:** AMD Ryzen 9 7940HS w/ Radeon 780M Graphics
**Platform:** Linux-6.19.14-200.fc43.x86_64-x86_64-with-glibc2.42
**Python:** 3.11.11
**Frames:** 600 cycled through synthetic fixture images (6 images)
**Target FPS:** 15 (66.7 ms frame budget)

```
Benchmark: 600 frames at target 15.0 fps
Host:    unknown CPU
Platform: Linux-6.19.14-200.fc43.x86_64-x86_64-with-glibc2.42
Python:  3.11.11

  detect_ms    N= 600  p50= 11.09ms  p95= 15.96ms  p99= 21.03ms  mean= 11.30ms
  classify_ms  N=   0  (no samples)
  total_ms     N= 600  p50= 11.12ms  p95= 15.76ms  p99= 20.84ms  mean= 11.25ms

Dropped (total > 66.7ms): 0 / 600 (0.0%)
Unconfirmed-track skips: 0

Phase 1 gate PASSED: p50 total = 11.12ms < 50ms target
```

**Phase 1 gate (p50 total < 50ms): PASS** (11.12 ms, ~4.5× headroom).

**Notes:**

- DESIGN.md §8 targets a Ryzen 7 reference; this run is on a Ryzen 9 7940HS
  (Zen 4 mobile), which is faster, so the headroom here will compress on
  weaker hardware. We do not currently have a Ryzen 7 box to validate on;
  re-run the benchmark on the eventual deployment host before treating
  these numbers as deployment-ready.
- `platform.processor()` returns an empty string on this kernel/glibc; the
  Host field above was pulled from `/proc/cpuinfo` and added manually. The
  script's "unknown CPU" line is cosmetic and does not affect timings.
- `classify_ms` has zero samples because the synthetic fixtures
  (`tests/fixtures/images/*.jpg`) are placeholder shapes that the YOLO
  detector does not fire on — no detections means the per-stage classify
  timing was never reached. `total_ms` therefore reflects the detect-only
  fast path. Once real-photograph fixtures are committed (see
  `tests/fixtures/images/README.md`), re-run this benchmark; the
  `classify_ms` line will populate and `total_ms` will rise to include
  the CLIP forward pass plus voting cost. The Phase 1 gate must be
  re-verified at that point.
- The Task 1.9 webcam smoke test reported ~18 fps and ~28.6 ms average
  latency on the same host with a real camera feed. That number IS hitting
  the classifier on detected guitars, and is still well under the 50 ms
  target — so even with the classify cost added, we are not at risk of
  blowing the gate on this hardware.
- Dropped-frame count is 0/600: the pipeline comfortably keeps up with
  the 15 fps target on this host.
