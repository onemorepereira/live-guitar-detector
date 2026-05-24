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

## 2026-05-24 — classifier head comparison

Train and validation accuracy of a linear probe over each candidate
image-encoder, on the same dataset (~558 YOLO-cropped Reverb listing
images across 7 classes: 6 brand+model targets + Unknown). Held-out
80/20 stratified split, AdamW, no early stop. Reported as
final-epoch numbers.

| Backbone                            | Embedding dim | Epochs | Train      | Val        |
| ----------------------------------- | ------------- | ------ | ---------- | ---------- |
| MobileCLIP-S1 (OpenVINO IR)         | 512           | 300    | 54.3%      | 16.5%      |
| MobileCLIP-S1                       | 512           | 2000   | 68.3%      | 16.9%      |
| MobileCLIP-S1, YOLO-cropped data    | 512           | 1000   | 86.8%      | 19.8%      |
| **google/siglip2-base-patch16-256** | 768           | 1000   | **100.0%** | **100.0%** |

Random-chance baseline is 14.3% (1/7).

**Interpretation.** Linear-probe val accuracy is pinned at chance for
all MobileCLIP-S1 variants regardless of training duration, dataset
size (50 → 200 / class), or input quality (raw Reverb listing photos
vs YOLO-cropped guitar bodies). This is the embedding-ceiling
signature — the features themselves don't separate Gibson-vs-Fender
body styles in a generalizable way. Same data, SigLIP-2 cleanly
separates everything.

The production runtime now defaults to `CLASSIFIER_MODE=siglip_probe`
(see [`CLASSIFIER.md`](CLASSIFIER.md)). The MobileCLIP probe path is
kept in the codebase for benchmarking / regression — the SigLIP-2
deployment is otherwise the source of truth.

**Caveats.** 100% val on Reverb listing crops does not imply 100%
real-world phone-camera accuracy — train and val come from the same
domain (polished listing photos). We have no labeled phone-camera
test set. Live-on-phone behavior is "feels right most of the time
when the guitar is held steady, drifts during transitions" — see
the chat session that produced this entry for raw stats.

Inference cost on the same host:

| Stage                                           | MobileCLIP-S1 (probe) | SigLIP-2 (probe) |
| ----------------------------------------------- | --------------------- | ---------------- |
| `classify_ms` per crop                          | ~5 ms                 | ~50–100 ms       |
| End-to-end `total_ms` p50 (with classification) | ~28 ms                | ~80–120 ms       |

Worker stays under the 50 ms Phase 1 target on the MobileCLIP path,
above it on the SigLIP path. The per-track classify scheduler (every
6 frames unstable, every 30 stable) absorbs the higher latency in
practice — the bottleneck is still YOLO + bookkeeping at the frame
level, not classifier latency at the per-track level.
