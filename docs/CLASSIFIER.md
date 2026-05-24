# Classifier modes

The inference worker can run one of three classifiers, selected via
the `CLASSIFIER_MODE` environment variable. They all consume a YOLO
bounding-box crop and emit a `{brand, model, confidence}` dict
through the same `classify()` interface, so the rest of the pipeline
(tracker, voter, gateway) doesn't care which is loaded.

| `CLASSIFIER_MODE`     | Backbone                        | Trained?                          | Latency / crop (CPU) | Where it shines                                                                                         |
| --------------------- | ------------------------------- | --------------------------------- | -------------------- | ------------------------------------------------------------------------------------------------------- |
| `zero_shot` (default) | MobileCLIP-S1 (image+text)      | no — uses `docs/prompts.md`       | ~5 ms                | Quick demo with no setup; survives a fresh checkout                                                     |
| `probe`               | MobileCLIP-S1 (image only)      | yes — `probe.npz` (~30 KB)        | ~5 ms                | Same backbone as zero_shot, but a learned head — kept for reproducibility / regression testing          |
| `siglip_probe`        | google/siglip2-base-patch16-256 | yes — `probe_siglip.npz` (~25 KB) | ~50–100 ms           | **Production**: the only mode that reliably separates the 6 target brand/models on Reverb-quality crops |

Both probe modes embed their label set into the trained `.npz`
artifact, so `docs/prompts.md` is only needed in `zero_shot` mode.

## Why three modes (the short version)

The original design (DESIGN.md §5.4) prescribed **zero-shot CLIP with
text prompts**. Manual phone testing surfaced that this caps at
classifying nearly every guitar crop as `Unknown` — the rejection
prompts (acoustic / bass) dominate the softmax. Dropping rejection
prompts hands the win to random fine-grained guitar prompts, with
similar uselessness.

A linear-probe head over MobileCLIP image embeddings turns out to be
no better — the embedding space itself doesn't separate Gibson-vs-
Fender body styles. `docs/BENCHMARKS.md` has the numbers. The signal
plateaus around chance regardless of training set size or epoch count.

SigLIP-2 base (`google/siglip2-base-patch16-256`) has stronger
fine-grained features. The exact same training script + dataset (just
swap `--backend clip` for `--backend siglip`) crosses 100% val
accuracy on the Reverb listing crops. Real-world phone-camera
accuracy is lower but visibly works.

All three implementations are preserved so the experiment is
reproducible and so a fresh checkout has a working default
(`zero_shot`) without needing a trained head.

## Choosing a mode at runtime

```bash
# Smoke / demo (no probe artifact needed):
CLASSIFIER_MODE=zero_shot python -m app.main

# MobileCLIP probe (mostly kept for benchmarking):
CLASSIFIER_MODE=probe \
    PROBE_PATH=./app/models/classifier-probe/probe.npz \
    python -m app.main

# Production:
CLASSIFIER_MODE=siglip_probe \
    SIGLIP_PROBE_PATH=./app/models/classifier-probe/probe_siglip.npz \
    python -m app.main
```

`docker-compose.yml` defaults the worker to `siglip_probe` and mounts
`./services/inference-worker/app/models/classifier-probe` so the
trained head is picked up from the host. Build via
[`make build-images`](../Makefile) bakes SigLIP-2 weights into the
worker image so no network calls happen at runtime.

## Training (or re-training) a probe

See [`services/inference-worker/scripts/TRAIN_PROBE.md`](../services/inference-worker/scripts/TRAIN_PROBE.md)
for the full guide. Short version:

```bash
cd services/inference-worker
source .venv/bin/activate

# Pull labeled images from the reverb-scrapper CSV export (one-off
# helper, gitignored, dev-only — see TRAIN_PROBE.md for alternatives).
python scripts/_pull_reverb_dataset.py \
    --csv ~/Extra/repos/personal/reverb-scrapper/_research/exports/listings_LATEST.csv \
    --out ./data --per-class 200

# YOLO-crop the listing photos so the probe sees the same kind of
# inputs the runtime pipeline does. About 44% of Reverb primary
# images survive this step (the rest are cases, headstock closeups,
# accessories, multi-guitar shots).
python scripts/_crop_dataset.py --in ./data --out ./data_crops

# Train (~10 seconds wall-clock on CPU; the slow part is embedding
# the ~600 crops, which takes ~50 s with SigLIP).
python scripts/train_probe.py \
    --backend siglip \
    --data-dir ./data_crops \
    --out ./app/models/classifier-probe/probe_siglip.npz

# Restart the worker to pick up the new head.
```

The trained `.npz` carries the canonical column ordering (one row
per `(brand, model)`) and a sibling `precision.json` with provenance
(backend, training set size per class, train/val accuracy).

## Honesty about val accuracy

`val_accuracy` in `precision.json` is measured on a stratified 80/20
split of the **same Reverb listing crops** the model was trained on.
It's a within-domain accuracy number — 100% there ≠ 100% on
phone-camera frames in real life. The right way to evaluate real
accuracy is to capture phone-camera crops, label them by hand, and
treat that as the real test set. We don't have such a dataset; the
production confidence numbers should be treated as "the probe's
guess" rather than "calibrated probability."

If labels are wrong systematically on a particular guitar shape,
remedies in order:

1. Confirm `CLASSIFIER_MODE=siglip_probe` (the other two modes are
   known-worse on this task; see DOCS/BENCHMARKS.md).
2. Add more `unknown/` training images so the probe can punt on
   ambiguous shapes rather than picking a wrong target.
3. Add more training crops for the under-performing brand/model.
4. Fine-tune the SigLIP-2 backbone itself (out of scope today;
   would need an unfrozen-last-block training pipeline).
