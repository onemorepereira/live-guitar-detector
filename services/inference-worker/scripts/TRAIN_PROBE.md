# Linear-probe classifier — training guide

The runtime worker can use one of three classifiers (selected by the
`CLASSIFIER_MODE` env var):

- `zero_shot` (default) — cosine similarity of MobileCLIP-S1 image
  features against text prompts from `docs/prompts.md`. No training
  required; accuracy is bounded by how well the prompts capture each
  class. **Known-poor on the 6 target brand/models.**
- `probe` — linear head over the same MobileCLIP-S1 image features.
  Kept for benchmarking; **same accuracy ceiling as zero_shot** on
  these classes (see [`docs/BENCHMARKS.md`](../../../docs/BENCHMARKS.md)).
- `siglip_probe` — linear head over `google/siglip2-base-patch16-256`
  image features. **Production choice**: separates the targets
  cleanly. Slower per-crop (~50–100 ms vs ~5 ms) but the per-track
  classify scheduler absorbs it.

This guide covers training and deploying the probe heads. The
backend is selected via `--backend {clip,siglip}` on the trainer; the
script reuses the same dataset layout and produces the same `.npz`
artifact format either way.

See [`docs/CLASSIFIER.md`](../../../docs/CLASSIFIER.md) for a
higher-level mode comparison.

## TL;DR (production — SigLIP probe)

```bash
# 1. Collect ~200 labeled crops per class under data/
data/
  gibson_les_paul/      *.jpg
  gibson_sg/            *.jpg
  gibson_explorer/      *.jpg
  gibson_flying_v/      *.jpg
  fender_stratocaster/  *.jpg
  fender_telecaster/    *.jpg
  unknown/              *.jpg   # acoustic / bass / other electrics

# 2. (Optional but recommended) YOLO-crop the dataset so the probe
#    learns from the same crop quality the runtime pipeline produces.
#    See the chat session 2026-05-24 for the rationale — about 56% of
#    Reverb listing primary images are cases / closeups / non-body
#    shots, and the probe's val accuracy depends on filtering those.
python scripts/_crop_dataset.py --in ./data --out ./data_crops

# 3. Train SigLIP probe (~50s embed + <1s linear-head training)
cd services/inference-worker
source .venv/bin/activate
python scripts/train_probe.py \
  --backend siglip \
  --data-dir ./data_crops \
  --out ./app/models/classifier-probe/probe_siglip.npz

# 4. Run the worker in SigLIP probe mode
CLASSIFIER_MODE=siglip_probe \
SIGLIP_PROBE_PATH=./app/models/classifier-probe/probe_siglip.npz \
REDIS_URL=redis://localhost:6379/0 \
  python -m app.main
```

For the legacy MobileCLIP `probe` mode, swap `--backend clip` (the
trainer's default) and pass `--models-dir ./app/models` so it can
find the MobileCLIP IR. The runtime env vars become `CLASSIFIER_MODE=probe`
and `PROBE_PATH=./app/models/classifier-probe/probe.npz`.

## Dataset

The probe needs **labeled crops of guitars** — ideally the bounding-box
output of the YOLO detector, but unstructured product photos work fine
too. As a rule of thumb, **50-100 images per class** is enough for a
useful first probe; more (200-500/class) gets meaningfully better.

Each `<class>` subdirectory under `--data-dir` maps to one
`(brand, model)` pair via `LABEL_MAP` at the top of
`scripts/train_probe.py`. Add new entries there if you extend the
class set.

The `unknown/` class is critical — fill it with images the classifier
should reject (acoustic guitars, basses, blank backgrounds, etc.). The
trained probe will return `(Unknown, Unknown)` for these and the
downstream voting will keep the HUD in `Analyzing…` state. Without an
Unknown class, the probe is forced to pick one of the brand/model
options every time.

**Where to get data:**

- Wikimedia Commons (CC-BY / CC0) — search for the model name.
- Reverb listings — listing thumbnails are usually crop-friendly.
- Manufacturer product pages (Gibson.com, Fender.com).
- Your own phone photos — the same lens that will be used at inference
  time, which helps a lot with domain match.

Try to vary lighting, angle, and background; avoid 50 nearly-identical
press photos of the same instrument.

## Training

```
python scripts/train_probe.py \
  --data-dir <DIR> \
  --models-dir <DIR> \
  --out <OUT.NPZ> \
  [--input-size 224] \
  [--epochs 300] \
  [--lr 0.05] \
  [--weight-decay 1e-4] \
  [--seed 0]
```

Defaults are tuned for the small-dataset regime (a few hundred images
total). On CPU the whole job takes under a minute for ~500 images.

Outputs:

- `<OUT.NPZ>` — the trained head (W, b, labels).
- `<OUT_DIR>/precision.json` — provenance (timestamp, per-class
  sample counts, train accuracy, epoch count).

## Deploying

Two env vars on the worker:

| Var               | Default                              | Effect                                                                                       |
| ----------------- | ------------------------------------ | -------------------------------------------------------------------------------------------- |
| `CLASSIFIER_MODE` | `zero_shot`                          | Set to `probe` to use the linear head                                                        |
| `PROBE_PATH`      | `/models/classifier-probe/probe.npz` | Container path; local-dev runs fall back to `<worker>/app/models/classifier-probe/probe.npz` |

The probe head is hot — restart the worker to pick up a new training
artifact.

## Iterating

1. Train an initial probe with whatever data you can scrounge.
2. Run the dev stack, point a camera at real guitars.
3. Note which classes the probe is bad at (the 5-second `classifier=[…]`
   stats line shows the breakdown).
4. Add more images for the weakest classes, retrain, restart the worker.
5. Repeat.

Two failure modes to watch for:

- **Probe always picks the most common class.** Class imbalance —
  ensure each class has a roughly comparable image count, or weight
  classes in the loss (a follow-up: add `--class-weights` to the
  training script).
- **Probe agrees with itself on the training set but fails live.**
  Overfitting on a small dataset. Add more diverse images, lower
  `--epochs`, or raise `--weight-decay`.

## Limitations

- The probe is bounded by what the **frozen MobileCLIP-S1 image
  features can separate**. Two guitars whose visual signatures fall
  near each other in the embedding space (e.g., Les Paul vs Custom)
  may be inseparable regardless of training data size. Escalation
  path: fine-tune the last block of CLIP itself (out of scope here).
- The probe inherits MobileCLIP's preprocessing — crops are square-
  padded with edge replication, resized to 224×224, normalised with
  OpenAI CLIP mean/std. Training and inference share the exact same
  pipeline (`preprocess_for_clip` in `app/classifier.py`).
