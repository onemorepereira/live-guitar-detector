"""Train a linear probe on top of MobileCLIP-S1 image embeddings.

The probe is a tiny `(K, D)` weight matrix plus a `(K,)` bias —
~30 KB for K=7 classes and D=512 features. Training is a multinomial
logistic regression over frozen CLIP features, run with AdamW for a
few hundred epochs on CPU; the whole job typically finishes in well
under a minute for the dataset sizes this project anticipates.

Expected dataset layout::

    data_dir/
      gibson_les_paul/
        photo1.jpg
        photo2.png
        ...
      gibson_sg/
      gibson_explorer/
      gibson_flying_v/
      fender_stratocaster/
      fender_telecaster/
      unknown/              # acoustic + bass + non-target electrics

Each subdirectory name must be a key in :data:`LABEL_MAP` (extend that
mapping when adding new classes). Subdirectories that aren't listed in
the map are silently skipped.

Output: a single ``.npz`` at the path passed via ``--out``, with arrays:
  - ``W``      (K, D) float32
  - ``b``      (K,)   float32
  - ``labels`` (K, 2) UTF-8 strings, ``[brand, model]`` per row

Plus a sibling ``precision.json`` with provenance (timestamp, sample
count per class, feature dimension) so it's traceable post hoc.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import openvino as ov
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger

from app.classifier import preprocess_for_clip

# Directory-name → (brand, model) mapping. Order in this dict defines
# the canonical column ordering of the trained probe.
LABEL_MAP: dict[str, tuple[str, str]] = {
    "gibson_les_paul": ("Gibson", "Les Paul"),
    "gibson_sg": ("Gibson", "SG"),
    "gibson_explorer": ("Gibson", "Explorer"),
    "gibson_flying_v": ("Gibson", "Flying V"),
    "fender_stratocaster": ("Fender", "Stratocaster"),
    "fender_telecaster": ("Fender", "Telecaster"),
    "unknown": ("Unknown", "Unknown"),
}

_VALID_IMAGE_GLOBS = ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG")


def _load_image_tower(models_dir: Path) -> tuple[ov.CompiledModel, ov.Output]:
    """Compile the MobileCLIP image tower and return (model, output_port)."""
    image_xml = models_dir / "mobileclip-image-fp16" / "image.xml"
    if not image_xml.is_file():
        raise FileNotFoundError(f"image tower IR missing: {image_xml}")
    core = ov.Core()
    model = core.compile_model(str(image_xml), device_name="CPU")
    return model, model.output("image_features")


def _embed_image(
    model: ov.CompiledModel,
    output: ov.Output,
    image_path: Path,
    input_size: int,
) -> np.ndarray:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"could not read {image_path}")
    prepped = preprocess_for_clip(img, input_size)
    feat = model(prepped)[output][0].astype(np.float32)
    feat = feat / max(float(np.linalg.norm(feat)), 1e-12)
    return feat


def _load_siglip(model_id: str):
    """Load SigLIP-2 model + processor via HuggingFace transformers."""
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model.eval()
    return model, processor


def _embed_images_siglip(
    model,
    processor,
    image_paths: list[Path],
    batch_size: int = 16,
) -> tuple[np.ndarray, list[Path]]:
    """Batch-embed images via SigLIP-2. Returns (features, paths_used)."""
    from PIL import Image

    feats: list[np.ndarray] = []
    used: list[Path] = []
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i : i + batch_size]
        pils = []
        for p in batch:
            try:
                pils.append((p, Image.open(p).convert("RGB")))
            except Exception as exc:
                logger.warning("    skipping {}: {}", p.name, exc)
        if not pils:
            continue
        inputs = processor(images=[img for _, img in pils], return_tensors="pt")
        with torch.no_grad():
            out = model.get_image_features(**inputs)
        out = out / out.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        feats.append(out.cpu().numpy().astype(np.float32))
        used.extend(p for p, _ in pils)
    if not feats:
        return np.zeros((0, 0), dtype=np.float32), []
    return np.concatenate(feats, axis=0), used


def _gather_images(class_dir: Path) -> list[Path]:
    """All image files under ``class_dir``, sorted for determinism."""
    out: list[Path] = []
    for pattern in _VALID_IMAGE_GLOBS:
        out.extend(class_dir.glob(pattern))
    return sorted(set(out))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="root with one subdir per class")
    parser.add_argument(
        "--models-dir",
        default=None,
        help="path containing mobileclip-image-fp16/ (only needed for --backend clip)",
    )
    parser.add_argument("--out", required=True, help="output .npz path for the trained probe")
    parser.add_argument(
        "--backend",
        choices=("clip", "siglip"),
        default="clip",
        help="image encoder: 'clip' uses the local MobileCLIP IR via OpenVINO, "
        "'siglip' loads google/siglip2-base-patch16-256 via HuggingFace "
        "transformers. SigLIP is slower (~50-100ms/crop on CPU vs ~5ms) "
        "but separates fine-grained brand/model far better.",
    )
    parser.add_argument(
        "--siglip-model-id",
        default="google/siglip2-base-patch16-256",
        help="HuggingFace model id when --backend=siglip",
    )
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="fraction of each class held out for validation accuracy "
        "(0 disables the split and reports train accuracy only)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_path = Path(args.out).resolve()
    models_dir = Path(args.models_dir).resolve() if args.models_dir else None
    if args.backend == "clip" and models_dir is None:
        logger.error("--backend=clip requires --models-dir (path to MobileCLIP IR)")
        return 1

    if not data_dir.is_dir():
        logger.error("data dir does not exist: {}", data_dir)
        return 1

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Discover classes (ordered as LABEL_MAP keys for stable column layout).
    classes = [c for c in LABEL_MAP if (data_dir / c).is_dir()]
    if not classes:
        logger.error("no recognized class subdirs under {}", data_dir)
        logger.error("expected subdirs (any subset): {}", sorted(LABEL_MAP))
        return 1
    logger.info("found {} classes: {}", len(classes), classes)

    # Compute embeddings for every training image. Backend choice gates
    # the loader; the downstream training + serialization is the same.
    feats: list[np.ndarray] = []
    labels: list[int] = []
    per_class_count: dict[str, int] = {}
    t_embed_start = time.time()

    if args.backend == "siglip":
        logger.info("loading SigLIP-2 model {}", args.siglip_model_id)
        siglip_model, siglip_processor = _load_siglip(args.siglip_model_id)
        for cls_idx, cls_name in enumerate(classes):
            cls_dir = data_dir / cls_name
            imgs = _gather_images(cls_dir)
            if not imgs:
                logger.warning("no images in {}", cls_dir)
                per_class_count[cls_name] = 0
                continue
            logger.info("  {}: {} images", cls_name, len(imgs))
            batch_feats, used_paths = _embed_images_siglip(siglip_model, siglip_processor, imgs)
            for f in batch_feats:
                feats.append(f)
                labels.append(cls_idx)
            per_class_count[cls_name] = len(used_paths)
        clip_loop = False
    else:
        clip_loop = True

    if clip_loop:
        logger.info("loading MobileCLIP image tower from {}", models_dir)
        model, output = _load_image_tower(models_dir)
        for cls_idx, cls_name in enumerate(classes):
            cls_dir = data_dir / cls_name
            imgs = _gather_images(cls_dir)
            if not imgs:
                logger.warning("no images in {}", cls_dir)
                per_class_count[cls_name] = 0
                continue
            logger.info("  {}: {} images", cls_name, len(imgs))
            used = 0
            for img_path in imgs:
                try:
                    feat = _embed_image(model, output, img_path, args.input_size)
                except Exception as exc:
                    logger.warning("    skipping {}: {}", img_path.name, exc)
                    continue
                feats.append(feat)
                labels.append(cls_idx)
                used += 1
        per_class_count[cls_name] = used
    logger.info("embeddings computed in {:.1f}s", time.time() - t_embed_start)

    if not feats:
        logger.error("no usable training images; aborting")
        return 1

    X = np.stack(feats).astype(np.float32)  # (N, D)
    y = np.array(labels, dtype=np.int64)  # (N,)
    n_samples, feature_dim = X.shape
    n_classes = len(classes)
    logger.info(
        "training data: N={} D={} K={} (per-class: {})",
        n_samples,
        feature_dim,
        n_classes,
        per_class_count,
    )

    # Per-class shuffle-then-split keeps validation balanced even when
    # classes are skewed (e.g., Explorer is capped at ~150 while
    # Stratocaster has thousands available). Stratified-by-construction.
    rng = np.random.default_rng(args.seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    if args.val_fraction > 0:
        for cls_idx in range(n_classes):
            cls_indices = np.where(y == cls_idx)[0]
            rng.shuffle(cls_indices)
            n_val = max(1, int(round(len(cls_indices) * args.val_fraction)))
            val_idx.extend(cls_indices[:n_val].tolist())
            train_idx.extend(cls_indices[n_val:].tolist())
    else:
        train_idx = list(range(n_samples))
    train_idx_arr = np.array(train_idx, dtype=np.int64)
    val_idx_arr = np.array(val_idx, dtype=np.int64) if val_idx else None
    logger.info(
        "split: train={} val={} (fraction={})",
        len(train_idx_arr),
        0 if val_idx_arr is None else len(val_idx_arr),
        args.val_fraction,
    )

    # Train a single linear layer with multinomial cross-entropy.
    head = nn.Linear(feature_dim, n_classes, bias=True)
    opt = optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    Xt_train = torch.from_numpy(X[train_idx_arr])
    yt_train = torch.from_numpy(y[train_idx_arr])
    Xt_val = torch.from_numpy(X[val_idx_arr]) if val_idx_arr is not None else None
    yt_val = torch.from_numpy(y[val_idx_arr]) if val_idx_arr is not None else None

    t_train_start = time.time()
    head.train()
    for epoch in range(args.epochs):
        opt.zero_grad()
        logits = head(Xt_train)
        loss = loss_fn(logits, yt_train)
        loss.backward()
        opt.step()
        if (epoch + 1) % 50 == 0 or epoch == args.epochs - 1:
            with torch.no_grad():
                train_acc_step = (logits.argmax(dim=1) == yt_train).float().mean().item()
                if Xt_val is not None and yt_val is not None:
                    val_acc_step = (head(Xt_val).argmax(dim=1) == yt_val).float().mean().item()
                    logger.info(
                        "  epoch {:4d}: loss={:.4f}  train={:.1f}%  val={:.1f}%",
                        epoch + 1,
                        loss.item(),
                        train_acc_step * 100,
                        val_acc_step * 100,
                    )
                else:
                    logger.info(
                        "  epoch {:4d}: loss={:.4f}  train_acc={:.1f}%",
                        epoch + 1,
                        loss.item(),
                        train_acc_step * 100,
                    )
    logger.info("trained in {:.1f}s", time.time() - t_train_start)

    head.eval()
    with torch.no_grad():
        train_acc = (head(Xt_train).argmax(dim=1) == yt_train).float().mean().item()
        val_acc = (
            (head(Xt_val).argmax(dim=1) == yt_val).float().mean().item()
            if Xt_val is not None and yt_val is not None
            else None
        )
    if val_acc is not None:
        logger.info("final train_acc={:.1f}%  val_acc={:.1f}%", train_acc * 100, val_acc * 100)
    else:
        logger.info("final train_acc={:.1f}%", train_acc * 100)
    logger.info("final train_acc={:.1f}%", train_acc * 100)

    W = head.weight.detach().numpy().astype(np.float32)  # (K, D)
    b = head.bias.detach().numpy().astype(np.float32)  # (K,)
    labels_arr = np.array(
        [list(LABEL_MAP[c]) for c in classes],
        dtype="<U64",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, W=W, b=b, labels=labels_arr)

    marker = out_path.parent / "precision.json"
    marker.write_text(
        json.dumps(
            {
                "head": "linear_probe",
                "backend": args.backend,
                "encoder": (
                    args.siglip_model_id if args.backend == "siglip" else "mobileclip-image-fp16"
                ),
                "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "n_classes": int(n_classes),
                "feature_dim": int(feature_dim),
                "training_set_size": int(n_samples),
                "per_class_count": per_class_count,
                "train_accuracy": float(train_acc),
                "val_accuracy": None if val_acc is None else float(val_acc),
                "val_fraction": float(args.val_fraction),
                "epochs": int(args.epochs),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info("saved probe to {} ({} bytes)", out_path, out_path.stat().st_size)
    logger.info("provenance: {}", marker)
    return 0


if __name__ == "__main__":
    sys.exit(main())
