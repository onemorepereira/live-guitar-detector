"""Download model weights and export to OpenVINO IR format.

This is a CLI utility for fetching the YOLOv8n-oiv7 detector and a MobileCLIP
classifier, then exporting both to OpenVINO IR. It is meant to be run once per
machine (or whenever models change) before starting the inference worker.

Usage:
    python scripts/download_models.py all --out app/models/
    python scripts/download_models.py download-yolo --out app/models/raw/
    python scripts/download_models.py export-yolo --in app/models/raw/yolov8n-oiv7.pt --out app/models/
    python scripts/download_models.py download-clip --out app/models/raw/
    python scripts/download_models.py export-clip --in app/models/raw/mobileclip-state.pt --out app/models/

Add --force to overwrite existing outputs (default: skip if present).

Notes / known fallbacks (see scripts/README.md for details):
  * YOLO INT8 quantization is currently DISABLED by default because the OIv7
    classes are not covered by the bundled `coco128.yaml` calibration set.
    Pass --yolo-int8 to attempt it (likely to fail without proper data).
  * MobileCLIP-S0 weights are not in OpenCLIP's public registry; we substitute
    MobileCLIP-S1 (smallest available). Override with --clip-model.
  * CLIP image-tower INT8 needs a calibration dataset; default is FP16 for
    both towers. INT8 will be a follow-up task.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from loguru import logger

# ---------------------------------------------------------------------------
# Defaults / constants
# ---------------------------------------------------------------------------

YOLO_WEIGHT = "yolov8n-oiv7.pt"
YOLO_EXPORT_DIRNAME = "yolov8n-oiv7"  # suffix -int8 or -fp32 appended at runtime

DEFAULT_CLIP_MODEL = "MobileCLIP-S1"
DEFAULT_CLIP_PRETRAINED = "datacompdr"
CLIP_STATE_FILENAME = "mobileclip-state.pt"
CLIP_IMAGE_DIRNAME = "mobileclip-image"  # suffix added at runtime
CLIP_TEXT_DIRNAME = "mobileclip-text"

CLIP_IMAGE_SIZE = 224
CLIP_TEXT_CTX = 77


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _write_precision_marker(out_dir: Path, precision: str, notes: str = "", **extra: str) -> None:
    """Drop a small precision.json next to the IR for downstream consumers.

    Arbitrary string-valued metadata can be appended via ``**extra`` (for
    example ``model_name`` / ``pretrained`` for CLIP) so a future operator can
    tell which weights a given IR came from without relying solely on the
    directory name.
    """
    info: dict[str, str] = {"precision": precision}
    if notes:
        info["notes"] = notes
    for key, value in extra.items():
        if value:
            info[key] = value
    (out_dir / "precision.json").write_text(json.dumps(info, indent=2) + "\n")


def _export_complete(target: Path) -> bool:
    """Return True if `target` contains a completed export (precision marker present).

    We use ``precision.json`` as the completion sentinel rather than the IR
    files (``.xml``/``.bin``) because a crashed export can leave the IR files
    partially written. The marker is the LAST file we write on success, so its
    presence reliably indicates the export finished. If a stale dir has
    ``.xml`` and ``.bin`` but no ``precision.json``, the next run will
    re-export.
    """
    return (target / "precision.json").is_file()


# ---------------------------------------------------------------------------
# YOLO
# ---------------------------------------------------------------------------


def cmd_download_yolo(out: Path, force: bool = False) -> Path:
    """Download YOLOv8n-oiv7 .pt weights into `out`. Returns path to the .pt."""
    _ensure_dir(out)
    target = out / YOLO_WEIGHT

    if target.exists() and not force:
        logger.info(
            f"YOLO weights already present at {target}; skipping download (use --force to redownload)"
        )
        return target

    logger.info(f"Downloading YOLO weights ({YOLO_WEIGHT})...")
    # Ultralytics downloads on instantiation; we then copy to our target path.
    from ultralytics import YOLO

    t0 = time.time()
    model = YOLO(YOLO_WEIGHT)
    # `model.ckpt_path` is the path Ultralytics downloaded to.
    src = Path(getattr(model, "ckpt_path", "") or "")
    if not src.exists():
        # Fallback: Ultralytics sometimes leaves the file in cwd.
        candidate = Path.cwd() / YOLO_WEIGHT
        if candidate.exists():
            src = candidate
    if not src.exists():
        raise RuntimeError(
            f"Could not locate downloaded YOLO weights (expected at {src or YOLO_WEIGHT})"
        )
    if src.resolve() != target.resolve():
        shutil.copy2(src, target)
    logger.info(
        f"Saved YOLO weights to {target} ({target.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.1f}s"
    )
    return target


def cmd_export_yolo(in_path: Path, out: Path, int8: bool = False, force: bool = False) -> Path:
    """Export YOLO .pt to OpenVINO IR. Returns the output IR directory.

    int8=True attempts INT8 quantization (requires a calibration dataset YAML
    that covers the model's classes; for OIv7 this is non-trivial).
    Default int8=False -> FP32 IR (OpenVINO default precision for export).
    """
    precision = "int8" if int8 else "fp32"
    suffix = "-int8" if int8 else "-fp32"
    final_dir = out / f"{YOLO_EXPORT_DIRNAME}{suffix}"

    if _export_complete(final_dir) and not force:
        logger.info(
            f"YOLO IR already present at {final_dir}; skipping export (use --force to re-export)"
        )
        return final_dir

    _ensure_dir(out)
    if not in_path.exists():
        raise FileNotFoundError(f"YOLO weights not found: {in_path}")

    logger.info(f"Exporting {in_path.name} -> OpenVINO IR (precision={precision})...")
    from ultralytics import YOLO

    t0 = time.time()
    model = YOLO(str(in_path))
    export_kwargs = {"format": "openvino", "imgsz": 416}
    if int8:
        export_kwargs["int8"] = True
        export_kwargs["data"] = "coco128.yaml"
        logger.warning(
            "Attempting INT8 export with coco128.yaml — OIv7 classes are NOT in COCO; "
            "calibration is unlikely to be meaningful. Consider --yolo-int8=false."
        )

    try:
        exported = model.export(**export_kwargs)
    except Exception as e:
        if int8:
            logger.warning(f"INT8 export failed ({e!r}); falling back to FP32 export.")
            precision = "fp32"
            suffix = "-fp32"
            final_dir = out / f"{YOLO_EXPORT_DIRNAME}{suffix}"
            if _export_complete(final_dir) and not force:
                return final_dir
            exported = model.export(format="openvino", imgsz=416)
        else:
            raise

    # Ultralytics writes to <weights_dir>/<name>_openvino_model/
    exported_path = Path(exported) if exported else None
    if exported_path is None or not exported_path.exists():
        # Locate any *_openvino_model dir next to the input .pt.
        candidates = list(in_path.parent.glob("*_openvino_model"))
        if not candidates:
            raise RuntimeError("OpenVINO export did not produce an output directory")
        exported_path = candidates[0]

    # Move/rename to our canonical layout.
    if final_dir.exists():
        shutil.rmtree(final_dir)
    shutil.move(str(exported_path), str(final_dir))
    _write_precision_marker(
        final_dir,
        precision,
        notes="INT8 deferred — no OIv7-aligned calibration set" if not int8 else "",
        model_name="yolov8n-oiv7",
    )
    logger.info(f"YOLO IR written to {final_dir} in {time.time() - t0:.1f}s")
    return final_dir


# ---------------------------------------------------------------------------
# CLIP (MobileCLIP)
# ---------------------------------------------------------------------------


def _resolve_clip_pretrained(model_name: str, requested: str) -> tuple[str, str]:
    """Pick the closest available (model, pretrained) pair in OpenCLIP."""
    import open_clip

    pretrained_pairs = open_clip.list_pretrained()
    # Exact match first.
    if (model_name, requested) in pretrained_pairs:
        return model_name, requested
    # Same model, any pretrained.
    same_model = [(m, p) for (m, p) in pretrained_pairs if m == model_name]
    if same_model:
        chosen = same_model[0]
        logger.warning(
            f"Requested pretrained='{requested}' not found for {model_name}; using {chosen}"
        )
        return chosen
    # Substitute MobileCLIP-S1 if user asked for S0.
    if model_name == "MobileCLIP-S0":
        fallback = [(m, p) for (m, p) in pretrained_pairs if m == "MobileCLIP-S1"]
        if fallback:
            chosen = fallback[0]
            logger.warning(
                f"MobileCLIP-S0 not in OpenCLIP registry; substituting {chosen}. "
                "See scripts/README.md for details."
            )
            return chosen
    raise RuntimeError(
        f"No OpenCLIP pretrained weights available for {model_name}. "
        f"Available MobileCLIP variants: "
        f"{[p for p in pretrained_pairs if 'obile' in p[0].lower()]}"
    )


def cmd_download_clip(
    out: Path,
    model_name: str = DEFAULT_CLIP_MODEL,
    pretrained: str = DEFAULT_CLIP_PRETRAINED,
    force: bool = False,
) -> Path:
    """Download MobileCLIP weights via OpenCLIP and save the state dict.

    Returns the path to the saved .pt file.
    """
    _ensure_dir(out)
    target = out / CLIP_STATE_FILENAME
    if target.exists() and not force:
        logger.info(
            f"CLIP weights already present at {target}; skipping download (use --force to redownload)"
        )
        return target

    import open_clip
    import torch

    actual_model, actual_pretrained = _resolve_clip_pretrained(model_name, pretrained)
    logger.info(
        f"Downloading CLIP weights: model={actual_model}, pretrained={actual_pretrained}..."
    )
    t0 = time.time()
    model, _, _ = open_clip.create_model_and_transforms(actual_model, pretrained=actual_pretrained)
    model.eval()

    # Persist the resolved identifiers alongside the state dict so the export
    # step can reconstruct the model architecture without guessing.
    payload = {
        "state_dict": model.state_dict(),
        "model_name": actual_model,
        "pretrained": actual_pretrained,
    }
    torch.save(payload, target)
    logger.info(
        f"Saved CLIP weights to {target} ({target.stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.1f}s"
    )
    return target


def _export_clip_image_tower(model, out_dir: Path, precision: str) -> Path:
    """Export the CLIP image tower to OpenVINO IR.

    For FP16 we let openvino.convert_model emit FP16 IR. INT8 is intentionally
    not implemented here (would need a calibration dataset).
    """
    import openvino as ov
    import torch

    _ensure_dir(out_dir)

    class ImageEncoder(torch.nn.Module):
        def __init__(self, clip):
            super().__init__()
            self.clip = clip

        def forward(self, image):
            return self.clip.encode_image(image)

    wrapper = ImageEncoder(model).eval()
    dummy = torch.randn(1, 3, CLIP_IMAGE_SIZE, CLIP_IMAGE_SIZE)

    onnx_path = out_dir / "image.onnx"
    logger.info(f"Tracing image tower -> {onnx_path}")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            str(onnx_path),
            input_names=["image"],
            output_names=["image_features"],
            opset_version=17,
            dynamic_axes={"image": {0: "batch"}, "image_features": {0: "batch"}},
        )

    logger.info(f"Converting image ONNX -> OpenVINO IR (precision={precision})")
    ov_model = ov.convert_model(str(onnx_path))
    xml_path = out_dir / "image.xml"
    compress = precision == "fp16"
    ov.save_model(ov_model, str(xml_path), compress_to_fp16=compress)
    # Clean up intermediate ONNX (and any external-data sidecar) to keep disk
    # usage low — torch.onnx may emit "<name>.onnx.data" alongside the .onnx
    # for models above the 2GB protobuf limit (or just opportunistically).
    for stale in out_dir.glob("image.onnx*"):
        stale.unlink(missing_ok=True)
    return xml_path


def _export_clip_text_tower(model, out_dir: Path, precision: str) -> Path:
    """Export the CLIP text tower to OpenVINO IR (FP16 by default)."""
    import openvino as ov
    import torch

    _ensure_dir(out_dir)

    class TextEncoder(torch.nn.Module):
        def __init__(self, clip):
            super().__init__()
            self.clip = clip

        def forward(self, text):
            return self.clip.encode_text(text)

    wrapper = TextEncoder(model).eval()
    dummy = torch.zeros(1, CLIP_TEXT_CTX, dtype=torch.long)

    onnx_path = out_dir / "text.onnx"
    logger.info(f"Tracing text tower -> {onnx_path}")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            str(onnx_path),
            input_names=["text"],
            output_names=["text_features"],
            opset_version=17,
            dynamic_axes={"text": {0: "batch"}, "text_features": {0: "batch"}},
        )

    logger.info(f"Converting text ONNX -> OpenVINO IR (precision={precision})")
    ov_model = ov.convert_model(str(onnx_path))
    xml_path = out_dir / "text.xml"
    compress = precision == "fp16"
    ov.save_model(ov_model, str(xml_path), compress_to_fp16=compress)
    for stale in out_dir.glob("text.onnx*"):
        stale.unlink(missing_ok=True)
    return xml_path


def cmd_export_clip(
    in_path: Path,
    out: Path,
    image_int8: bool = False,
    force: bool = False,
) -> tuple[Path, Path]:
    """Export both CLIP towers to OpenVINO IR.

    Per DESIGN.md §5.4: image tower INT8 (preferred) + text tower FP16. INT8 is
    deferred for now (no calibration dataset wired up) — both towers default to
    FP16. Returns (image_dir, text_dir).
    """
    image_precision = "int8" if image_int8 else "fp16"
    text_precision = "fp16"
    image_dir = out / f"{CLIP_IMAGE_DIRNAME}-{image_precision}"
    text_dir = out / f"{CLIP_TEXT_DIRNAME}-{text_precision}"

    if _export_complete(image_dir) and _export_complete(text_dir) and not force:
        logger.info(
            f"CLIP IRs already present at {image_dir} and {text_dir}; "
            "skipping export (use --force to re-export)"
        )
        return image_dir, text_dir

    # Fail fast with an actionable message if the optional [export] extras
    # aren't installed. Without onnxscript, torch.onnx.export blows up ~50
    # lines deep with a much less obvious error.
    try:
        import onnxscript  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "CLIP export requires the optional [export] extras. "
            "Install with: pip install -e '.[export]'"
        ) from exc

    if not in_path.exists():
        raise FileNotFoundError(f"CLIP weights not found: {in_path}")

    if image_int8:
        logger.warning(
            "INT8 for CLIP image tower is not implemented (needs calibration data). "
            "Falling back to FP16. See scripts/README.md."
        )
        image_precision = "fp16"
        image_dir = out / f"{CLIP_IMAGE_DIRNAME}-{image_precision}"

    import open_clip
    import torch

    payload = torch.load(in_path, map_location="cpu", weights_only=False)
    model_name = payload["model_name"]
    pretrained = payload["pretrained"]
    logger.info(f"Reconstructing {model_name} ({pretrained}) for export...")
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=None)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    t0 = time.time()
    _export_clip_image_tower(model, image_dir, image_precision)
    _write_precision_marker(
        image_dir,
        image_precision,
        notes="INT8 deferred — needs calibration dataset" if not image_int8 else "",
        model_name=model_name,
        pretrained=pretrained,
    )

    _export_clip_text_tower(model, text_dir, text_precision)
    _write_precision_marker(
        text_dir,
        text_precision,
        model_name=model_name,
        pretrained=pretrained,
    )

    logger.info(f"CLIP IRs written to {image_dir} and {text_dir} in {time.time() - t0:.1f}s")
    return image_dir, text_dir


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def cmd_all(
    out: Path,
    force: bool = False,
    yolo_int8: bool = False,
    clip_image_int8: bool = False,
    clip_model: str = DEFAULT_CLIP_MODEL,
    clip_pretrained: str = DEFAULT_CLIP_PRETRAINED,
) -> None:
    """Run download + export for both models into one output directory."""
    raw_dir = out / "raw"
    t0 = time.time()
    logger.info(f"Output root: {out.resolve()}")

    # Track which sub-steps actually did work versus were short-circuited by
    # the idempotency checks, so the final summary can make that explicit.
    # Each check probes the same condition the corresponding cmd_* function
    # uses to decide whether to skip.
    yolo_pt_path = raw_dir / YOLO_WEIGHT
    yolo_ir_dir = out / f"{YOLO_EXPORT_DIRNAME}{'-int8' if yolo_int8 else '-fp32'}"
    clip_pt_path = raw_dir / CLIP_STATE_FILENAME
    clip_image_dir = out / f"{CLIP_IMAGE_DIRNAME}-{'int8' if clip_image_int8 else 'fp16'}"
    clip_text_dir = out / f"{CLIP_TEXT_DIRNAME}-fp16"

    yolo_dl_skipped = yolo_pt_path.exists() and not force
    yolo_ex_skipped = _export_complete(yolo_ir_dir) and not force
    clip_dl_skipped = clip_pt_path.exists() and not force
    clip_ex_skipped = (
        _export_complete(clip_image_dir) and _export_complete(clip_text_dir) and not force
    )

    yolo_pt = cmd_download_yolo(raw_dir, force=force)
    cmd_export_yolo(yolo_pt, out, int8=yolo_int8, force=force)

    clip_pt = cmd_download_clip(
        raw_dir, model_name=clip_model, pretrained=clip_pretrained, force=force
    )
    cmd_export_clip(clip_pt, out, image_int8=clip_image_int8, force=force)

    def _tag(skipped: bool) -> str:
        return "skipped" if skipped else "ran"

    steps = (
        f"YOLO download ({_tag(yolo_dl_skipped)}), "
        f"YOLO export ({_tag(yolo_ex_skipped)}), "
        f"CLIP download ({_tag(clip_dl_skipped)}), "
        f"CLIP export ({_tag(clip_ex_skipped)})"
    )
    logger.info(f"All done in {time.time() - t0:.1f}s. Steps: {steps}.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download + export YOLOv8n-oiv7 and MobileCLIP to OpenVINO IR.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing outputs (default: skip if present).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_dl_yolo = sub.add_parser("download-yolo", help="Download YOLOv8n-oiv7 .pt weights")
    p_dl_yolo.add_argument("--out", type=Path, required=True, help="Directory for the .pt file")

    p_ex_yolo = sub.add_parser("export-yolo", help="Export YOLO .pt to OpenVINO IR")
    p_ex_yolo.add_argument("--in", dest="in_path", type=Path, required=True, help="Path to .pt")
    p_ex_yolo.add_argument("--out", type=Path, required=True, help="Output dir")
    p_ex_yolo.add_argument("--int8", action="store_true", help="Attempt INT8 quantization")

    p_dl_clip = sub.add_parser("download-clip", help="Download MobileCLIP weights via OpenCLIP")
    p_dl_clip.add_argument("--out", type=Path, required=True, help="Directory for the .pt file")
    p_dl_clip.add_argument("--model", default=DEFAULT_CLIP_MODEL, help="OpenCLIP model name")
    p_dl_clip.add_argument(
        "--pretrained", default=DEFAULT_CLIP_PRETRAINED, help="OpenCLIP pretrained tag"
    )

    p_ex_clip = sub.add_parser("export-clip", help="Export CLIP towers to OpenVINO IR")
    p_ex_clip.add_argument(
        "--in", dest="in_path", type=Path, required=True, help="Path to CLIP state .pt"
    )
    p_ex_clip.add_argument("--out", type=Path, required=True, help="Output dir")
    p_ex_clip.add_argument(
        "--image-int8", action="store_true", help="Attempt INT8 for image tower (deferred)"
    )

    p_all = sub.add_parser("all", help="Download and export both models")
    p_all.add_argument("--out", type=Path, required=True, help="Output root (raw/ subdir created)")
    p_all.add_argument(
        "--yolo-int8", action="store_true", help="Attempt YOLO INT8 (likely to fail; deferred)"
    )
    p_all.add_argument(
        "--clip-image-int8", action="store_true", help="Attempt CLIP image INT8 (deferred)"
    )
    p_all.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL, help="OpenCLIP model name")
    p_all.add_argument(
        "--clip-pretrained", default=DEFAULT_CLIP_PRETRAINED, help="OpenCLIP pretrained tag"
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Loguru: keep it simple, info+ to stderr.
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | {message}",
    )

    try:
        if args.cmd == "download-yolo":
            cmd_download_yolo(args.out, force=args.force)
        elif args.cmd == "export-yolo":
            cmd_export_yolo(args.in_path, args.out, int8=args.int8, force=args.force)
        elif args.cmd == "download-clip":
            cmd_download_clip(
                args.out,
                model_name=args.model,
                pretrained=args.pretrained,
                force=args.force,
            )
        elif args.cmd == "export-clip":
            cmd_export_clip(args.in_path, args.out, image_int8=args.image_int8, force=args.force)
        elif args.cmd == "all":
            cmd_all(
                args.out,
                force=args.force,
                yolo_int8=args.yolo_int8,
                clip_image_int8=args.clip_image_int8,
                clip_model=args.clip_model,
                clip_pretrained=args.clip_pretrained,
            )
        else:  # pragma: no cover — argparse will already have rejected this.
            parser.error(f"Unknown command: {args.cmd}")
    except Exception as e:
        logger.exception(f"Command {args.cmd!r} failed: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
