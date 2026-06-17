"""MobileCLIP zero-shot classifier per DESIGN.md §5.4.

Loads pre-exported OpenVINO IRs for the MobileCLIP image and text towers,
precomputes text features for the prompt list once at construction time, and
provides :meth:`Classifier.classify` that runs the image tower + cosine
similarity for each incoming BGR frame crop.

Note: DESIGN.md names MobileCLIP-S0; we ship S1 because S0 is not in the
OpenCLIP registry (see ``scripts/README.md`` "Known follow-ups").
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import openvino as ov

from app.openvino_runtime import compile_config
from app.prompts import Prompt

# CLIP standard temperature: the trained model maps cosine similarities into
# logit space with a learned scale that converges near exp(4.6) ≈ 100. We use
# the canonical value rather than reading it from the IR because the exported
# graph only emits the projected text/image features, not the logit_scale
# parameter itself.
CLIP_LOGIT_SCALE = 100.0

# OpenAI CLIP normalization stats. MobileCLIP follows OpenCLIP's convention of
# using these defaults unless a checkpoint explicitly overrides them; the
# MobileCLIP-S1/datacompdr checkpoint we ship does not.
OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# Context length for the CLIP text tower. MobileCLIP keeps the default 77.
CLIP_CONTEXT_LENGTH = 77


class Classifier:
    """Zero-shot image classifier backed by two MobileCLIP OpenVINO IRs.

    The constructor:

    * Locates ``mobileclip-image-fp16/image.xml`` and
      ``mobileclip-text-fp16/text.xml`` under ``model_dir``.
    * Reads ``precision.json`` next to the image IR to pick the matching
      OpenCLIP tokenizer (default: ``MobileCLIP-S1`` / ``datacompdr``).
    * Tokenizes every prompt once, runs the text tower, and L2-normalizes
      each row of the resulting ``(N_prompts, D)`` matrix into
      ``self._text_features``. Per DESIGN.md §5.4 we never re-tokenize or
      re-run the text tower at inference time.

    :meth:`classify` runs the image tower on a single BGR uint8 frame and
    returns the prompt with the highest cosine similarity (after CLIP's
    standard temperature + softmax) as ``{"brand", "model", "confidence"}``.
    """

    def __init__(
        self,
        model_dir: Path,
        prompts: list[Prompt],
        input_size: int = 224,
        device: str = "CPU",
        num_threads: int = 0,
    ) -> None:
        if not prompts:
            # Empty prompt list would produce a (0, D) matrix; classify()
            # would then call np.argmax on an empty array and raise. Fail
            # fast at construction so the error points at the caller.
            raise ValueError("prompts list must not be empty")

        model_dir = Path(model_dir)
        self._input_size = int(input_size)
        self._prompts = list(prompts)

        image_xml = model_dir / "mobileclip-image-fp16" / "image.xml"
        text_xml = model_dir / "mobileclip-text-fp16" / "text.xml"
        if not image_xml.is_file():
            raise FileNotFoundError(f"image tower IR missing: {image_xml}")
        if not text_xml.is_file():
            raise FileNotFoundError(f"text tower IR missing: {text_xml}")

        # Read model identity from precision.json so we tokenize with the
        # matching model. Default to MobileCLIP-S1/datacompdr if absent so
        # ad-hoc test exports without metadata still load.
        marker = model_dir / "mobileclip-image-fp16" / "precision.json"
        meta = json.loads(marker.read_text(encoding="utf-8")) if marker.is_file() else {}
        self._model_name: str = meta.get("model_name", "MobileCLIP-S1")
        self._pretrained: str = meta.get("pretrained", "datacompdr")

        core = ov.Core()
        config = compile_config(num_threads)
        self._image_model = core.compile_model(str(image_xml), device_name=device, config=config)
        self._text_model = core.compile_model(str(text_xml), device_name=device, config=config)

        # Bind the single named output of each tower so we read it back by name
        # rather than by dict-iteration order. If a future re-export adds extra
        # outputs (e.g., intermediate features), this fails loudly at startup
        # instead of silently picking the wrong tensor.
        self._image_output = self._image_model.output("image_features")
        self._text_output = self._text_model.output("text_features")

        self._text_features = self._compute_text_features(self._prompts)

    # ------------------------------------------------------------------
    # Public read-only attributes
    # ------------------------------------------------------------------

    @property
    def prompts(self) -> list[Prompt]:
        """Defensive copy of the prompt list the classifier was built with."""
        return list(self._prompts)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def classify(self, image_bgr: np.ndarray) -> dict:
        """Return the most-similar prompt's ``{brand, model, confidence}``.

        ``image_bgr`` is a BGR uint8 ``HxWx3`` array (OpenCV convention).
        ``confidence`` is the softmax probability of the chosen prompt under
        CLIP's standard logit_scale=100 temperature, so it lives in ``[0, 1]``
        and the values across all prompts sum to 1.
        """
        prepped = self._preprocess(image_bgr)
        image_out = self._image_model(prepped)
        image_feat = image_out[self._image_output][0].astype(np.float32)
        image_feat = image_feat / max(float(np.linalg.norm(image_feat)), 1e-12)

        # Cosine similarity against the precomputed (and already-normalized)
        # text features: (N, D) @ (D,) -> (N,).
        sims = self._text_features @ image_feat
        logits = sims * CLIP_LOGIT_SCALE

        # Numerically stable softmax: subtract the max before exp.
        logits = logits - float(np.max(logits))
        probs = np.exp(logits)
        probs = probs / probs.sum()

        idx = int(np.argmax(probs))
        chosen = self._prompts[idx]
        return {
            "brand": chosen.brand,
            "model": chosen.model,
            "confidence": float(probs[idx]),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute_text_features(self, prompts: list[Prompt]) -> np.ndarray:
        """Tokenize + encode every prompt once; return an L2-normalized matrix.

        Imports ``open_clip`` lazily so callers that only ever construct a
        ``Classifier`` from a pre-baked feature matrix in some future test
        path don't pay the (substantial) torch+open_clip import cost. Today
        every constructor path goes through here, so the import is effectively
        eager — but the laziness is free.
        """
        import open_clip

        # Apple's MobileCLIP family reuses OpenAI CLIP's BPE vocabulary and
        # does not register a HuggingFace tokenizer in OpenCLIP, so
        # get_tokenizer() returns SimpleTokenizer — that is the canonical
        # tokenizer for these models, not a fallback.
        tokenizer = open_clip.get_tokenizer(self._model_name)

        feats: list[np.ndarray] = []
        for p in prompts:
            # Tokenizer returns a (1, 77) int64 torch tensor; OpenVINO needs
            # the same shape/dtype on the numpy side.
            tokens = tokenizer([p.text]).numpy().astype(np.int64)
            out = self._text_model(tokens)
            feat = out[self._text_output]  # (1, D)
            feats.append(np.asarray(feat[0], dtype=np.float32))

        stacked = np.stack(feats).astype(np.float32)  # (N, D)
        # L2-normalize each row so the @ in classify() is a true cosine.
        norms = np.linalg.norm(stacked, axis=1, keepdims=True)
        return stacked / np.clip(norms, 1e-12, None)

    def _preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        """Instance shim around :func:`preprocess_for_clip`."""
        return preprocess_for_clip(image_bgr, self._input_size)


def preprocess_for_clip(image_bgr: np.ndarray, input_size: int) -> np.ndarray:
    """BGR uint8 HxWx3 -> (1, 3, input_size, input_size) float32 CHW.

    Square-pads with edge replication before resizing so aspect ratio is
    preserved without injecting black bars (CLIP can latch onto those as
    a "studio" cue). Shared between :class:`Classifier` (zero-shot text
    matching) and :class:`ProbeClassifier` (linear head over the same
    image embeddings) so the train- and inference-time preprocessing are
    byte-identical.
    """
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"classifier expects HxWx3 BGR image, got shape {image_bgr.shape}")
    if image_bgr.dtype != np.uint8:
        # A float32 image already in [0, 1] would silently round-trip into
        # the [0, 1/255] range after the /255 step below and produce
        # garbage features. Reject upfront.
        raise ValueError(f"classifier expects uint8 image, got dtype {image_bgr.dtype}")

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    h, w = rgb.shape[:2]
    if h != w:
        target = max(h, w)
        top = (target - h) // 2
        bottom = target - h - top
        left = (target - w) // 2
        right = target - w - left
        rgb = cv2.copyMakeBorder(rgb, top, bottom, left, right, cv2.BORDER_REPLICATE)

    # INTER_AREA is the right downsampler when shrinking, INTER_CUBIC the
    # right upsampler when enlarging — OpenCV docs are explicit on this
    # and conflating them costs visible quality on small crops.
    interp = cv2.INTER_AREA if rgb.shape[0] >= input_size else cv2.INTER_CUBIC
    resized = cv2.resize(rgb, (input_size, input_size), interpolation=interp)

    arr = resized.astype(np.float32) / 255.0
    mean = np.array(OPENAI_CLIP_MEAN, dtype=np.float32)
    std = np.array(OPENAI_CLIP_STD, dtype=np.float32)
    arr = (arr - mean) / std

    # HWC -> CHW, add batch dim. ``np.ascontiguousarray`` makes the
    # buffer C-contiguous after transpose so OpenVINO doesn't have to
    # copy internally.
    chw = np.ascontiguousarray(arr.transpose(2, 0, 1))
    return chw[np.newaxis, ...]
