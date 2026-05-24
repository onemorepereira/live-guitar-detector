"""Linear-probe classifier on top of SigLIP-2 image embeddings.

Sibling of :class:`app.probe_classifier.ProbeClassifier` — same API surface
(``classify(image_bgr) -> {brand, model, confidence}``) but uses Google's
``google/siglip2-base-patch16-256`` for the image encoder instead of
MobileCLIP-S1.

Why two backends:
  - MobileCLIP-S1 features (512-dim, ~50 MB IR) are fast (~5 ms / crop)
    but don't separate Gibson-vs-Fender body styles reliably — manual
    testing pinned val accuracy at ~20% (chance is 14%).
  - SigLIP-2 base (768-dim, ~370 MB model) is slower (~50-100 ms / crop
    on CPU) but clean-separates the same classes (100% val on Reverb
    listing crops — see chat session 2026-05-24).

The pipeline only calls `classify()` on a fraction of frames per track
(per-track scheduling in :mod:`app.tracks`), so even at 100 ms / call
the latency budget is preserved.

Probe artifact format matches :class:`ProbeClassifier`:
    .npz with keys: W (K, D), b (K,), labels (K, 2) UTF-8 strings.

Built and saved by ``scripts/train_probe.py --backend siglip``.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class SigLIPProbeClassifier:
    """Linear probe over SigLIP-2 image features.

    Loads the SigLIP-2 model + processor via HuggingFace ``transformers``
    on construction (one-time ~10 s) and caches them. The probe weights
    are loaded from ``probe_path`` (``.npz``).
    """

    def __init__(
        self,
        probe_path: Path,
        model_id: str = "google/siglip2-base-patch16-256",
    ) -> None:
        probe_path = Path(probe_path)
        if not probe_path.is_file():
            raise FileNotFoundError(f"probe head missing: {probe_path}")

        # Heavy imports kept inside the constructor so the rest of the
        # worker package stays importable in environments where
        # transformers/torch aren't available (e.g., the gateway).
        import torch
        from transformers import AutoModel, AutoProcessor

        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = AutoModel.from_pretrained(model_id)
        self._model.eval()

        # allow_pickle=False is deliberate — refuse pickle blobs.
        data = np.load(probe_path, allow_pickle=False)
        self._W: np.ndarray = data["W"].astype(np.float32)
        self._b: np.ndarray = data["b"].astype(np.float32)
        labels_arr = data["labels"]
        if labels_arr.ndim != 2 or labels_arr.shape[1] != 2:
            raise ValueError(f"probe labels must be shape (K, 2); got {labels_arr.shape}")
        self._labels: list[tuple[str, str]] = [
            (str(brand), str(model)) for brand, model in labels_arr
        ]

        if self._W.ndim != 2 or self._W.shape[0] != len(self._labels):
            raise ValueError(f"probe shape mismatch: W={self._W.shape} labels={len(self._labels)}")
        if self._b.shape != (len(self._labels),):
            raise ValueError(
                f"probe bias shape mismatch: b={self._b.shape} labels={len(self._labels)}"
            )

    @property
    def labels(self) -> list[tuple[str, str]]:
        return list(self._labels)

    @property
    def feature_dim(self) -> int:
        return int(self._W.shape[1])

    def classify(self, image_bgr: np.ndarray) -> dict:
        """Return ``{"brand", "model", "confidence"}`` for ``image_bgr``.

        ``image_bgr`` is a uint8 BGR ``HxWx3`` array (OpenCV convention).
        Internally converted to PIL RGB for the SigLIP processor.
        """
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError(f"expected HxWx3 BGR image, got shape {image_bgr.shape}")
        if image_bgr.dtype != np.uint8:
            raise ValueError(f"expected uint8 image, got dtype {image_bgr.dtype}")

        from PIL import Image  # local import for the same reason as above

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        inputs = self._processor(images=pil, return_tensors="pt")
        with self._torch.no_grad():
            feat = self._model.get_image_features(**inputs)
            feat = feat / feat.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        feat_np = feat[0].cpu().numpy().astype(np.float32)

        logits = self._W @ feat_np + self._b
        logits = logits - float(np.max(logits))
        probs = np.exp(logits)
        probs = probs / probs.sum()

        idx = int(np.argmax(probs))
        brand, model = self._labels[idx]
        return {"brand": brand, "model": model, "confidence": float(probs[idx])}
