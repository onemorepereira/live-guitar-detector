"""Linear-probe classifier on top of MobileCLIP-S1 image embeddings.

Trained offline via :mod:`scripts.train_probe` on a labeled dataset of
guitar crops; the resulting tiny (~30 KB) head replaces the zero-shot
text-prompt classifier in the runtime pipeline.

The class label space is baked into the probe artifact (the trained
``.npz`` carries a ``labels`` array of ``(brand, model)`` pairs) so the
worker can run without any prompts file when in probe mode.

API surface mirrors :class:`app.classifier.Classifier` exactly so the
pipeline and webcam runner can swap them with no other changes:
``classify(image_bgr) -> {"brand", "model", "confidence"}``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import openvino as ov

from app.classifier import preprocess_for_clip
from app.openvino_runtime import compile_config

# Float floor to keep L2-normalization safe on degenerate (near-zero)
# embeddings — the OpenVINO image tower never produces zero vectors in
# practice, but a defensive clip costs nothing and prevents NaN from
# infecting the linear head's logits.
_NORM_EPS = 1e-12


class ProbeClassifier:
    """Linear probe over MobileCLIP image features.

    Parameters mirror :class:`Classifier` for drop-in compatibility:
      - ``model_dir``  parent directory containing ``mobileclip-image-fp16/``
      - ``probe_path`` path to the trained probe ``.npz`` (keys: ``W``,
                       ``b``, ``labels``)
      - ``input_size`` image-tower input edge length (defaults to 224 to
                       match the exported IR)
    """

    def __init__(
        self,
        model_dir: Path,
        probe_path: Path,
        input_size: int = 224,
        device: str = "CPU",
        num_threads: int = 0,
    ) -> None:
        model_dir = Path(model_dir)
        probe_path = Path(probe_path)

        image_xml = model_dir / "mobileclip-image-fp16" / "image.xml"
        if not image_xml.is_file():
            raise FileNotFoundError(f"image tower IR missing: {image_xml}")
        if not probe_path.is_file():
            raise FileNotFoundError(f"probe head missing: {probe_path}")

        self._input_size = int(input_size)

        core = ov.Core()
        self._image_model = core.compile_model(
            str(image_xml), device_name=device, config=compile_config(num_threads)
        )
        self._image_output = self._image_model.output("image_features")

        # Load probe weights + labels. allow_pickle=False is deliberate:
        # we want to refuse pickle blobs (which could execute code) and
        # only accept plain numpy arrays.
        data = np.load(probe_path, allow_pickle=False)
        self._W: np.ndarray = data["W"].astype(np.float32)  # (K, D)
        self._b: np.ndarray = data["b"].astype(np.float32)  # (K,)
        labels_arr = data["labels"]  # (K, 2) of UTF-8 strings
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

    # ------------------------------------------------------------------
    # Public read-only attributes
    # ------------------------------------------------------------------

    @property
    def labels(self) -> list[tuple[str, str]]:
        """Defensive copy of the ``(brand, model)`` label set the probe knows."""
        return list(self._labels)

    @property
    def feature_dim(self) -> int:
        return int(self._W.shape[1])

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def classify(self, image_bgr: np.ndarray) -> dict:
        """Return ``{"brand", "model", "confidence"}`` for ``image_bgr``.

        ``image_bgr`` is a uint8 BGR ``HxWx3`` array (OpenCV convention).
        ``confidence`` is the softmax probability of the chosen class.
        """
        prepped = preprocess_for_clip(image_bgr, self._input_size)
        image_out = self._image_model(prepped)
        feat = image_out[self._image_output][0].astype(np.float32)
        feat = feat / max(float(np.linalg.norm(feat)), _NORM_EPS)

        logits = self._W @ feat + self._b  # (K,)
        # Numerically stable softmax.
        logits = logits - float(np.max(logits))
        probs = np.exp(logits)
        probs = probs / probs.sum()

        idx = int(np.argmax(probs))
        brand, model = self._labels[idx]
        return {"brand": brand, "model": model, "confidence": float(probs[idx])}
