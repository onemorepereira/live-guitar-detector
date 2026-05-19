"""Rolling-window weighted voting for per-track label smoothing.

Implements the algorithm described in DESIGN.md §5.4: each frame's raw
classification result for a track is fed in via :meth:`TrackVote.update`,
and :meth:`TrackVote.current` returns the confidence-weighted majority
label, the smoothed confidence (winner weight / total weight), and a
``stable`` flag indicating whether the track has converged.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass(frozen=True)
class SmoothedLabel:
    label: dict | None  # {"brand": str, "model": str} or None
    confidence: float  # 0..1
    stable: bool
    samples: int


class TrackVote:
    def __init__(self, window: int, stable_min: int, stable_conf: float):
        self._buf: deque[tuple[str, str, float]] = deque(maxlen=window)
        self._stable_min = stable_min
        self._stable_conf = stable_conf

    def update(self, raw: dict) -> None:
        self._buf.append((raw["brand"], raw["model"], float(raw["confidence"])))

    def current(self) -> SmoothedLabel:
        if not self._buf:
            return SmoothedLabel(None, 0.0, False, 0)
        weights: dict[tuple[str, str], float] = defaultdict(float)
        total = 0.0
        for brand, model, conf in self._buf:
            weights[(brand, model)] += conf
            total += conf
        (brand, model), w = max(weights.items(), key=lambda kv: kv[1])
        smoothed_conf = w / total if total > 0 else 0.0
        is_unknown = brand == "Unknown" or model == "Unknown"
        label = None if is_unknown else {"brand": brand, "model": model}
        stable = (
            len(self._buf) >= self._stable_min
            and smoothed_conf >= self._stable_conf
            and not is_unknown
        )
        return SmoothedLabel(label, smoothed_conf, stable, len(self._buf))
