"""Tests for the rolling-window weighted voting module (DESIGN.md §5.4)."""

from __future__ import annotations

import pytest
from app.voting import SmoothedLabel, TrackVote  # noqa: F401


def make_label(brand: str = "Gibson", model: str = "Les Paul", conf: float = 0.9) -> dict:
    return {"brand": brand, "model": model, "confidence": conf}


def test_empty_vote_returns_none_label():
    v = TrackVote(window=15, stable_min=8, stable_conf=0.55)
    out = v.current()
    assert out.label is None
    assert out.stable is False


def test_consistent_label_becomes_stable_after_min_samples():
    v = TrackVote(window=15, stable_min=8, stable_conf=0.55)
    for _ in range(8):
        v.update(make_label("Gibson", "Les Paul", 0.9))
    out = v.current()
    assert out.label == {"brand": "Gibson", "model": "Les Paul"}
    assert out.stable is True
    assert out.confidence > 0.55


def test_below_stable_min_is_not_stable():
    v = TrackVote(window=15, stable_min=8, stable_conf=0.55)
    for _ in range(7):
        v.update(make_label("Gibson", "Les Paul", 0.9))
    assert v.current().stable is False


def test_flapping_labels_not_stable():
    v = TrackVote(window=15, stable_min=8, stable_conf=0.55)
    for i in range(15):
        v.update(
            make_label("Gibson", "Les Paul", 0.6)
            if i % 2
            else make_label("Fender", "Stratocaster", 0.6)
        )
    assert v.current().stable is False


def test_unknown_winning_emits_null_label_unstable():
    v = TrackVote(window=15, stable_min=8, stable_conf=0.55)
    for _ in range(10):
        v.update(make_label("Unknown", "Unknown", 0.8))
    out = v.current()
    assert out.label is None
    assert out.stable is False


def test_window_eviction():
    v = TrackVote(window=5, stable_min=3, stable_conf=0.5)
    for _ in range(5):
        v.update(make_label("Gibson", "Les Paul", 0.9))
    for _ in range(5):
        v.update(make_label("Fender", "Stratocaster", 0.9))
    out = v.current()
    assert out.label == {"brand": "Fender", "model": "Stratocaster"}


def test_smoothed_confidence_is_ratio_of_winner_to_total():
    v = TrackVote(window=10, stable_min=3, stable_conf=0.5)
    v.update(make_label("Gibson", "Les Paul", 0.8))
    v.update(make_label("Gibson", "Les Paul", 0.8))
    v.update(make_label("Fender", "Stratocaster", 0.4))
    out = v.current()
    # winner weight 1.6 / total 2.0 = 0.8
    assert out.confidence == pytest.approx(0.8, abs=0.01)
