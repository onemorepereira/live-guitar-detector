"""Tests for the track lifecycle manager (DESIGN.md §5.4).

The :class:`TrackRegistry` answers two questions the pipeline needs:

1. "Should I pay the CLIP cost on this track this frame?"
2. "Which tracks have gone stale and can be forgotten?"
"""

from __future__ import annotations

import pytest
from app.tracks import TrackRegistry

# A bbox area fraction that is comfortably above the small-bbox skip
# threshold (DESIGN.md §5.4 sets the floor at 0.5% of frame area). Using
# 0.5 here means every test that isn't *about* the small-bbox skip can
# ignore it.
BIG_BBOX = 0.5


def test_observe_records_first_and_last_seen():
    """observe() sets first_seen on first call, updates last_seen on each call."""
    reg = TrackRegistry()
    reg.observe(track_id=1, frame_no=10)
    reg.observe(track_id=1, frame_no=15)
    # age = current_frame_no - first_seen_frame_no = 15 - 10 = 5
    assert reg.age(track_id=1, frame_no=15) == 5


def test_age_unknown_track_raises():
    """age() for a track that was never observed raises KeyError."""
    reg = TrackRegistry()
    with pytest.raises(KeyError):
        reg.age(track_id=99, frame_no=100)


def test_should_classify_warmup_first_five_frames():
    """First 5 frames (age 0..4) always classify, regardless of stable flag."""
    reg = TrackRegistry()
    first_seen = 100
    reg.observe(track_id=1, frame_no=first_seen)
    for age in range(5):
        frame_no = first_seen + age
        # Re-observe so last_seen is current — observe() is idempotent for
        # first_seen but does update last_seen.
        reg.observe(track_id=1, frame_no=frame_no)
        assert (
            reg.should_classify(
                track_id=1,
                frame_no=frame_no,
                stable=False,
                bbox_area_fraction=BIG_BBOX,
            )
            is True
        ), f"warm-up failed at age {age} (unstable)"
        assert (
            reg.should_classify(
                track_id=1,
                frame_no=frame_no,
                stable=True,
                bbox_area_fraction=BIG_BBOX,
            )
            is True
        ), f"warm-up failed at age {age} (stable)"


def test_should_classify_unstable_every_sixth_frame_after_warmup():
    """Unstable tracks classify at ages 5, 11, 17, 23 and skip ages 6..10."""
    reg = TrackRegistry()
    first_seen = 0
    reg.observe(track_id=1, frame_no=first_seen)

    expect_true = {5, 11, 17, 23}
    expect_false = {6, 7, 8, 9, 10}

    for age in expect_true | expect_false:
        frame_no = first_seen + age
        got = reg.should_classify(
            track_id=1,
            frame_no=frame_no,
            stable=False,
            bbox_area_fraction=BIG_BBOX,
        )
        want = age in expect_true
        assert got is want, f"unstable age {age}: want {want}, got {got}"


def test_should_classify_stable_every_thirtieth_frame_after_warmup():
    """Stable tracks classify at ages 5, 35, 65 and skip ages 6, 11, 17, 34."""
    reg = TrackRegistry()
    first_seen = 0
    reg.observe(track_id=1, frame_no=first_seen)

    expect_true = {5, 35, 65}
    expect_false = {6, 11, 17, 34}

    for age in expect_true | expect_false:
        frame_no = first_seen + age
        got = reg.should_classify(
            track_id=1,
            frame_no=frame_no,
            stable=True,
            bbox_area_fraction=BIG_BBOX,
        )
        want = age in expect_true
        assert got is want, f"stable age {age}: want {want}, got {got}"


def test_should_classify_skips_small_bbox():
    """Even on a warm-up frame, bbox below the area floor blocks classification."""
    reg = TrackRegistry()
    reg.observe(track_id=1, frame_no=0)
    # age = 0 (warm-up), but bbox is tiny → must skip
    assert (
        reg.should_classify(
            track_id=1,
            frame_no=0,
            stable=False,
            bbox_area_fraction=0.001,
        )
        is False
    )


def test_prune_removes_stale_tracks():
    """Tracks not seen for ≥90 frames are pruned; their IDs are returned."""
    reg = TrackRegistry()
    reg.observe(track_id=1, frame_no=0)
    reg.observe(track_id=2, frame_no=50)

    pruned = reg.prune(current_frame_no=95)

    assert pruned == [1]
    # track 2 is still alive — observing it again should not raise.
    reg.observe(track_id=2, frame_no=96)
    assert reg.age(track_id=2, frame_no=96) == 46  # first_seen was 50


def test_prune_returns_empty_when_no_stale_tracks():
    """prune() with all tracks fresh returns an empty list."""
    reg = TrackRegistry()
    reg.observe(track_id=1, frame_no=10)
    assert reg.prune(current_frame_no=10) == []


def test_should_classify_unknown_track_treats_as_new():
    """should_classify() for a never-observed track raises KeyError.

    The caller contract is: always observe() before should_classify().
    """
    reg = TrackRegistry()
    with pytest.raises(KeyError):
        reg.should_classify(
            track_id=42,
            frame_no=0,
            stable=False,
            bbox_area_fraction=BIG_BBOX,
        )
