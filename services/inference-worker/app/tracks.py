"""Per-track lifecycle bookkeeping and classify scheduling (DESIGN.md §5.4).

This module owns two responsibilities for each tracked detection:

1. **When to classify** — running CLIP on every frame for every track is
   wasteful. The policy from DESIGN.md §5.4 is:

   - Always classify on the first :data:`WARMUP_FRAMES` of a new track
     (so the rolling vote has data immediately).
   - After the warm-up, classify every :data:`UNSTABLE_CLASSIFY_INTERVAL`
     frames while the track is unstable, and every
     :data:`STABLE_CLASSIFY_INTERVAL` frames once it has stabilized
     (drift check only).
   - Skip entirely when the bbox covers less than
     :data:`MIN_BBOX_AREA_FRACTION` of the frame — too small for CLIP to
     produce a useful signal.

2. **When to forget a track** — once a track hasn't been seen for
   :data:`PRUNE_AFTER_FRAMES` frames, its bookkeeping is dropped via
   :meth:`TrackRegistry.prune`.

The module is intentionally state-only (plain ``dict`` of dataclasses) and
stdlib-only; no I/O, no logging, no config. The caller (the inference
worker main loop) is responsible for feeding it observations and acting on
its decisions.
"""

from __future__ import annotations

from dataclasses import dataclass

# DESIGN.md §5.4 — classify scheduling constants.
WARMUP_FRAMES = 5  # always classify for the first N frames of a new track
UNSTABLE_CLASSIFY_INTERVAL = 6  # post-warmup cadence while vote is unstable
STABLE_CLASSIFY_INTERVAL = 30  # post-warmup cadence once vote is stable (drift check)
PRUNE_AFTER_FRAMES = 90  # forget tracks unseen for >= this many frames
MIN_BBOX_AREA_FRACTION = 0.005  # skip CLIP if bbox covers < 0.5% of frame area


@dataclass
class _TrackState:
    first_seen: int
    last_seen: int


class TrackRegistry:
    """Per-track first-/last-seen bookkeeping plus classify scheduling.

    Usage contract: call :meth:`observe` for every track sighting on every
    frame *before* calling :meth:`should_classify` or :meth:`age` for that
    track. Both raise :class:`KeyError` for unknown track IDs — the caller
    is expected to have observed first.
    """

    def __init__(self) -> None:
        self._tracks: dict[int, _TrackState] = {}

    def observe(self, track_id: int, frame_no: int) -> None:
        """Record a sighting of ``track_id`` at ``frame_no``.

        On the first observation, ``first_seen`` and ``last_seen`` are both
        set to ``frame_no``. On subsequent observations only ``last_seen``
        is updated; ``first_seen`` is preserved.
        """
        state = self._tracks.get(track_id)
        if state is None:
            self._tracks[track_id] = _TrackState(first_seen=frame_no, last_seen=frame_no)
        else:
            state.last_seen = frame_no

    def age(self, track_id: int, frame_no: int) -> int:
        """Frames elapsed since the track was first observed.

        Raises :class:`KeyError` if ``track_id`` has never been observed.
        """
        return frame_no - self._tracks[track_id].first_seen

    def should_classify(
        self,
        track_id: int,
        frame_no: int,
        *,
        stable: bool,
        bbox_area_fraction: float,
    ) -> bool:
        """Return True if the caller should run the classifier on this track.

        Policy (DESIGN.md §5.4):

        - Warm-up: ``age < WARMUP_FRAMES`` → always True (subject to area).
        - After warm-up:
          - stable=True  → every :data:`STABLE_CLASSIFY_INTERVAL` frames.
          - stable=False → every :data:`UNSTABLE_CLASSIFY_INTERVAL` frames.
        - If ``bbox_area_fraction < MIN_BBOX_AREA_FRACTION`` the result is
          always False, even during warm-up.

        ``bbox_area_fraction`` is the bbox area divided by the frame area
        (a value in ``[0, 1]``); the caller is responsible for the
        normalization so this module stays pure.

        Raises :class:`KeyError` if ``track_id`` has never been observed.
        """
        a = self.age(track_id, frame_no)  # raises KeyError if unknown
        if bbox_area_fraction < MIN_BBOX_AREA_FRACTION:
            return False
        if a < WARMUP_FRAMES:
            return True
        interval = STABLE_CLASSIFY_INTERVAL if stable else UNSTABLE_CLASSIFY_INTERVAL
        return (a - WARMUP_FRAMES) % interval == 0

    def prune(self, current_frame_no: int) -> list[int]:
        """Drop tracks unseen for ``>= PRUNE_AFTER_FRAMES`` frames.

        Returns the list of pruned track IDs (order is insertion order of
        the underlying dict, which Python guarantees since 3.7).
        """
        stale = [
            tid
            for tid, st in self._tracks.items()
            if current_frame_no - st.last_seen >= PRUNE_AFTER_FRAMES
        ]
        for tid in stale:
            del self._tracks[tid]
        return stale
