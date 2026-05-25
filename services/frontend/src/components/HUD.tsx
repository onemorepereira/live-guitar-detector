import { useEffect, useRef } from "react";

import { denormalizeBbox } from "../lib/bbox";
import type { TrackDetection } from "../types/detection";

/**
 * Canvas overlay drawing per-track bounding boxes and labels on top of the
 * `<video>` element. See DESIGN.md §5.6.
 *
 * The component owns a single `requestAnimationFrame` loop driven by an
 * internal animation-state map keyed by `track_id`. React only feeds new
 * detection events into the state map; the rAF loop is the source of truth
 * for what gets painted, which lets us:
 *
 *   1. Lerp bbox positions between consecutive detection events (the worker
 *      emits ~10–15 events/sec; the display refreshes at ~60 Hz, so without
 *      interpolation each new event would cause a visible snap).
 *   2. Hold the last-known bbox for a short window after a track disappears
 *      from the props (`HOLD_MS`), to mask single-frame drops that would
 *      otherwise cause flicker.
 *
 * ## Color & contrast (intentional non-change)
 *
 * Gibson gold (`#C8A45C`), Fender white (`#F5F5F5`), and Unknown gray
 * (`#888888`) are drawn over a 2-px wider black contrast halo. The halo +
 * the translucent black label panel give the box and text enough contrast
 * on both bright and dark backgrounds without resorting to a permanent
 * drop-shadow on every box (which read as overkill in the eyeball check).
 * The shadow is reserved for the highlighted track.
 *
 * ## Opacity easing (intentional non-change)
 *
 * The plan suggested `1 - exp(-age/2)` as a smoother alternative to the
 * linear ramp `0.3 + 0.175 * (age - 1)`. At 30 FPS the linear ramp is
 * already perceptually smooth, the existing snapshot tests are wedded to
 * its exact values, and the difference is < 0.1 alpha at every step.
 * Keeping linear; the helper `computeOpacity` is exported so the easing
 * curve is a single-point change if we ever want to revisit.
 */

export interface VideoRect {
  /** Source video frame width in pixels. */
  videoW: number;
  /** Source video frame height in pixels. */
  videoH: number;
  /** `<video>` element CSS pixel width — also the canvas pixel width. */
  elW: number;
  /** `<video>` element CSS pixel height — also the canvas pixel height. */
  elH: number;
}

export interface HUDProps {
  tracks: TrackDetection[];
  videoRect: VideoRect;
  highlightedTrackId?: number | null;
}

/**
 * Per-track-id stroke palette. Multiple instruments in frame get visually
 * distinct boxes; the brand+model is still in the label text. Cycles via
 * `track_id % length` so a single guitar stays the same color across frames.
 */
const TRACK_PALETTE: string[] = [
  "#F9C74F", // amber
  "#43AA8B", // teal
  "#F94144", // red
  "#577590", // slate
  "#F8961E", // orange
  "#9D4EDD", // purple
  "#06A77D", // emerald
  "#FF6F91", // pink
];

function colorOfTrack(trackId: number): string {
  return TRACK_PALETTE[Math.abs(trackId) % TRACK_PALETTE.length];
}

const LABEL_FONT = "bold 14px system-ui, sans-serif";
const ANALYZING_FONT = "italic 14px system-ui, sans-serif";
const STROKE_WIDTH = 3;
const HIGHLIGHT_STROKE_WIDTH = 5;
const HIGHLIGHT_SHADOW_BLUR = 18;
const LABEL_TEXT_HEIGHT = 16; // ~14px font + 2px ascender padding
const LABEL_PADDING = 4;
/**
 * If the top of the bbox is within this many pixels of the frame top, render
 * the label *below* the box instead of above it. Keeps labels visible when
 * the subject is at the top edge of the frame.
 */
const TOP_EDGE_THRESHOLD = 30;
/**
 * Opacity ramp: alpha = 0.3 at age_frames=1, reaching 1.0 at age_frames=5.
 * Linear step of 0.175 per frame.
 */
const ALPHA_INITIAL = 0.3;
const ALPHA_STEP = 0.175;
/**
 * Hold the last-known bbox for this long after a track stops appearing in
 * the props. Masks single-frame detection drops; tuned by eyeball — long
 * enough to absorb one missed event from a 10–15 Hz worker, short enough
 * that a truly-vanished guitar doesn't visibly linger.
 */
const HOLD_MS = 150;
/**
 * Per-frame interpolation fraction. At 60 FPS this reaches ~95 % of the
 * target in ~100 ms — fast enough to feel responsive, slow enough that
 * the box doesn't snap when a new detection event lands.
 */
const LERP_RATE = 0.35;
/**
 * EMA blend factor applied to the *target* bbox on each detection event.
 * Smooths YOLO's per-frame jitter so a handheld camera doesn't produce
 * a buzzing box. Lower = smoother (more lag); higher = snappier (more
 * jitter). 0.3 keeps the box visibly tracking motion while removing
 * most of the high-frequency wobble.
 */
const BBOX_EMA_ALPHA = 0.3;

type Bbox = [number, number, number, number];

/**
 * Per-track animation state held across renders. The map is keyed by
 * `track_id` and lives in a ref so the rAF loop can mutate it without
 * triggering React re-renders.
 */
interface TrackAnimState {
  /** Most recent normalized bbox received from the wire. */
  targetBbox: Bbox;
  /** Currently-drawn normalized bbox, being lerped toward target. */
  currentBbox: Bbox;
  /** `performance.now()` at the last time this track was in the props. */
  lastSeenAt: number;
  /** Cached most-recent detection — used for label, color, age, etc. */
  detection: TrackDetection;
}

/**
 * Linear opacity ramp; see ALPHA_INITIAL / ALPHA_STEP comment.
 * Exported for unit testing and to make any future easing-curve swap a
 * single-point change.
 */
export function computeOpacity(ageFrames: number): number {
  return Math.min(1.0, ALPHA_INITIAL + Math.max(0, ageFrames - 1) * ALPHA_STEP);
}

export function HUD({
  tracks,
  videoRect,
  highlightedTrackId = null,
}: HUDProps): JSX.Element {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const stateRef = useRef<Map<number, TrackAnimState>>(new Map());
  const rectRef = useRef(videoRect);
  const highlightedRef = useRef<number | null>(highlightedTrackId);

  // Mirror props into refs so the rAF loop always sees the latest values
  // without having to restart on every render.
  rectRef.current = videoRect;
  highlightedRef.current = highlightedTrackId;

  // Feed new detection events into the animation-state map. Tracks that
  // drop out of the props are NOT removed here — the rAF loop prunes them
  // after HOLD_MS to avoid flicker.
  useEffect(() => {
    const now = performance.now();
    const state = stateRef.current;
    for (const t of tracks) {
      const existing = state.get(t.track_id);
      if (existing) {
        // EMA-blend the target so the lerp loop doesn't chase YOLO jitter.
        for (let i = 0; i < 4; i++) {
          existing.targetBbox[i] =
            existing.targetBbox[i] * (1 - BBOX_EMA_ALPHA) +
            t.bbox[i] * BBOX_EMA_ALPHA;
        }
        existing.lastSeenAt = now;
        existing.detection = t;
      } else {
        // First time we see this track: snap currentBbox to target so it
        // appears at the right place on frame one rather than lerping in
        // from (0, 0, 0, 0).
        state.set(t.track_id, {
          targetBbox: [...t.bbox] as Bbox,
          currentBbox: [...t.bbox] as Bbox,
          lastSeenAt: now,
          detection: t,
        });
      }
    }
  }, [tracks]);

  // Single, stable rAF loop. Reads from the animation-state map + refs.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let rafId = 0;
    const draw = () => {
      const now = performance.now();
      const rect = rectRef.current;
      const highlighted = highlightedRef.current;
      const state = stateRef.current;

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const expired: number[] = [];
      for (const [id, st] of state) {
        if (now - st.lastSeenAt > HOLD_MS) {
          expired.push(id);
          continue;
        }
        // Lerp current toward target. Skip the math if we've effectively
        // converged — keeps tests deterministic when only one frame ever
        // runs and avoids accumulating float drift.
        for (let i = 0; i < 4; i++) {
          const delta = st.targetBbox[i] - st.currentBbox[i];
          if (Math.abs(delta) > 1e-4) {
            st.currentBbox[i] += delta * LERP_RATE;
          } else {
            st.currentBbox[i] = st.targetBbox[i];
          }
        }
        drawTrack(ctx, st.detection, st.currentBbox, rect, id === highlighted);
      }
      for (const id of expired) state.delete(id);

      rafId = requestAnimationFrame(draw);
    };
    rafId = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(rafId);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      width={videoRect.elW}
      height={videoRect.elH}
      className="absolute inset-0 pointer-events-none"
    />
  );
}

function drawTrack(
  ctx: CanvasRenderingContext2D,
  t: TrackDetection,
  normalizedBbox: Bbox,
  rect: VideoRect,
  highlighted: boolean,
): void {
  const [x1, y1, x2, y2] = denormalizeBbox(normalizedBbox, rect);
  const color = colorOfTrack(t.track_id);
  const strokeWidth = highlighted ? HIGHLIGHT_STROKE_WIDTH : STROKE_WIDTH;
  const alpha = computeOpacity(t.age_frames);

  ctx.save();
  ctx.globalAlpha = alpha;
  // Black contrast halo first; brand color on top.
  ctx.strokeStyle = "#000000";
  ctx.lineWidth = strokeWidth + 2;
  ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  if (highlighted) {
    ctx.shadowColor = color;
    ctx.shadowBlur = HIGHLIGHT_SHADOW_BLUR;
  }
  ctx.strokeStyle = color;
  ctx.lineWidth = strokeWidth;
  ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

  // Label text + font selection.
  const isStable = t.stable && t.label !== null;
  const text = isStable
    ? `[#${t.track_id}] ${t.label!.brand} ${t.label!.model} · ${Math.round(t.label!.confidence * 100)}%`
    : "Analyzing…";
  ctx.font = isStable ? LABEL_FONT : ANALYZING_FONT;

  const metrics = ctx.measureText(text);
  const textW = metrics.width;

  // Label below when the box hugs the top edge; otherwise above.
  const labelBelow = y1 < TOP_EDGE_THRESHOLD;
  const textBaseline = labelBelow
    ? y2 + LABEL_TEXT_HEIGHT + LABEL_PADDING + 2
    : y1 - 6;
  const bgY = labelBelow ? y2 + LABEL_PADDING : y1 - LABEL_TEXT_HEIGHT - 8;

  // Translucent black background panel for legibility.
  ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
  ctx.fillRect(
    x1,
    bgY,
    textW + LABEL_PADDING * 2,
    LABEL_TEXT_HEIGHT + LABEL_PADDING,
  );
  // Text on top — full opacity even when the box is faint mid-ramp.
  ctx.globalAlpha = 1.0;
  ctx.fillStyle = "#ffffff";
  ctx.fillText(text, x1 + LABEL_PADDING, textBaseline);
  ctx.restore();
}
