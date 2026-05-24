import { useEffect, useRef } from "react";

import { denormalizeBbox } from "../lib/bbox";
import type { TrackDetection } from "../types/detection";

/**
 * Canvas overlay drawing per-track bounding boxes and labels on top of the
 * `<video>` element. See DESIGN.md §5.6.
 *
 * The component owns its own `requestAnimationFrame` loop so the canvas
 * redraws at display refresh rate; React only re-renders when `tracks` or
 * `videoRect` change (which triggers the effect to restart the loop with
 * fresh closures).
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
 * Brand → stroke color. Values from DESIGN.md §5.6.
 * `Unknown` covers both the literal "Unknown" brand and any null-label track.
 */
const BRAND_COLOR: Record<string, string> = {
  Gibson: "#C8A45C",
  Fender: "#F5F5F5",
  Unknown: "#888888",
};

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

export function HUD({
  tracks,
  videoRect,
  highlightedTrackId = null,
}: HUDProps): JSX.Element {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let rafId = 0;
    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      for (const t of tracks) {
        drawTrack(ctx, t, videoRect, t.track_id === highlightedTrackId);
      }
      rafId = requestAnimationFrame(draw);
    };
    rafId = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(rafId);
    };
  }, [tracks, videoRect, highlightedTrackId]);

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
  rect: VideoRect,
  highlighted: boolean,
): void {
  const [x1, y1, x2, y2] = denormalizeBbox(t.bbox, rect);
  const brand = t.label?.brand ?? "Unknown";
  const color = BRAND_COLOR[brand] ?? BRAND_COLOR.Unknown;
  const strokeWidth = highlighted ? HIGHLIGHT_STROKE_WIDTH : STROKE_WIDTH;

  // Opacity ramp 0.3 → 1.0 over ages 1..5; clamp at 1.0.
  const alpha = Math.min(
    1.0,
    ALPHA_INITIAL + Math.max(0, t.age_frames - 1) * ALPHA_STEP,
  );

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
