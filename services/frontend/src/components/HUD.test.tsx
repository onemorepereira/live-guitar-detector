import { render } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { HUD, type VideoRect } from "./HUD";
import type { TrackDetection } from "../types/detection";

/**
 * jsdom does not implement <canvas>; `getContext` returns null by default.
 * We install a Proxy-based spy so tests can read every assignment to context
 * properties (`strokeStyle = ...`, `font = ...`, `globalAlpha = ...`) via the
 * setter trap. This is lighter than pulling in `vitest-canvas-mock`.
 */
type SpyCtx = {
  ctx: CanvasRenderingContext2D;
  calls: Array<{ prop: string; value: unknown }>;
};

const mockCtx = (): SpyCtx => {
  const calls: Array<{ prop: string; value: unknown }> = [];
  const target: Record<string, unknown> = {
    clearRect: vi.fn(),
    strokeRect: vi.fn(),
    fillRect: vi.fn(),
    fillText: vi.fn(),
    save: vi.fn(),
    restore: vi.fn(),
    measureText: vi.fn(() => ({ width: 100 })),
  };
  const proxy = new Proxy(target, {
    set(t, key, value) {
      t[String(key)] = value;
      calls.push({ prop: String(key), value });
      return true;
    },
    get(t, key) {
      return t[String(key)];
    },
  });
  return { ctx: proxy as unknown as CanvasRenderingContext2D, calls };
};

const rect: VideoRect = { videoW: 1920, videoH: 1080, elW: 960, elH: 540 };

let stubCtx: SpyCtx;
let originalRAF: typeof requestAnimationFrame;
let originalCAF: typeof cancelAnimationFrame;

beforeEach(() => {
  stubCtx = mockCtx();
  HTMLCanvasElement.prototype.getContext = vi.fn(
    () => stubCtx.ctx,
  ) as unknown as HTMLCanvasElement["getContext"];
  originalRAF = globalThis.requestAnimationFrame;
  originalCAF = globalThis.cancelAnimationFrame;
  // Synchronous RAF so the draw runs during render() and tests can assert
  // immediately. We return 0 and the cleanup function will cancel "the next
  // frame" — but since we never schedule one for real here, nothing else runs.
  let scheduled = false;
  globalThis.requestAnimationFrame = ((cb: FrameRequestCallback) => {
    if (scheduled) return 0;
    scheduled = true;
    cb(performance.now());
    scheduled = false;
    return 0 as unknown as number;
  }) as typeof requestAnimationFrame;
  globalThis.cancelAnimationFrame =
    vi.fn() as unknown as typeof cancelAnimationFrame;
});

afterEach(() => {
  globalThis.requestAnimationFrame = originalRAF;
  globalThis.cancelAnimationFrame = originalCAF;
  vi.restoreAllMocks();
});

function gibsonTrack(overrides: Partial<TrackDetection> = {}): TrackDetection {
  return {
    track_id: 1,
    bbox: [0.1, 0.1, 0.4, 0.4],
    detection_confidence: 0.9,
    label: { brand: "Gibson", model: "Les Paul", confidence: 0.85 },
    stable: true,
    age_frames: 30,
    ...overrides,
  };
}

describe("HUD", () => {
  // Colors are per-track-id (so multiple visible instruments get distinct
  // colors). track_id=1 → palette index 1 → "#43AA8B"; track_id=2 → "#F94144".
  it("uses the track-id palette color for a track", () => {
    render(<HUD tracks={[gibsonTrack()]} videoRect={rect} />);
    const strokeColors = stubCtx.calls
      .filter((c) => c.prop === "strokeStyle")
      .map((c) => c.value);
    expect(strokeColors).toContain("#43AA8B"); // palette[1]
  });

  it("distinct track_ids get distinct palette colors", () => {
    const t2 = gibsonTrack({ track_id: 2 });
    render(<HUD tracks={[gibsonTrack(), t2]} videoRect={rect} />);
    const strokeColors = stubCtx.calls
      .filter((c) => c.prop === "strokeStyle")
      .map((c) => c.value as string);
    expect(strokeColors).toContain("#43AA8B"); // palette[1]
    expect(strokeColors).toContain("#F94144"); // palette[2]
  });

  it("renders a stroke (color independent of stable/label) for an unstable track", () => {
    const t = gibsonTrack({ stable: false, label: null });
    render(<HUD tracks={[t]} videoRect={rect} />);
    const strokeColors = stubCtx.calls
      .filter((c) => c.prop === "strokeStyle")
      .map((c) => c.value as string);
    // Track-id palette is independent of label state — track_id=1 → palette[1].
    expect(strokeColors).toContain("#43AA8B");
  });

  it("renders 'Analyzing…' text for stable=false", () => {
    const t = gibsonTrack({ stable: false, label: null });
    const fillText = vi.fn();
    stubCtx.ctx.fillText = fillText as unknown as typeof stubCtx.ctx.fillText;
    render(<HUD tracks={[t]} videoRect={rect} />);
    const texts = fillText.mock.calls.map((c) => c[0]);
    expect(texts.some((s: string) => /Analyzing/.test(s))).toBe(true);
  });

  it("uses italic font for unstable label", () => {
    const t = gibsonTrack({ stable: false, label: null });
    render(<HUD tracks={[t]} videoRect={rect} />);
    const fonts = stubCtx.calls
      .filter((c) => c.prop === "font")
      .map((c) => c.value as string);
    expect(fonts.some((f) => f.includes("italic"))).toBe(true);
  });

  it("draws label below the box when bbox is near the top edge", () => {
    // bbox [0, 0, 0.1, 0.05] → in 960x540 with matching 16:9 aspect:
    // y1 = 0, y2 = 0.05 * 540 = 27. Top-edge heuristic triggers (y1 < 30),
    // so label baseline should be below y2.
    const t = gibsonTrack({ bbox: [0.0, 0.0, 0.1, 0.05] });
    const fillText = vi.fn();
    stubCtx.ctx.fillText = fillText as unknown as typeof stubCtx.ctx.fillText;
    render(<HUD tracks={[t]} videoRect={rect} />);
    const lastCall = fillText.mock.calls[fillText.mock.calls.length - 1];
    // fillText signature: (text, x, y). The y argument is index 2.
    expect(lastCall[2]).toBeGreaterThan(27);
  });

  it("ramps opacity from 0.65 → 1.0 once the track passes the min-draw age", () => {
    // Tracks with age_frames < MIN_DRAW_AGE_FRAMES (3) are intentionally
    // suppressed to mask ByteTrack ghosts. The opacity ramp itself still
    // starts at 0.3 for age=1; we just don't render until age=3.
    const ages = [3, 4, 5, 10];
    const alphas: number[] = [];
    for (const age of ages) {
      stubCtx = mockCtx();
      HTMLCanvasElement.prototype.getContext = vi.fn(
        () => stubCtx.ctx,
      ) as unknown as HTMLCanvasElement["getContext"];
      const { unmount } = render(
        <HUD tracks={[gibsonTrack({ age_frames: age })]} videoRect={rect} />,
      );
      const alphaCalls = stubCtx.calls
        .filter((c) => c.prop === "globalAlpha")
        .map((c) => c.value as number);
      // We set globalAlpha twice per track: first to the ramp value, then to
      // 1.0 for crisp text. The ramp value is the FIRST assignment.
      alphas.push(alphaCalls[0]);
      unmount();
    }
    expect(alphas[0]).toBeCloseTo(0.65, 2); // age=3 → 0.3 + 2*0.175
    expect(alphas[1]).toBeCloseTo(0.825, 2); // age=4 → 0.3 + 3*0.175
    expect(alphas[2]).toBeCloseTo(1.0, 2); // age=5 → 0.3 + 4*0.175 = 1.0
    expect(alphas[3]).toBeCloseTo(1.0, 2); // age=10 → clamped at 1.0
  });

  it("renders an empty canvas (no errors) when tracks is empty", () => {
    const { container } = render(<HUD tracks={[]} videoRect={rect} />);
    expect(container.querySelector("canvas")).toBeTruthy();
  });

  it("draws a thicker stroke and a shadow for the highlighted track", () => {
    render(
      <HUD tracks={[gibsonTrack()]} videoRect={rect} highlightedTrackId={1} />,
    );
    const lineWidths = stubCtx.calls
      .filter((c) => c.prop === "lineWidth")
      .map((c) => c.value as number);
    // The inner brand-color stroke should use the highlighted width (5)
    // rather than the default (3).
    expect(lineWidths).toContain(5);
    const shadowColors = stubCtx.calls
      .filter((c) => c.prop === "shadowColor")
      .map((c) => c.value as string);
    // Highlight shadow uses the same per-track palette color.
    expect(shadowColors).toContain("#43AA8B");
    const shadowBlurs = stubCtx.calls
      .filter((c) => c.prop === "shadowBlur")
      .map((c) => c.value as number);
    expect(shadowBlurs.some((b) => b > 0)).toBe(true);
  });
});

/**
 * Tests for the lerp + hold behaviour need to step the rAF loop manually
 * across multiple "frames" with controlled time, so they install their
 * own shim that captures the latest scheduled callback. `stepFrame(t)`
 * sets `performance.now()` to `t`, invokes the captured callback, and
 * leaves the next callback queued for the following step.
 */
describe("HUD smoothing + hold", () => {
  let lastCb: FrameRequestCallback | null = null;
  let fakeNow = 0;
  let savedRAF: typeof requestAnimationFrame;
  let savedCAF: typeof cancelAnimationFrame;
  let savedPerfNow: typeof performance.now;
  let smoothCtx: SpyCtx;

  beforeEach(() => {
    smoothCtx = mockCtx();
    HTMLCanvasElement.prototype.getContext = vi.fn(
      () => smoothCtx.ctx,
    ) as unknown as HTMLCanvasElement["getContext"];
    savedRAF = globalThis.requestAnimationFrame;
    savedCAF = globalThis.cancelAnimationFrame;
    savedPerfNow = performance.now.bind(performance);
    fakeNow = 0;
    lastCb = null;
    globalThis.requestAnimationFrame = ((cb: FrameRequestCallback) => {
      lastCb = cb;
      return 1 as unknown as number;
    }) as typeof requestAnimationFrame;
    globalThis.cancelAnimationFrame = ((_id: number) => {
      lastCb = null;
    }) as typeof cancelAnimationFrame;
    performance.now = (() => fakeNow) as typeof performance.now;
  });

  afterEach(() => {
    globalThis.requestAnimationFrame = savedRAF;
    globalThis.cancelAnimationFrame = savedCAF;
    performance.now = savedPerfNow;
    vi.restoreAllMocks();
  });

  function stepFrame(t: number): void {
    fakeNow = t;
    const cb = lastCb;
    lastCb = null;
    if (cb) cb(t);
  }

  it("holds the last bbox briefly after tracks become empty, then drops it", () => {
    const t0 = gibsonTrack({ bbox: [0.1, 0.1, 0.4, 0.4] });
    const { rerender } = render(<HUD tracks={[t0]} videoRect={rect} />);
    // First paint at t=0 — the box is drawn.
    stepFrame(0);
    const strokeRectsBefore = (
      smoothCtx.ctx.strokeRect as unknown as ReturnType<typeof vi.fn>
    ).mock.calls.length;
    expect(strokeRectsBefore).toBeGreaterThan(0);

    // Tracks vanish. Within the hold window the box must still be drawn.
    rerender(<HUD tracks={[]} videoRect={rect} />);
    stepFrame(1000); // 1 s after lastSeenAt → still inside HOLD_MS (1500).
    const strokeRectsHold = (
      smoothCtx.ctx.strokeRect as unknown as ReturnType<typeof vi.fn>
    ).mock.calls.length;
    expect(strokeRectsHold).toBeGreaterThan(strokeRectsBefore);

    // Past the hold window the box must be pruned.
    stepFrame(2000); // 2 s after lastSeenAt → outside HOLD_MS.
    const strokeRectsAfter = (
      smoothCtx.ctx.strokeRect as unknown as ReturnType<typeof vi.fn>
    ).mock.calls.length;
    // No new strokeRect calls between t=1000 and t=2000 — the box is gone.
    expect(strokeRectsAfter).toBe(strokeRectsHold);
  });

  it("lerps the bbox between two consecutive detection updates", () => {
    // Capture the (x, y, w, h) of the inner brand-color strokeRect — it is
    // the second strokeRect call per draw (after the black halo). We track
    // the LATEST call to inspect the smoothed bbox at any point.
    const strokeRect = smoothCtx.ctx.strokeRect as unknown as ReturnType<
      typeof vi.fn
    >;

    const start = gibsonTrack({ bbox: [0.1, 0.1, 0.2, 0.2] });
    const { rerender } = render(<HUD tracks={[start]} videoRect={rect} />);
    stepFrame(0);

    // The first paint should be exactly at the start position (snap on
    // first-seen). denormalizeBbox in a 1920x1080 → 960x540 letterbox-free
    // case scales by 0.5: x1 = 0.1*960 = 96, y1 = 0.1*540 = 54.
    const firstCall = strokeRect.mock.calls[strokeRect.mock.calls.length - 1];
    expect(firstCall[0]).toBeCloseTo(96, 1);
    expect(firstCall[1]).toBeCloseTo(54, 1);

    // New detection event: bbox jumps to [0.5, 0.5, 0.6, 0.6]. The target
    // is first EMA-blended toward the measurement (BBOX_EMA_ALPHA = 0.3):
    //   target_x_norm = 0.7 * 0.1 + 0.3 * 0.5 = 0.22  (in px: 0.22*960 = 211.2)
    // Then the rAF loop lerps the current toward that smoothed target
    // (LERP_RATE = 0.35):
    //   current_x_px = 96 + 0.35 * (211.2 - 96) ≈ 136.32
    const next = gibsonTrack({ bbox: [0.5, 0.5, 0.6, 0.6] });
    rerender(<HUD tracks={[next]} videoRect={rect} />);
    stepFrame(16); // one ~60 Hz frame later.

    const lerpedCall = strokeRect.mock.calls[strokeRect.mock.calls.length - 1];
    const lerpedX = lerpedCall[0] as number;
    const lerpedY = lerpedCall[1] as number;
    expect(lerpedX).toBeGreaterThan(96);
    expect(lerpedX).toBeLessThan(480);
    expect(lerpedY).toBeGreaterThan(54);
    expect(lerpedY).toBeLessThan(270);
    // Sanity: the smoothed step matches the EMA+LERP composition.
    const emaTargetPx = (0.7 * 0.1 + 0.3 * 0.5) * 960;
    expect(lerpedX).toBeCloseTo(96 + 0.35 * (emaTargetPx - 96), 1);
  });
});
