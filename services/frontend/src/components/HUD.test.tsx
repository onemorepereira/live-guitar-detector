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
  it("uses Gibson gold for a stable Gibson Les Paul track", () => {
    render(<HUD tracks={[gibsonTrack()]} videoRect={rect} />);
    const strokeColors = stubCtx.calls
      .filter((c) => c.prop === "strokeStyle")
      .map((c) => c.value);
    expect(strokeColors).toContain("#C8A45C");
  });

  it("uses Fender white for a stable Fender track", () => {
    const t = gibsonTrack({
      label: { brand: "Fender", model: "Stratocaster", confidence: 0.9 },
    });
    render(<HUD tracks={[t]} videoRect={rect} />);
    const strokeColors = stubCtx.calls
      .filter((c) => c.prop === "strokeStyle")
      .map((c) => c.value);
    expect(strokeColors).toContain("#F5F5F5");
  });

  it("uses gray for an unstable / null-label track", () => {
    const t = gibsonTrack({ stable: false, label: null });
    render(<HUD tracks={[t]} videoRect={rect} />);
    const strokeColors = stubCtx.calls
      .filter((c) => c.prop === "strokeStyle")
      .map((c) => c.value);
    expect(strokeColors).toContain("#888888");
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

  it("ramps opacity from 0.3 → 1.0 across age_frames 1..5", () => {
    const ages = [1, 3, 5, 10];
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
    expect(alphas[0]).toBeCloseTo(0.3, 2); // age=1 → 0.3
    expect(alphas[1]).toBeCloseTo(0.65, 2); // age=3 → 0.3 + 2*0.175
    expect(alphas[2]).toBeCloseTo(1.0, 2); // age=5 → 0.3 + 4*0.175 = 1.0
    expect(alphas[3]).toBeCloseTo(1.0, 2); // age=10 → clamped at 1.0
  });

  it("renders an empty canvas (no errors) when tracks is empty", () => {
    const { container } = render(<HUD tracks={[]} videoRect={rect} />);
    expect(container.querySelector("canvas")).toBeTruthy();
  });
});
