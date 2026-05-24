import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useMetrics } from "./useMetrics";
import type { DetectionEvent } from "../types/detection";

function makeEvent(frame_id: number, frame_ts: number): DetectionEvent {
  return {
    session_id: "s",
    frame_id,
    frame_ts,
    inference_ts: frame_ts + 20,
    tracks: [],
  };
}

beforeEach(() => {
  vi.useFakeTimers();
  // Stub rAF with a controllable queue so the hook's render-loop doesn't
  // run away under fake timers.
  let rafId = 0;
  const cbs = new Map<number, FrameRequestCallback>();
  vi.stubGlobal("requestAnimationFrame", (cb: FrameRequestCallback) => {
    rafId += 1;
    cbs.set(rafId, cb);
    return rafId;
  });
  vi.stubGlobal("cancelAnimationFrame", (id: number) => {
    cbs.delete(id);
  });
  // Helper to manually advance rAF (unused by the current assertions but
  // available if a test wants to drive videoFps directly).
  (globalThis as unknown as { __flushRaf: () => void }).__flushRaf = () => {
    const snapshot = [...cbs.entries()];
    cbs.clear();
    for (const [, cb] of snapshot) cb(performance.now());
  };
});

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
});

describe("useMetrics", () => {
  it("starts at zero and null age", () => {
    const { result } = renderHook(() => useMetrics(null));
    expect(result.current.videoFps).toBe(0);
    expect(result.current.detectionFps).toBe(0);
    expect(result.current.lastFrameAgeMs).toBeNull();
  });

  it("computes lastFrameAgeMs from the latest event's frame_ts", () => {
    const now = Date.now();
    const ev = makeEvent(1, now - 50);
    const { result, rerender } = renderHook(({ e }) => useMetrics(e), {
      initialProps: { e: ev as DetectionEvent | null },
    });
    rerender({ e: ev });
    act(() => {
      vi.advanceTimersByTime(300);
    });
    expect(result.current.lastFrameAgeMs).toBeGreaterThanOrEqual(50);
    expect(result.current.lastFrameAgeMs).toBeLessThanOrEqual(400);
  });

  it("counts detection events in the sliding window", () => {
    const { result, rerender } = renderHook(({ e }) => useMetrics(e), {
      initialProps: { e: null as DetectionEvent | null },
    });
    // Push 10 events spread over 1 second.
    for (let i = 0; i < 10; i++) {
      rerender({ e: makeEvent(i, Date.now()) });
      act(() => {
        vi.advanceTimersByTime(100);
      });
    }
    // After 1s of 10 events, the 2s window should report a positive rate
    // bounded well below the per-call rerender count.
    act(() => {
      vi.advanceTimersByTime(250); // wait for next derive tick
    });
    expect(result.current.detectionFps).toBeGreaterThan(0);
    expect(result.current.detectionFps).toBeLessThan(15);
  });
});
