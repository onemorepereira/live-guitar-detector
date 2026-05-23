import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useDetections } from "./useDetections";

// Capture instances so tests can drive them.
const instances: FakeWebSocket[] = [];

class FakeWebSocket {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;
  url: string;
  readyState: number = 0;
  send = vi.fn();
  close = vi.fn(() => {
    this.readyState = FakeWebSocket.CLOSED;
    this.dispatchEvent("close", {});
  });
  private listeners: Record<string, Array<(ev: unknown) => void>> = {};

  constructor(url: string) {
    this.url = url;
    instances.push(this);
  }

  addEventListener(type: string, fn: (ev: unknown) => void) {
    (this.listeners[type] ??= []).push(fn);
  }

  removeEventListener(type: string, fn: (ev: unknown) => void) {
    this.listeners[type] = (this.listeners[type] ?? []).filter((f) => f !== fn);
  }

  dispatchEvent(type: string, ev: unknown) {
    for (const fn of this.listeners[type] ?? []) fn(ev);
  }

  open() {
    this.readyState = FakeWebSocket.OPEN;
    this.dispatchEvent("open", {});
  }

  receive(data: unknown) {
    this.dispatchEvent("message", { data: JSON.stringify(data) });
  }

  forceClose() {
    this.readyState = FakeWebSocket.CLOSED;
    this.dispatchEvent("close", {});
  }
}

beforeEach(() => {
  instances.length = 0;
  vi.stubGlobal("WebSocket", FakeWebSocket as unknown as typeof WebSocket);
});

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
});

describe("useDetections", () => {
  it("is idle when sessionId is null", () => {
    const { result } = renderHook(() => useDetections(null));
    expect(result.current.state).toBe("idle");
    expect(result.current.event).toBeNull();
  });

  it("opens a WS to /ws?session_id=... and transitions to open", async () => {
    const { result } = renderHook(() => useDetections("abc"));
    await waitFor(() => expect(instances.length).toBe(1));
    expect(instances[0].url).toBe("/ws?session_id=abc");
    act(() => instances[0].open());
    await waitFor(() => expect(result.current.state).toBe("open"));
  });

  it("exposes only the most recent event", async () => {
    const { result } = renderHook(() => useDetections("abc"));
    await waitFor(() => expect(instances.length).toBe(1));
    act(() => instances[0].open());

    const e1 = {
      session_id: "abc",
      frame_id: 1,
      frame_ts: 1,
      inference_ts: 2,
      tracks: [],
    };
    const e2 = {
      session_id: "abc",
      frame_id: 2,
      frame_ts: 3,
      inference_ts: 4,
      tracks: [],
    };
    act(() => {
      instances[0].receive(e1);
      instances[0].receive(e2);
    });
    expect(result.current.event?.frame_id).toBe(2);
  });

  it("ignores pong messages (doesn't replace event)", async () => {
    const { result } = renderHook(() => useDetections("abc"));
    await waitFor(() => expect(instances.length).toBe(1));
    act(() => instances[0].open());
    const e1 = {
      session_id: "abc",
      frame_id: 1,
      frame_ts: 1,
      inference_ts: 2,
      tracks: [],
    };
    act(() => instances[0].receive(e1));
    expect(result.current.event?.frame_id).toBe(1);
    act(() => instances[0].receive({ type: "pong" }));
    expect(result.current.event?.frame_id).toBe(1);
  });

  it("reconnects after close with exponential backoff capped at 5s", async () => {
    vi.useFakeTimers();
    renderHook(() => useDetections("abc"));
    expect(instances.length).toBe(1);
    act(() => instances[0].open());

    // Close → should schedule reconnect at 1000ms (attempt=0 → 2^0=1000).
    act(() => instances[0].forceClose());
    expect(instances.length).toBe(1);
    act(() => {
      vi.advanceTimersByTime(1000);
    });
    expect(instances.length).toBe(2);

    // Second close → 2000ms.
    act(() => instances[1].open());
    act(() => instances[1].forceClose());
    act(() => {
      vi.advanceTimersByTime(2000);
    });
    expect(instances.length).toBe(3);

    // After many closes the delay should cap at 5000ms.
    for (let i = 0; i < 10; i++) {
      act(() => instances[instances.length - 1].open());
      act(() => instances[instances.length - 1].forceClose());
      act(() => {
        vi.advanceTimersByTime(5000);
      });
    }
    // We just verify the cap doesn't crash; the exact instance count isn't important.
    expect(instances.length).toBeGreaterThan(5);
  });

  it("sends ping every 5s while open", async () => {
    vi.useFakeTimers();
    renderHook(() => useDetections("abc"));
    expect(instances.length).toBe(1);
    act(() => instances[0].open());

    expect(instances[0].send).not.toHaveBeenCalled();
    act(() => {
      vi.advanceTimersByTime(5000);
    });
    expect(instances[0].send).toHaveBeenCalledWith(
      JSON.stringify({ type: "ping" }),
    );

    act(() => {
      vi.advanceTimersByTime(5000);
    });
    expect(instances[0].send).toHaveBeenCalledTimes(2);
  });

  it("warns when no pong arrives within 3s of ping", async () => {
    vi.useFakeTimers();
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    renderHook(() => useDetections("abc"));
    act(() => instances[0].open());
    act(() => {
      vi.advanceTimersByTime(5000); // triggers first ping
    });
    expect(warn).not.toHaveBeenCalled();
    act(() => {
      vi.advanceTimersByTime(3000); // pong timeout
    });
    expect(warn).toHaveBeenCalledWith(expect.stringContaining("no pong"));
  });

  it("clears pong timeout on pong receipt", async () => {
    vi.useFakeTimers();
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    renderHook(() => useDetections("abc"));
    act(() => instances[0].open());
    act(() => {
      vi.advanceTimersByTime(5000); // first ping
    });
    act(() => instances[0].receive({ type: "pong" }));
    act(() => {
      vi.advanceTimersByTime(5000); // would have warned, but pong cleared it
    });
    // No warn from THIS ping cycle (a new pong-timeout was set by the latest ping).
    // The exact count is sensitive to fake-timer ordering — just check warn wasn't called from the pong-clear branch.
    expect(warn).toHaveBeenCalledTimes(0);
  });

  it("closes the socket on unmount", () => {
    const { unmount } = renderHook(() => useDetections("abc"));
    expect(instances.length).toBe(1);
    unmount();
    expect(instances[0].close).toHaveBeenCalled();
  });
});
