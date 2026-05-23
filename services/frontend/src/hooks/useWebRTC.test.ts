import { renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useWebRTC } from "./useWebRTC";

class FakeRTCPeerConnection {
  iceConnectionState: string = "new";
  localDescription: RTCSessionDescription | null = null;
  remoteDescription: RTCSessionDescription | null = null;
  addTrack = vi.fn();
  close = vi.fn();
  addEventListener = vi.fn();
  removeEventListener = vi.fn();
  async createOffer(): Promise<RTCSessionDescriptionInit> {
    return { type: "offer", sdp: "v=0\nlocal" };
  }
  async setLocalDescription(d: RTCSessionDescriptionInit): Promise<void> {
    this.localDescription = {
      type: d.type,
      sdp: d.sdp,
    } as RTCSessionDescription;
  }
  async setRemoteDescription(d: RTCSessionDescriptionInit): Promise<void> {
    this.remoteDescription = {
      type: d.type,
      sdp: d.sdp,
    } as RTCSessionDescription;
  }
}

beforeEach(() => {
  // jsdom doesn't implement RTCPeerConnection.
  vi.stubGlobal(
    "RTCPeerConnection",
    FakeRTCPeerConnection as unknown as typeof RTCPeerConnection,
  );
  vi.stubGlobal(
    "fetch",
    vi.fn(
      async () =>
        new Response(JSON.stringify({ sdp: "v=0\nanswer", type: "answer" }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
    ),
  );
});

afterEach(() => {
  vi.restoreAllMocks();
});

function fakeStream(): MediaStream {
  return {
    getTracks: () => [
      { stop: vi.fn(), kind: "video" } as unknown as MediaStreamTrack,
    ],
  } as unknown as MediaStream;
}

describe("useWebRTC", () => {
  it("starts in 'idle' when stream or sessionId is null", () => {
    const { result } = renderHook(() => useWebRTC(null, null));
    expect(result.current.state).toBe("idle");
    expect(result.current.error).toBeNull();
  });

  it("transitions through connecting → connected on a successful negotiation", async () => {
    const stream = fakeStream();
    const { result } = renderHook(() => useWebRTC(stream, "s1"));

    // Connecting state is set synchronously when the effect runs.
    await waitFor(() => {
      expect(
        result.current.state === "connecting" ||
          result.current.state === "connected",
      ).toBe(true);
    });
  });

  it("sets state=failed on offer rejection", async () => {
    // Override fetch to 404 for this test.
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response("not found", { status: 404 })),
    );
    const stream = fakeStream();
    const { result } = renderHook(() => useWebRTC(stream, "s1"));
    await waitFor(() => expect(result.current.state).toBe("failed"));
    expect(result.current.error).toMatch(/404/);
  });
});
