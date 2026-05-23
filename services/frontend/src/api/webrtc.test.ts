import { afterEach, describe, expect, it, vi } from "vitest";

import { postOffer } from "./webrtc";

afterEach(() => {
  vi.restoreAllMocks();
});

describe("postOffer", () => {
  it("posts JSON to /api/webrtc/offer and returns the parsed answer", async () => {
    const fetchMock = vi.fn(
      async () =>
        new Response(JSON.stringify({ sdp: "v=0...answer", type: "answer" }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const result = await postOffer({
      session_id: "s",
      sdp: "v=0",
      type: "offer",
    });
    expect(result).toEqual({ sdp: "v=0...answer", type: "answer" });
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/webrtc/offer",
      expect.objectContaining({
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: "s", sdp: "v=0", type: "offer" }),
      }),
    );
  });

  it("throws OfferError with status on non-2xx", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response("session not found", { status: 404 })),
    );
    await expect(
      postOffer({ session_id: "s", sdp: "v=0", type: "offer" }),
    ).rejects.toMatchObject({
      name: "OfferError",
      status: 404,
    });
  });
});
