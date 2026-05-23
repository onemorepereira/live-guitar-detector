import { afterEach, describe, expect, it, vi } from "vitest";

import { createSession, deleteSession } from "./session";

afterEach(() => {
  vi.restoreAllMocks();
});

describe("createSession", () => {
  it("posts the session id and resolves on 200", async () => {
    const fetchMock = vi.fn(
      async () => new Response(JSON.stringify({ ok: true }), { status: 200 }),
    );
    vi.stubGlobal("fetch", fetchMock);
    await expect(createSession("abc")).resolves.toBeUndefined();
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/session",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ session_id: "abc" }),
      }),
    );
  });

  it("throws SessionError with status on 409", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response("dup", { status: 409 })),
    );
    await expect(createSession("abc")).rejects.toMatchObject({
      name: "SessionError",
      status: 409,
    });
  });
});

describe("deleteSession", () => {
  it("DELETEs the session id and tolerates 204", async () => {
    const fetchMock = vi.fn(async () => new Response(null, { status: 204 }));
    vi.stubGlobal("fetch", fetchMock);
    await expect(deleteSession("abc")).resolves.toBeUndefined();
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/session/abc",
      expect.objectContaining({ method: "DELETE" }),
    );
  });

  it("tolerates 404 (idempotent delete)", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response("missing", { status: 404 })),
    );
    await expect(deleteSession("abc")).resolves.toBeUndefined();
  });

  it("throws SessionError on other non-2xx", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response("boom", { status: 500 })),
    );
    await expect(deleteSession("abc")).rejects.toMatchObject({
      name: "SessionError",
      status: 500,
    });
  });
});
