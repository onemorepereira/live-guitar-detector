/** HTTP client for the gateway's session endpoint. */

export class SessionError extends Error {
  constructor(
    message: string,
    public readonly status: number,
  ) {
    super(message);
    this.name = "SessionError";
  }
}

export async function createSession(
  sessionId: string,
  baseUrl = "",
): Promise<void> {
  const resp = await fetch(`${baseUrl}/api/session`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  });
  if (!resp.ok) {
    const detail = await resp.text();
    throw new SessionError(
      `create rejected (${resp.status}): ${detail}`,
      resp.status,
    );
  }
}

export async function deleteSession(
  sessionId: string,
  baseUrl = "",
): Promise<void> {
  const resp = await fetch(
    `${baseUrl}/api/session/${encodeURIComponent(sessionId)}`,
    {
      method: "DELETE",
    },
  );
  // 204 expected; 404 is also OK (idempotent delete).
  if (!resp.ok && resp.status !== 404) {
    const detail = await resp.text();
    throw new SessionError(
      `delete rejected (${resp.status}): ${detail}`,
      resp.status,
    );
  }
}
