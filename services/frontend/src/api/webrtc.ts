/** HTTP client for the gateway's WebRTC signalling endpoint. */

export interface OfferRequest {
  session_id: string;
  sdp: string;
  type: "offer";
}

export interface AnswerResponse {
  sdp: string;
  type: "answer";
}

export class OfferError extends Error {
  constructor(
    message: string,
    public readonly status: number,
  ) {
    super(message);
    this.name = "OfferError";
  }
}

export async function postOffer(
  req: OfferRequest,
  baseUrl = "",
): Promise<AnswerResponse> {
  const resp = await fetch(`${baseUrl}/api/webrtc/offer`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!resp.ok) {
    const detail = await resp.text();
    throw new OfferError(
      `offer rejected (${resp.status}): ${detail}`,
      resp.status,
    );
  }
  return (await resp.json()) as AnswerResponse;
}
