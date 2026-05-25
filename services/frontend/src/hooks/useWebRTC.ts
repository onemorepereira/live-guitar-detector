import { useEffect, useState } from "react";

import { fetchConfig } from "../api/config";
import { postOffer } from "../api/webrtc";

export type WebRTCState =
  | "idle" // no stream yet
  | "connecting" // peer construction + offer in flight
  | "connected" // ICE connected
  | "failed" // error path
  | "closed"; // peer closed

export interface UseWebRTCResult {
  state: WebRTCState;
  error: string | null;
}

/**
 * Negotiates a WebRTC connection from the browser to the gateway.
 *
 * Lifecycle:
 *  - When `stream` and `sessionId` are both provided, constructs a fresh
 *    RTCPeerConnection, adds the stream's tracks, creates+posts an SDP
 *    offer, and sets the answer as the remote description.
 *  - When either prop changes (or the component unmounts), the previous
 *    peer is closed.
 *  - State transitions reflect both the local negotiation flow and the
 *    underlying ICE connection state.
 */
export function useWebRTC(
  stream: MediaStream | null,
  sessionId: string | null,
): UseWebRTCResult {
  const [state, setState] = useState<WebRTCState>("idle");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!stream || !sessionId) {
      setState("idle");
      setError(null);
      return;
    }

    let cancelled = false;
    let peer: RTCPeerConnection | null = null;

    const onIce = () => {
      if (!peer || cancelled) return;
      const ice = peer.iceConnectionState;
      if (ice === "connected" || ice === "completed") {
        setState("connected");
      } else if (ice === "failed" || ice === "disconnected") {
        setState("failed");
        setError(`ICE state ${ice}`);
      } else if (ice === "closed") {
        setState("closed");
      }
    };

    setState("connecting");
    setError(null);
    void (async () => {
      try {
        const cfg = await fetchConfig();
        if (cancelled) return;
        peer = new RTCPeerConnection({ iceServers: cfg.iceServers });
        peer.addEventListener("iceconnectionstatechange", onIce);
        for (const track of stream.getTracks()) {
          peer.addTrack(track, stream);
        }
        const offer = await peer.createOffer();
        await peer.setLocalDescription(offer);
        if (cancelled) return;
        const answer = await postOffer({
          session_id: sessionId,
          sdp: peer.localDescription?.sdp ?? "",
          type: "offer",
        });
        if (cancelled) return;
        await peer.setRemoteDescription({
          type: answer.type,
          sdp: answer.sdp,
        });
      } catch (e) {
        if (cancelled) return;
        setState("failed");
        setError(e instanceof Error ? e.message : String(e));
      }
    })();

    return () => {
      cancelled = true;
      if (peer) {
        peer.removeEventListener("iceconnectionstatechange", onIce);
        try {
          peer.close();
        } catch {
          // ignore
        }
      }
    };
  }, [stream, sessionId]);

  return { state, error };
}
