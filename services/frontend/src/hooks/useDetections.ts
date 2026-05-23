import { useEffect, useState } from "react";

import type { DetectionEvent } from "../types/detection";

export type WSState = "idle" | "connecting" | "open" | "closed";

export interface UseDetectionsResult {
  /** Most recent detection event; null until the first arrival. */
  event: DetectionEvent | null;
  /** Current WebSocket state from the hook's perspective. */
  state: WSState;
  /** Number of reconnect attempts since the last successful open. 0 on first connect. */
  reconnectAttempt: number;
}

const PING_INTERVAL_MS = 5000;
const PONG_TIMEOUT_MS = 3000;
const BACKOFF_BASE_MS = 1000;
const BACKOFF_CAP_MS = 5000;

/**
 * Subscribes to the gateway's detection-event WebSocket for a session.
 *
 * Behavior:
 *  - When `sessionId` is null, the hook stays idle and opens no socket.
 *  - On mount / sessionId change: opens `/ws?session_id=...` (the browser
 *    fills in the protocol and host).
 *  - Only the most recent event is exposed — we never queue stale frames
 *    since the UI redraws at canvas frame rate.
 *  - On close (other than a caller-initiated unmount / sessionId change),
 *    reconnects with exponential backoff: 1s, 2s, 4s, then capped at 5s.
 *  - Every 5 seconds while open, sends `{"type":"ping"}`. If no pong
 *    arrives within 3 seconds of the latest ping, logs a console.warn.
 *  - Inbound `{"type":"pong"}` messages clear the pong timeout and do
 *    not replace the exposed event.
 */
export function useDetections(sessionId: string | null): UseDetectionsResult {
  const [event, setEvent] = useState<DetectionEvent | null>(null);
  const [state, setState] = useState<WSState>("idle");
  const [reconnectAttempt, setReconnectAttempt] = useState(0);

  useEffect(() => {
    if (!sessionId) {
      setState("idle");
      return;
    }

    let cancelled = false;
    let socket: WebSocket | null = null;
    let pingInterval: ReturnType<typeof setInterval> | null = null;
    let pongTimeout: ReturnType<typeof setTimeout> | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    let attempt = 0;

    const clearTimers = () => {
      if (pingInterval) clearInterval(pingInterval);
      if (pongTimeout) clearTimeout(pongTimeout);
      if (reconnectTimer) clearTimeout(reconnectTimer);
      pingInterval = null;
      pongTimeout = null;
      reconnectTimer = null;
    };

    const sendPing = () => {
      if (!socket || socket.readyState !== WebSocket.OPEN) return;
      socket.send(JSON.stringify({ type: "ping" }));
      if (pongTimeout) clearTimeout(pongTimeout);
      pongTimeout = setTimeout(() => {
        console.warn("ws: no pong within 3s of ping");
      }, PONG_TIMEOUT_MS);
    };

    const connect = () => {
      if (cancelled) return;
      setState("connecting");
      socket = new WebSocket(`/ws?session_id=${encodeURIComponent(sessionId)}`);

      socket.addEventListener("open", () => {
        if (cancelled) return;
        setState("open");
        attempt = 0;
        setReconnectAttempt(0);
        pingInterval = setInterval(sendPing, PING_INTERVAL_MS);
      });

      socket.addEventListener("message", (ev) => {
        if (cancelled) return;
        const data = (ev as MessageEvent).data;
        const raw = typeof data === "string" ? data : "";
        try {
          const parsed = JSON.parse(raw) as Partial<DetectionEvent> & {
            type?: string;
          };
          if (parsed.type === "pong") {
            if (pongTimeout) {
              clearTimeout(pongTimeout);
              pongTimeout = null;
            }
            return;
          }
          // Treat anything else as a DetectionEvent.
          setEvent(parsed as DetectionEvent);
        } catch {
          // Drop malformed messages silently — server-side bug, not actionable on client.
        }
      });

      socket.addEventListener("close", () => {
        if (cancelled) return;
        setState("closed");
        if (pingInterval) clearInterval(pingInterval);
        if (pongTimeout) clearTimeout(pongTimeout);
        pingInterval = null;
        pongTimeout = null;
        // Reconnect with exponential backoff.
        const delay = Math.min(BACKOFF_BASE_MS * 2 ** attempt, BACKOFF_CAP_MS);
        attempt += 1;
        setReconnectAttempt(attempt);
        reconnectTimer = setTimeout(() => {
          if (!cancelled) connect();
        }, delay);
      });

      socket.addEventListener("error", () => {
        // We rely on the 'close' that follows to drive reconnect.
      });
    };

    connect();

    return () => {
      cancelled = true;
      clearTimers();
      if (socket) {
        try {
          socket.close();
        } catch {
          // ignore
        }
      }
    };
  }, [sessionId]);

  return { event, state, reconnectAttempt };
}
