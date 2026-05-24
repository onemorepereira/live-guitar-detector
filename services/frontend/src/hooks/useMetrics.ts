import { useEffect, useRef, useState } from "react";

import type { DetectionEvent } from "../types/detection";

export interface MetricsResult {
  /** Average frames per second of the render loop (sliding window). */
  videoFps: number;
  /** Average detection events per second from the WebSocket (sliding window). */
  detectionFps: number;
  /** ms elapsed since the gateway stamped the most recent event. */
  lastFrameAgeMs: number | null;
}

const WINDOW_MS = 2000;
const REFRESH_MS = 250; // panel update cadence

/**
 * Derives debug-panel metrics from a stream of detection events plus a
 * requestAnimationFrame-driven render-loop sample.
 *
 * Both rates are computed over a sliding 2-second window and refreshed at
 * 4 Hz so the panel updates feel live without thrashing React.
 *
 * Note: "dropped frames" is intentionally not exposed here — the worker
 * counts unconfirmed_skips internally but does not yet publish that signal
 * over the WebSocket, so the client has nothing to display.
 */
export function useMetrics(event: DetectionEvent | null): MetricsResult {
  const [metrics, setMetrics] = useState<MetricsResult>({
    videoFps: 0,
    detectionFps: 0,
    lastFrameAgeMs: null,
  });

  // Sliding-window sample buffers (performance.now() timestamps).
  const videoFrames = useRef<number[]>([]);
  const detectionFrames = useRef<number[]>([]);
  const lastEventFrameTs = useRef<number | null>(null);

  // rAF loop — counts a sample each frame.
  useEffect(() => {
    let raf = 0;
    const tick = () => {
      const now = performance.now();
      const buf = videoFrames.current;
      buf.push(now);
      while (buf.length > 0 && now - buf[0] > WINDOW_MS) buf.shift();
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  // Detection event sample — push a timestamp each time `event` reference
  // changes (the hook receives the most-recent only).
  useEffect(() => {
    if (event === null) return;
    const now = performance.now();
    const buf = detectionFrames.current;
    buf.push(now);
    while (buf.length > 0 && now - buf[0] > WINDOW_MS) buf.shift();
    lastEventFrameTs.current = event.frame_ts;
  }, [event]);

  // Periodically derive averaged numbers + age into React state.
  useEffect(() => {
    const interval = setInterval(() => {
      const now = performance.now();
      const vBuf = videoFrames.current;
      const dBuf = detectionFrames.current;
      while (vBuf.length > 0 && now - vBuf[0] > WINDOW_MS) vBuf.shift();
      while (dBuf.length > 0 && now - dBuf[0] > WINDOW_MS) dBuf.shift();
      const ts = lastEventFrameTs.current;
      setMetrics({
        videoFps: vBuf.length * (1000 / WINDOW_MS),
        detectionFps: dBuf.length * (1000 / WINDOW_MS),
        lastFrameAgeMs: ts === null ? null : Date.now() - ts,
      });
    }, REFRESH_MS);
    return () => clearInterval(interval);
  }, []);

  return metrics;
}
