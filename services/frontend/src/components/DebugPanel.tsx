import { useEffect, useState } from "react";

import type { WSState } from "../hooks/useDetections";

export interface DebugPanelProps {
  wsState: WSState;
  webrtcState: string;
  detectionFps: number;
  videoFps: number;
  lastFrameAgeMs: number | null;
}

/**
 * Floating diagnostics overlay. Only rendered when ?debug=1 is in the URL.
 * Strictly informational — never mutates state.
 */
export function DebugPanel(props: DebugPanelProps): JSX.Element | null {
  const [enabled, setEnabled] = useState(false);
  useEffect(() => {
    setEnabled(new URLSearchParams(window.location.search).has("debug"));
  }, []);
  if (!enabled) return null;

  return (
    <aside className="fixed bottom-4 right-4 z-50 rounded bg-zinc-950/90 border border-zinc-700 text-xs text-zinc-100 px-3 py-2 font-mono shadow-lg">
      <div>
        WS: <span className="text-zinc-300">{props.wsState}</span>
      </div>
      <div>
        WebRTC: <span className="text-zinc-300">{props.webrtcState}</span>
      </div>
      <div>
        det FPS:{" "}
        <span className="text-zinc-300">{props.detectionFps.toFixed(1)}</span>
      </div>
      <div>
        video FPS:{" "}
        <span className="text-zinc-300">{props.videoFps.toFixed(1)}</span>
      </div>
      <div>
        last frame age:{" "}
        <span className="text-zinc-300">
          {props.lastFrameAgeMs == null
            ? "—"
            : `${props.lastFrameAgeMs.toFixed(0)}ms`}
        </span>
      </div>
    </aside>
  );
}
