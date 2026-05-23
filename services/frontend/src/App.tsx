import { useCallback, useState } from "react";

import { createSession, deleteSession } from "./api/session";
import { CameraPicker } from "./components/CameraPicker";
import { DebugPanel } from "./components/DebugPanel";
import { VideoStage } from "./components/VideoStage";
import { useCamera } from "./hooks/useCamera";
import { useDetections } from "./hooks/useDetections";
import { useWebRTC } from "./hooks/useWebRTC";

type AppPhase = "idle" | "starting" | "running" | "error";

export function App(): JSX.Element {
  const camera = useCamera();
  const [phase, setPhase] = useState<AppPhase>("idle");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [appError, setAppError] = useState<string | null>(null);

  const detections = useDetections(phase === "running" ? sessionId : null);
  const webrtc = useWebRTC(
    phase === "running" ? camera.stream : null,
    sessionId,
  );

  const handleStart = useCallback(async () => {
    if (!camera.stream || !camera.selected) {
      // useCamera.select kicks off stream acquisition asynchronously; if the
      // stream isn't ready yet, the button shouldn't have been clickable, but
      // guard anyway.
      setAppError("Camera not ready");
      setPhase("error");
      return;
    }
    setPhase("starting");
    setAppError(null);
    const sid = crypto.randomUUID();
    try {
      await createSession(sid);
      setSessionId(sid);
      setPhase("running");
    } catch (e) {
      setAppError(e instanceof Error ? e.message : String(e));
      setPhase("error");
    }
  }, [camera.stream, camera.selected]);

  const handleStop = useCallback(async () => {
    const sid = sessionId;
    setPhase("idle");
    setSessionId(null);
    if (sid) {
      try {
        await deleteSession(sid);
      } catch {
        // Best-effort; the server has an idle-sweep that will clean up.
      }
    }
  }, [sessionId]);

  const handleReset = useCallback(() => {
    setAppError(null);
    setPhase("idle");
    setSessionId(null);
  }, []);

  return (
    <main className="min-h-screen flex flex-col items-center justify-center gap-6 p-8">
      <header className="text-center">
        <h1 className="text-3xl font-bold tracking-tight">Guitar Detect</h1>
        <p className="text-sm text-zinc-400">
          Point a camera at a guitar — get a brand and model lock-on.
        </p>
      </header>

      {phase === "idle" && (
        <CameraPicker camera={camera} onStart={handleStart} starting={false} />
      )}

      {phase === "starting" && (
        <p className="text-sm text-zinc-300">Starting session…</p>
      )}

      {phase === "running" && (
        <>
          <VideoStage
            stream={camera.stream}
            tracks={detections.event?.tracks ?? []}
          />
          <button
            type="button"
            onClick={handleStop}
            className="rounded bg-zinc-700 hover:bg-zinc-600 px-4 py-2 text-zinc-100"
          >
            Stop
          </button>
        </>
      )}

      {phase === "error" && (
        <div className="flex flex-col items-center gap-2">
          <p role="alert" className="text-rose-400">
            {appError ?? "Unknown error"}
          </p>
          <button
            type="button"
            onClick={handleReset}
            className="rounded bg-zinc-700 hover:bg-zinc-600 px-4 py-2 text-zinc-100"
          >
            Reset
          </button>
        </div>
      )}

      <DebugPanel
        wsState={detections.state}
        webrtcState={webrtc.state}
        detectionFps={0}
        videoFps={0}
        lastFrameAgeMs={null}
      />
    </main>
  );
}
