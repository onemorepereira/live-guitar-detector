import { useCallback, useEffect, useRef, useState } from "react";

import { createSession, deleteSession } from "./api/session";
import { CameraPicker } from "./components/CameraPicker";
import { DebugPanel } from "./components/DebugPanel";
import { GalleryPanel } from "./components/GalleryPanel";
import { VideoStage } from "./components/VideoStage";
import { useCamera } from "./hooks/useCamera";
import { useDetections } from "./hooks/useDetections";
import { useGallery } from "./hooks/useGallery";
import { useWebRTC } from "./hooks/useWebRTC";

type AppPhase = "idle" | "starting" | "running" | "error";

export function App(): JSX.Element {
  const camera = useCamera();
  const [phase, setPhase] = useState<AppPhase>("idle");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [appError, setAppError] = useState<string | null>(null);
  const [highlightedTrackId, setHighlightedTrackId] = useState<number | null>(
    null,
  );

  const detections = useDetections(phase === "running" ? sessionId : null);
  const webrtc = useWebRTC(
    phase === "running" ? camera.stream : null,
    sessionId,
  );
  const gallery = useGallery();
  const videoRef = useRef<HTMLVideoElement | null>(null);

  // When detection events arrive, observe for gallery capture.
  useEffect(() => {
    if (!detections.event) return;
    gallery.observe(detections.event.tracks, videoRef.current);
  }, [detections.event, gallery]);

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
    setHighlightedTrackId(null);
    gallery.clear();
    if (sid) {
      try {
        await deleteSession(sid);
      } catch {
        // Best-effort; the server has an idle-sweep that will clean up.
      }
    }
  }, [sessionId, gallery]);

  const handleReset = useCallback(() => {
    setAppError(null);
    setPhase("idle");
    setSessionId(null);
    setHighlightedTrackId(null);
    gallery.clear();
  }, [gallery]);

  const handleVideoReady = useCallback((el: HTMLVideoElement | null) => {
    videoRef.current = el;
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
          <div className="flex flex-col md:flex-row gap-4 w-full max-w-6xl">
            <div className="flex-1">
              <VideoStage
                stream={camera.stream}
                tracks={detections.event?.tracks ?? []}
                highlightedTrackId={highlightedTrackId}
                onVideoReady={handleVideoReady}
              />
            </div>
            <GalleryPanel
              items={gallery.items}
              highlightedTrackId={highlightedTrackId}
              onSelect={setHighlightedTrackId}
            />
          </div>
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
