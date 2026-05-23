import { useEffect, useRef, useState } from "react";

import { HUD } from "./HUD";
import type { TrackDetection } from "../types/detection";

export interface VideoStageProps {
  stream: MediaStream | null;
  tracks: TrackDetection[];
}

/**
 * Renders a <video> element with the live MediaStream and overlays the
 * canvas HUD. Element dimensions are tracked via ResizeObserver so the
 * HUD's denormalizeBbox math has accurate elW/elH at all times.
 */
export function VideoStage({ stream, tracks }: VideoStageProps): JSX.Element {
  const videoRef = useRef<HTMLVideoElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [videoSize, setVideoSize] = useState({ videoW: 1280, videoH: 720 });
  const [elementSize, setElementSize] = useState({ elW: 0, elH: 0 });

  // Attach the stream
  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    if (stream) {
      v.srcObject = stream;
      void v.play().catch(() => {
        // autoplay can fail without user gesture; user-visible play() will retry
      });
    } else {
      v.srcObject = null;
    }
  }, [stream]);

  // Track intrinsic video size
  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    const onMeta = () => {
      setVideoSize({
        videoW: v.videoWidth || 1280,
        videoH: v.videoHeight || 720,
      });
    };
    v.addEventListener("loadedmetadata", onMeta);
    return () => v.removeEventListener("loadedmetadata", onMeta);
  }, [stream]);

  // Track rendered element size
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const obs = new ResizeObserver((entries) => {
      const { width, height } = entries[0].contentRect;
      setElementSize({ elW: Math.round(width), elH: Math.round(height) });
    });
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  return (
    <div
      ref={containerRef}
      className="relative w-full max-w-5xl aspect-video bg-black rounded overflow-hidden"
    >
      <video
        ref={videoRef}
        autoPlay
        muted
        playsInline
        className="absolute inset-0 w-full h-full object-contain"
      />
      {elementSize.elW > 0 && (
        <HUD tracks={tracks} videoRect={{ ...videoSize, ...elementSize }} />
      )}
    </div>
  );
}
