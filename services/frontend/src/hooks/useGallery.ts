import { useCallback, useRef, useState } from "react";

import type { TrackDetection } from "../types/detection";

export interface GalleryItem {
  track_id: number;
  brand: string; // from label.brand when capture happened
  model: string;
  confidence: number; // smoothed confidence at capture time
  thumbnailDataUrl: string;
  capturedAt: number; // unix ms
}

export interface UseGalleryResult {
  items: GalleryItem[];
  /** Called per detection event with current tracks + the live video element. */
  observe: (tracks: TrackDetection[], video: HTMLVideoElement | null) => void;
  /** Clear all captured items (e.g., on session reset). */
  clear: () => void;
}

const THUMBNAIL_W = 160;
const THUMBNAIL_H = 120;
const THUMBNAIL_QUALITY = 0.85;

/**
 * In-memory session gallery of unique guitar sightings.
 *
 * Captures a thumbnail from the live `<video>` the first time a track
 * becomes `stable` (with a non-null label) and dedupes by `track_id`
 * for the lifetime of the session — call `clear()` on reset.
 */
export function useGallery(): UseGalleryResult {
  const [items, setItems] = useState<GalleryItem[]>([]);
  // Track IDs we've already captured — dedupe key.
  const seenRef = useRef<Set<number>>(new Set());
  // Offscreen canvas reused across captures (cheap to construct once).
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const observe = useCallback(
    (tracks: TrackDetection[], video: HTMLVideoElement | null) => {
      if (!video || video.videoWidth === 0 || video.videoHeight === 0) return;
      const seen = seenRef.current;
      const newItems: GalleryItem[] = [];
      for (const t of tracks) {
        if (!t.stable || t.label === null) continue;
        if (seen.has(t.track_id)) continue;
        const thumb = captureThumbnail(video, canvasRef);
        if (thumb === null) continue;
        seen.add(t.track_id);
        newItems.push({
          track_id: t.track_id,
          brand: t.label.brand,
          model: t.label.model,
          confidence: t.label.confidence,
          thumbnailDataUrl: thumb,
          capturedAt: Date.now(),
        });
      }
      if (newItems.length > 0) {
        setItems((prev) => [...prev, ...newItems]);
      }
    },
    [],
  );

  const clear = useCallback(() => {
    seenRef.current.clear();
    setItems([]);
  }, []);

  return { items, observe, clear };
}

function captureThumbnail(
  video: HTMLVideoElement,
  canvasRef: React.MutableRefObject<HTMLCanvasElement | null>,
): string | null {
  let canvas = canvasRef.current;
  if (!canvas) {
    canvas = document.createElement("canvas");
    canvas.width = THUMBNAIL_W;
    canvas.height = THUMBNAIL_H;
    canvasRef.current = canvas;
  }
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  try {
    // Cover the canvas with the video frame, preserving aspect ratio via
    // a basic letterbox/pillarbox crop. For thumbnails this is fine even
    // if we lose a few pixels at the edges.
    ctx.drawImage(video, 0, 0, THUMBNAIL_W, THUMBNAIL_H);
    return canvas.toDataURL("image/jpeg", THUMBNAIL_QUALITY);
  } catch {
    // taintedCanvas on cross-origin video; tolerate without crashing.
    return null;
  }
}
