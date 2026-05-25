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

// Max thumbnail dimension. The actual aspect ratio comes from the bbox
// crop — boxes are usually portrait for guitars, so we cap the long edge
// and let the short edge scale. Quality of 0.92 keeps faces and inlays
// recognizable in screenshots without blowing up data-URL size.
const THUMBNAIL_MAX = 320;
const THUMBNAIL_QUALITY = 0.92;

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
        const thumb = captureThumbnail(video, canvasRef, t.bbox);
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
  bbox: [number, number, number, number],
): string | null {
  // Translate the normalized [0..1] bbox into source-pixel coordinates.
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  const sx = Math.max(0, Math.floor(bbox[0] * vw));
  const sy = Math.max(0, Math.floor(bbox[1] * vh));
  const sw = Math.max(1, Math.floor((bbox[2] - bbox[0]) * vw));
  const sh = Math.max(1, Math.floor((bbox[3] - bbox[1]) * vh));

  // Scale so the long edge fits THUMBNAIL_MAX while preserving aspect.
  const scale = Math.min(1, THUMBNAIL_MAX / Math.max(sw, sh));
  const dw = Math.max(1, Math.round(sw * scale));
  const dh = Math.max(1, Math.round(sh * scale));

  let canvas = canvasRef.current;
  if (!canvas) {
    canvas = document.createElement("canvas");
    canvasRef.current = canvas;
  }
  canvas.width = dw;
  canvas.height = dh;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  try {
    ctx.drawImage(video, sx, sy, sw, sh, 0, 0, dw, dh);
    return canvas.toDataURL("image/jpeg", THUMBNAIL_QUALITY);
  } catch {
    // taintedCanvas on cross-origin video; tolerate without crashing.
    return null;
  }
}
