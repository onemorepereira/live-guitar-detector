/**
 * Detection event types shared between the gateway, worker, and frontend.
 *
 * These are the wire-level shapes produced by the inference worker and
 * forwarded to the browser via the gateway WebSocket. They mirror
 * DESIGN.md §5.1 exactly; do not diverge field names or literal unions
 * without updating the design doc first.
 */

export type DetectionEvent = {
  session_id: string;
  frame_id: number; // monotonic per session, set by gateway at ingest
  frame_ts: number; // unix ms, set by gateway at ingest
  inference_ts: number; // unix ms, set by worker on emit
  tracks: TrackDetection[];
};

export type TrackDetection = {
  track_id: number; // assigned by ByteTrack, stable per object
  bbox: [number, number, number, number]; // [x1, y1, x2, y2] normalized 0..1
  detection_confidence: number; // YOLO confidence 0..1
  label: ClassificationLabel | null; // null while vote is still warming up
  stable: boolean; // true once vote window is full
  age_frames: number; // frames since track first appeared
};

export type ClassificationLabel = {
  brand: "Gibson" | "Fender" | "Unknown";
  model:
    | "Les Paul"
    | "SG"
    | "Explorer"
    | "Flying V"
    | "Stratocaster"
    | "Telecaster"
    | "Unknown";
  confidence: number; // smoothed vote score 0..1
};
