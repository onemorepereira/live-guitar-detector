import { useCallback, useEffect, useRef, useState } from "react";

export interface UseCameraResult {
  /** All `videoinput` MediaDeviceInfo entries reported by the browser. */
  devices: MediaDeviceInfo[];
  /** Currently-selected deviceId, or null before the user picks one. */
  selected: string | null;
  /** Active MediaStream after a successful getUserMedia, or null. */
  stream: MediaStream | null;
  /** User-facing error string (permission denied, no camera, etc.), or null. */
  error: string | null;
  /** Select a device by id; kicks off getUserMedia. */
  select: (deviceId: string) => void;
}

/**
 * useCamera: enumerate available video input devices and expose a stream
 * for the user-selected device. Errors are surfaced through state — the
 * hook never throws into React's render path.
 */
export function useCamera(): UseCameraResult {
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [error, setError] = useState<string | null>(null);
  const currentStreamRef = useRef<MediaStream | null>(null);

  // Enumerate video inputs once on mount.
  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const list = await navigator.mediaDevices.enumerateDevices();
        if (cancelled) return;
        setDevices(list.filter((d) => d.kind === "videoinput"));
      } catch (e) {
        if (cancelled) return;
        setError(formatError(e, "Could not enumerate camera devices"));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // Release any active stream on unmount.
  useEffect(() => {
    return () => {
      currentStreamRef.current?.getTracks().forEach((t) => t.stop());
      currentStreamRef.current = null;
    };
  }, []);

  const select = useCallback((deviceId: string) => {
    setSelected(deviceId);
    setError(null);
    void (async () => {
      try {
        // Stop the previous stream's tracks before starting a new one so
        // the camera indicator turns off and the device is freed.
        currentStreamRef.current?.getTracks().forEach((t) => t.stop());
        const s = await navigator.mediaDevices.getUserMedia({
          video: { deviceId: { exact: deviceId } },
        });
        currentStreamRef.current = s;
        setStream(s);
      } catch (e) {
        currentStreamRef.current = null;
        setStream(null);
        setError(formatError(e, "Could not start camera"));
      }
    })();
  }, []);

  return { devices, selected, stream, error, select };
}

function formatError(e: unknown, fallback: string): string {
  if (e instanceof DOMException) {
    if (e.name === "NotAllowedError") return "Camera permission denied";
    if (e.name === "NotFoundError") return "No camera found";
    if (e.name === "OverconstrainedError") {
      return "Camera does not match the requested constraints";
    }
    return `${fallback}: ${e.message}`;
  }
  if (e instanceof Error) return `${fallback}: ${e.message}`;
  return `${fallback}: ${String(e)}`;
}
