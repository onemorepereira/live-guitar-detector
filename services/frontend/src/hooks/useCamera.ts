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

  // Enumerate video inputs once on mount. Chromium anonymizes the result
  // (empty `deviceId` / `label`) until camera permission has been granted
  // for this origin, so if we see anonymized entries we trigger a one-shot
  // permission prompt via `getUserMedia({video:true})`, close the
  // resulting stream immediately, then re-enumerate to get real ids.
  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        let list = (await navigator.mediaDevices.enumerateDevices()).filter(
          (d) => d.kind === "videoinput",
        );
        const anonymized =
          list.length > 0 &&
          list.every((d) => d.deviceId === "" || d.label === "");
        if (anonymized) {
          // Prompt for permission; immediately release the stream.
          const probe = await navigator.mediaDevices.getUserMedia({
            video: true,
          });
          probe.getTracks().forEach((t) => t.stop());
          if (cancelled) return;
          list = (await navigator.mediaDevices.enumerateDevices()).filter(
            (d) => d.kind === "videoinput",
          );
        }
        if (cancelled) return;
        setDevices(list);
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
    if (!deviceId) {
      // Empty id can slip in if the dropdown was opened and dismissed
      // without picking a real device; treat as "no selection".
      setSelected(null);
      return;
    }
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
  // Duck-type on the `name` property rather than class instance checks.
  // OverconstrainedError is its own class (not a DOMException), and
  // jsdom's DOMException may not extend Error — so `instanceof` is
  // unreliable across environments.
  if (e && typeof e === "object" && "name" in e) {
    const name = String((e as { name: unknown }).name);
    const message =
      "message" in e ? String((e as { message: unknown }).message) : String(e);
    if (name === "OverconstrainedError") {
      return "Camera does not match the requested constraints";
    }
    if (name === "NotAllowedError") return "Camera permission denied";
    if (name === "NotFoundError") return "No camera found";
    return `${fallback}: ${message}`;
  }
  return `${fallback}: ${String(e)}`;
}
