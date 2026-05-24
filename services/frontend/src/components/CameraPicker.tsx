import type { UseCameraResult } from "../hooks/useCamera";

export interface CameraPickerProps {
  camera: UseCameraResult;
  onStart: () => void;
  starting?: boolean;
}

/**
 * Camera dropdown + Start button. Stateless — selection lives in `useCamera`,
 * onStart fires the parent's session-create flow.
 */
export function CameraPicker({
  camera,
  onStart,
  starting,
}: CameraPickerProps): JSX.Element {
  // Stream must be live before Start so we don't kick off a session against
  // a camera that hasn't actually attached. `selected` alone isn't enough —
  // getUserMedia may still be in flight.
  const canStart = !!camera.selected && !!camera.stream && !starting;
  return (
    <div className="flex flex-col gap-3 max-w-md">
      <label className="flex flex-col gap-1 text-sm">
        <span className="font-medium text-zinc-200">Camera</span>
        <select
          className="rounded bg-zinc-900 border border-zinc-700 px-3 py-2 text-zinc-100"
          value={camera.selected ?? ""}
          onChange={(e) => camera.select(e.target.value)}
          disabled={camera.devices.length === 0}
        >
          <option value="" disabled>
            {camera.devices.length === 0
              ? "No cameras detected"
              : "Select a camera…"}
          </option>
          {camera.devices.map((d) => (
            <option key={d.deviceId} value={d.deviceId}>
              {d.label || `Camera ${d.deviceId.slice(0, 6)}`}
            </option>
          ))}
        </select>
      </label>
      {camera.error && (
        <p role="alert" className="text-sm text-rose-400">
          {camera.error}
        </p>
      )}
      <button
        type="button"
        onClick={onStart}
        disabled={!canStart}
        className="rounded bg-amber-500 hover:bg-amber-400 disabled:bg-zinc-700 disabled:text-zinc-400 px-4 py-2 font-semibold text-zinc-900 transition"
      >
        {starting ? "Starting…" : "Start"}
      </button>
    </div>
  );
}
