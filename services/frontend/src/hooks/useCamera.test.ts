import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { useCamera } from "./useCamera";

function setMediaDevices(mock: Partial<MediaDevices>) {
  // jsdom doesn't ship navigator.mediaDevices; install a stub per-test.
  Object.defineProperty(globalThis.navigator, "mediaDevices", {
    value: mock,
    configurable: true,
    writable: true,
  });
}

const fakeVideoDevice: MediaDeviceInfo = {
  deviceId: "cam-1",
  kind: "videoinput",
  label: "Front Camera",
  groupId: "g1",
  toJSON() {
    return this;
  },
};

const fakeAudioDevice: MediaDeviceInfo = {
  deviceId: "mic-1",
  kind: "audioinput",
  label: "Mic",
  groupId: "g2",
  toJSON() {
    return this;
  },
};

function fakeStream(): MediaStream {
  const stop = vi.fn();
  const track = { stop, kind: "video" } as unknown as MediaStreamTrack;
  return {
    getTracks: () => [track],
  } as unknown as MediaStream;
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("useCamera", () => {
  it("enumerates videoinput devices on mount", async () => {
    const enumerateDevices = vi.fn(async () => [
      fakeVideoDevice,
      fakeAudioDevice,
    ]);
    const getUserMedia = vi.fn();
    setMediaDevices({
      enumerateDevices,
      getUserMedia,
    } as Partial<MediaDevices>);

    const { result } = renderHook(() => useCamera());
    await waitFor(() => expect(result.current.devices.length).toBe(1));
    expect(result.current.devices[0].deviceId).toBe("cam-1");
    expect(enumerateDevices).toHaveBeenCalledTimes(1);
  });

  it("selecting a device calls getUserMedia with exact deviceId and stores the stream", async () => {
    const enumerateDevices = vi.fn(async () => [fakeVideoDevice]);
    const stream = fakeStream();
    const getUserMedia = vi.fn(async () => stream);
    setMediaDevices({
      enumerateDevices,
      getUserMedia,
    } as Partial<MediaDevices>);

    const { result } = renderHook(() => useCamera());
    await waitFor(() => expect(result.current.devices.length).toBe(1));

    act(() => {
      result.current.select("cam-1");
    });

    await waitFor(() => expect(result.current.stream).toBe(stream));
    expect(getUserMedia).toHaveBeenCalledWith({
      video: { deviceId: { exact: "cam-1" } },
    });
    expect(result.current.selected).toBe("cam-1");
    expect(result.current.error).toBeNull();
  });

  it("permission denial surfaces as error state, not a thrown exception", async () => {
    const enumerateDevices = vi.fn(async () => [fakeVideoDevice]);
    const denied = new DOMException("permission denied", "NotAllowedError");
    const getUserMedia = vi.fn(async () => {
      throw denied;
    });
    setMediaDevices({
      enumerateDevices,
      getUserMedia,
    } as Partial<MediaDevices>);

    const { result } = renderHook(() => useCamera());
    await waitFor(() => expect(result.current.devices.length).toBe(1));

    act(() => {
      result.current.select("cam-1");
    });

    await waitFor(() =>
      expect(result.current.error).toBe("Camera permission denied"),
    );
    expect(result.current.stream).toBeNull();
  });

  it("selecting a second device stops the first stream's tracks", async () => {
    const enumerateDevices = vi.fn(async () => [
      fakeVideoDevice,
      { ...fakeVideoDevice, deviceId: "cam-2" },
    ]);
    const stream1 = fakeStream();
    const stream2 = fakeStream();
    let callIndex = 0;
    const getUserMedia = vi.fn(async () => {
      callIndex += 1;
      return callIndex === 1 ? stream1 : stream2;
    });
    setMediaDevices({
      enumerateDevices,
      getUserMedia,
    } as Partial<MediaDevices>);

    const { result } = renderHook(() => useCamera());
    await waitFor(() => expect(result.current.devices.length).toBe(2));

    act(() => {
      result.current.select("cam-1");
    });
    await waitFor(() => expect(result.current.stream).toBe(stream1));

    act(() => {
      result.current.select("cam-2");
    });
    await waitFor(() => expect(result.current.stream).toBe(stream2));

    const stream1Stop = stream1.getTracks()[0].stop as ReturnType<typeof vi.fn>;
    expect(stream1Stop).toHaveBeenCalled();
  });
});
