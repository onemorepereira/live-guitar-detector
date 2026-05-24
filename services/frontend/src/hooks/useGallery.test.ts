import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useGallery } from "./useGallery";
import type { TrackDetection } from "../types/detection";

// Stub <video> + canvas drawImage/toDataURL.

function fakeVideo(): HTMLVideoElement {
  return {
    videoWidth: 1280,
    videoHeight: 720,
  } as unknown as HTMLVideoElement;
}

beforeEach(() => {
  // jsdom canvas: getContext returns null by default. Patch to a spy that
  // exposes drawImage + toDataURL.
  HTMLCanvasElement.prototype.getContext = vi.fn(() => ({
    drawImage: vi.fn(),
  })) as unknown as HTMLCanvasElement["getContext"];
  HTMLCanvasElement.prototype.toDataURL = vi.fn(
    () => "data:image/jpeg;base64,FAKE",
  ) as unknown as HTMLCanvasElement["toDataURL"];
});

afterEach(() => {
  vi.restoreAllMocks();
});

function gibsonStable(track_id: number): TrackDetection {
  return {
    track_id,
    bbox: [0.1, 0.1, 0.5, 0.5],
    detection_confidence: 0.9,
    label: { brand: "Gibson", model: "Les Paul", confidence: 0.85 },
    stable: true,
    age_frames: 30,
  };
}

describe("useGallery", () => {
  it("captures a stable track once and exposes it as an item", () => {
    const { result } = renderHook(() => useGallery());
    act(() => result.current.observe([gibsonStable(1)], fakeVideo()));
    expect(result.current.items).toHaveLength(1);
    expect(result.current.items[0]).toMatchObject({
      track_id: 1,
      brand: "Gibson",
      model: "Les Paul",
      confidence: 0.85,
      thumbnailDataUrl: "data:image/jpeg;base64,FAKE",
    });
    expect(typeof result.current.items[0].capturedAt).toBe("number");
  });

  it("does not capture the same track_id twice", () => {
    const { result } = renderHook(() => useGallery());
    act(() => result.current.observe([gibsonStable(1)], fakeVideo()));
    act(() => result.current.observe([gibsonStable(1)], fakeVideo()));
    expect(result.current.items).toHaveLength(1);
  });

  it("captures different track_ids with the same label as distinct sightings", () => {
    const { result } = renderHook(() => useGallery());
    act(() =>
      result.current.observe([gibsonStable(1), gibsonStable(2)], fakeVideo()),
    );
    expect(result.current.items).toHaveLength(2);
    expect(result.current.items.map((i) => i.track_id).sort()).toEqual([1, 2]);
  });

  it("skips unstable tracks", () => {
    const { result } = renderHook(() => useGallery());
    const t: TrackDetection = {
      ...gibsonStable(1),
      stable: false,
      label: null,
    };
    act(() => result.current.observe([t], fakeVideo()));
    expect(result.current.items).toHaveLength(0);
  });

  it("skips when video is not ready (no dimensions)", () => {
    const { result } = renderHook(() => useGallery());
    const blankVideo = { videoWidth: 0, videoHeight: 0 } as HTMLVideoElement;
    act(() => result.current.observe([gibsonStable(1)], blankVideo));
    expect(result.current.items).toHaveLength(0);
  });

  it("skips when video is null", () => {
    const { result } = renderHook(() => useGallery());
    act(() => result.current.observe([gibsonStable(1)], null));
    expect(result.current.items).toHaveLength(0);
  });

  it("clear() resets items and seen set so the same track_id can re-capture", () => {
    const { result } = renderHook(() => useGallery());
    act(() => result.current.observe([gibsonStable(1)], fakeVideo()));
    expect(result.current.items).toHaveLength(1);
    act(() => result.current.clear());
    expect(result.current.items).toHaveLength(0);
    act(() => result.current.observe([gibsonStable(1)], fakeVideo()));
    expect(result.current.items).toHaveLength(1);
  });
});
