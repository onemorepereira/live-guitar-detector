import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { CameraPicker } from "../src/components/CameraPicker";
import { DebugPanel } from "../src/components/DebugPanel";
import { VideoStage } from "../src/components/VideoStage";

describe("composition components mount without crashing", () => {
  it("CameraPicker", () => {
    const camera = {
      devices: [],
      selected: null,
      stream: null,
      error: null,
      select: () => {},
    };
    const { container } = render(
      <CameraPicker camera={camera} onStart={() => {}} />,
    );
    expect(container.querySelector("select")).toBeTruthy();
  });

  it("VideoStage renders a video element", () => {
    const { container } = render(<VideoStage stream={null} tracks={[]} />);
    expect(container.querySelector("video")).toBeTruthy();
  });

  it("DebugPanel returns null without ?debug=1", () => {
    const { container } = render(
      <DebugPanel
        wsState="idle"
        webrtcState="idle"
        detectionFps={0}
        videoFps={0}
        lastFrameAgeMs={null}
      />,
    );
    expect(container.children.length).toBe(0);
  });
});
