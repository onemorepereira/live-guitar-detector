import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { GalleryPanel } from "./GalleryPanel";
import type { GalleryItem } from "../hooks/useGallery";

const items: GalleryItem[] = [
  {
    track_id: 1,
    brand: "Gibson",
    model: "Les Paul",
    confidence: 0.85,
    thumbnailDataUrl: "data:image/jpeg;base64,A",
    capturedAt: 1000,
  },
  {
    track_id: 2,
    brand: "Fender",
    model: "Stratocaster",
    confidence: 0.78,
    thumbnailDataUrl: "data:image/jpeg;base64,B",
    capturedAt: 2000,
  },
];

describe("GalleryPanel", () => {
  it("renders an empty-state when no items", () => {
    render(
      <GalleryPanel items={[]} highlightedTrackId={null} onSelect={() => {}} />,
    );
    expect(screen.getByText(/No guitars locked on/i)).toBeInTheDocument();
  });

  it("renders one button per item with label + confidence", () => {
    render(
      <GalleryPanel
        items={items}
        highlightedTrackId={null}
        onSelect={() => {}}
      />,
    );
    expect(screen.getByText("Gibson Les Paul")).toBeInTheDocument();
    expect(screen.getByText("Fender Stratocaster")).toBeInTheDocument();
    expect(screen.getByText(/#1 · 85%/)).toBeInTheDocument();
  });

  it("clicking an item calls onSelect with its track_id", () => {
    const onSelect = vi.fn();
    render(
      <GalleryPanel
        items={items}
        highlightedTrackId={null}
        onSelect={onSelect}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /gibson les paul/i }));
    expect(onSelect).toHaveBeenCalledWith(1);
  });

  it("clicking the highlighted item again clears the highlight", () => {
    const onSelect = vi.fn();
    render(
      <GalleryPanel items={items} highlightedTrackId={1} onSelect={onSelect} />,
    );
    fireEvent.click(screen.getByRole("button", { name: /gibson les paul/i }));
    expect(onSelect).toHaveBeenCalledWith(null);
  });

  it("marks the highlighted item as pressed for accessibility", () => {
    render(
      <GalleryPanel items={items} highlightedTrackId={2} onSelect={() => {}} />,
    );
    expect(
      screen.getByRole("button", { name: /fender strat/i }),
    ).toHaveAttribute("aria-pressed", "true");
    expect(
      screen.getByRole("button", { name: /gibson les paul/i }),
    ).toHaveAttribute("aria-pressed", "false");
  });
});
