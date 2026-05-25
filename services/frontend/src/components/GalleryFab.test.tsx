import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { GalleryFab } from "./GalleryFab";
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
];

describe("GalleryFab", () => {
  it("renders nothing when there are no items", () => {
    const { container } = render(
      <GalleryFab items={[]} highlightedTrackId={null} onSelect={() => {}} />,
    );
    expect(container.firstChild).toBeNull();
  });

  it("shows a pill with the capture count, sheet hidden by default", () => {
    render(
      <GalleryFab
        items={items}
        highlightedTrackId={null}
        onSelect={() => {}}
      />,
    );
    expect(
      screen.getByRole("button", { name: /open captures \(1\)/i }),
    ).toBeInTheDocument();
    // Sheet content isn't in the DOM until opened.
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("opens the sheet when the pill is tapped", () => {
    render(
      <GalleryFab
        items={items}
        highlightedTrackId={null}
        onSelect={() => {}}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /open captures/i }));
    expect(screen.getByRole("dialog")).toBeInTheDocument();
    // Item is rendered inside the sheet (re-uses GalleryPanel).
    expect(screen.getByText("Gibson Les Paul")).toBeInTheDocument();
  });

  it("selecting an item from the sheet calls onSelect and closes the sheet", () => {
    const onSelect = vi.fn();
    render(
      <GalleryFab
        items={items}
        highlightedTrackId={null}
        onSelect={onSelect}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /open captures/i }));
    fireEvent.click(
      screen.getByRole("button", { name: /gibson les paul/i, pressed: false }),
    );
    expect(onSelect).toHaveBeenCalledWith(1);
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("close button dismisses the sheet without selecting", () => {
    const onSelect = vi.fn();
    render(
      <GalleryFab
        items={items}
        highlightedTrackId={null}
        onSelect={onSelect}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /open captures/i }));
    fireEvent.click(screen.getByRole("button", { name: /close captures/i }));
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(onSelect).not.toHaveBeenCalled();
  });
});
