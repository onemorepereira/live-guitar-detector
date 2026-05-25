import { useState } from "react";

import { GalleryPanel } from "./GalleryPanel";
import type { GalleryItem } from "../hooks/useGallery";

export interface GalleryFabProps {
  items: GalleryItem[];
  highlightedTrackId: number | null;
  onSelect: (trackId: number | null) => void;
}

/**
 * Mobile-first captures launcher.
 *
 * Default state: a small floating pill in the bottom-right showing the
 * capture count. Nothing obscures the camera.
 *
 * Tapped: a bottom sheet slides up over a dimmed backdrop, containing
 * the full GalleryPanel (thumbnails + brand/model + per-row download).
 * Tapping the backdrop, the close button, or a select button dismisses
 * the sheet so the user can see the highlighted bbox on the canvas.
 */
export function GalleryFab({
  items,
  highlightedTrackId,
  onSelect,
}: GalleryFabProps): JSX.Element | null {
  const [open, setOpen] = useState(false);
  if (items.length === 0) return null;

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        aria-label={`Open captures (${items.length})`}
        className="fixed bottom-6 right-4 z-20 rounded-full bg-zinc-900/80 hover:bg-zinc-800 px-4 py-3 text-sm font-medium text-zinc-100 shadow-lg backdrop-blur-sm focus:outline-none focus:ring-2 focus:ring-amber-400"
      >
        Captures · {items.length}
      </button>

      {open && (
        <>
          <div
            className="fixed inset-0 z-30 bg-black/60"
            onClick={() => setOpen(false)}
            aria-hidden="true"
          />
          <div
            role="dialog"
            aria-label="Captures this session"
            className="fixed left-0 right-0 bottom-0 z-30 max-h-[70vh] rounded-t-2xl bg-zinc-950 shadow-2xl flex flex-col"
          >
            <div className="flex items-center justify-between px-4 pt-3 pb-2">
              <div className="flex flex-col">
                <span
                  aria-hidden="true"
                  className="mx-auto h-1 w-10 rounded-full bg-zinc-700 mb-2"
                />
                <h2 className="text-sm uppercase tracking-wider text-zinc-400">
                  Captures ({items.length})
                </h2>
              </div>
              <button
                type="button"
                onClick={() => setOpen(false)}
                aria-label="Close captures"
                className="rounded p-2 text-zinc-300 hover:bg-zinc-800 focus:outline-none focus:ring-2 focus:ring-amber-400"
              >
                ✕
              </button>
            </div>
            <div className="overflow-y-auto px-3 pb-4">
              <GalleryPanel
                items={items}
                highlightedTrackId={highlightedTrackId}
                onSelect={(id) => {
                  onSelect(id);
                  setOpen(false); // dismiss so the highlighted bbox is visible.
                }}
              />
            </div>
          </div>
        </>
      )}
    </>
  );
}
