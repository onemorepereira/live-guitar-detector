import type { GalleryItem } from "../hooks/useGallery";

export interface GalleryPanelProps {
  items: GalleryItem[];
  highlightedTrackId: number | null;
  /**
   * Called with a `track_id` when an item is selected, or `null` when the
   * currently highlighted item is clicked again (toggle off).
   */
  onSelect: (trackId: number | null) => void;
}

/**
 * Side panel listing every unique guitar sighting captured this session.
 * Pure presentational component — the parent owns both the items list
 * (via `useGallery`) and the highlight state.
 */
export function GalleryPanel({
  items,
  highlightedTrackId,
  onSelect,
}: GalleryPanelProps): JSX.Element {
  if (items.length === 0) {
    return (
      <aside className="w-full md:w-64 p-4 text-sm text-zinc-400 border border-zinc-800 rounded">
        No guitars locked on yet.
      </aside>
    );
  }
  return (
    <aside className="w-full md:w-64 flex flex-col gap-2 p-3 border border-zinc-800 rounded max-h-[60vh] overflow-y-auto">
      <h2 className="text-xs uppercase tracking-wider text-zinc-400 px-1">
        Seen this session
      </h2>
      {items.map((item) => {
        const isHighlighted = item.track_id === highlightedTrackId;
        return (
          <button
            key={item.track_id}
            type="button"
            onClick={() => onSelect(isHighlighted ? null : item.track_id)}
            className={
              "flex items-center gap-3 rounded p-2 text-left transition focus:outline-none focus:ring-2 focus:ring-amber-400 " +
              (isHighlighted
                ? "bg-amber-500/20 ring-1 ring-amber-400"
                : "bg-zinc-900 hover:bg-zinc-800")
            }
            aria-pressed={isHighlighted}
          >
            <img
              src={item.thumbnailDataUrl}
              alt={`Track ${item.track_id}: ${item.brand} ${item.model}`}
              className="h-12 w-16 object-cover rounded bg-black"
            />
            <div className="flex flex-col min-w-0">
              <span className="text-sm font-medium truncate">
                {item.brand} {item.model}
              </span>
              <span className="text-xs text-zinc-400">
                #{item.track_id} · {Math.round(item.confidence * 100)}%
              </span>
            </div>
          </button>
        );
      })}
    </aside>
  );
}
