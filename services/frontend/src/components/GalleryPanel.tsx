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

function downloadItem(item: GalleryItem): void {
  const safe = (s: string) =>
    s.replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^-+|-+$/g, "");
  const name = `${safe(item.brand)}-${safe(item.model)}-track${item.track_id}-${item.capturedAt}.jpg`;
  const a = document.createElement("a");
  a.href = item.thumbnailDataUrl;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
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
          <div
            key={item.track_id}
            className={
              "flex items-center gap-2 rounded p-2 transition " +
              (isHighlighted
                ? "bg-amber-500/20 ring-1 ring-amber-400"
                : "bg-zinc-900 hover:bg-zinc-800")
            }
          >
            <button
              type="button"
              onClick={() => onSelect(isHighlighted ? null : item.track_id)}
              className="flex items-center gap-3 flex-1 text-left focus:outline-none focus:ring-2 focus:ring-amber-400 rounded"
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
            <button
              type="button"
              onClick={() => downloadItem(item)}
              aria-label={`Download ${item.brand} ${item.model} crop`}
              className="p-2 rounded text-zinc-300 hover:bg-zinc-700 focus:outline-none focus:ring-2 focus:ring-amber-400"
            >
              ⬇
            </button>
          </div>
        );
      })}
    </aside>
  );
}
