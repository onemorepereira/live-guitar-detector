/**
 * Bounding-box geometry helpers.
 *
 * The inference worker emits bboxes in normalized video coordinates
 * (`[x1, y1, x2, y2]` each in `0..1`, relative to the source frame).
 * The browser draws those bboxes on a canvas overlaid on a `<video>`
 * element whose CSS pixel size rarely matches the source resolution.
 *
 * "Denormalize" here means projecting those `0..1` coordinates into the
 * element's CSS pixel space — i.e. the same coordinate system the
 * overlay canvas uses.
 *
 * Why letterboxing matters: HTML video defaults to `object-fit: contain`,
 * which scales the source uniformly to fit the element and pads the
 * remaining space with bars (letterbox = horizontal bars top/bottom,
 * pillarbox = vertical bars left/right). The visible video occupies a
 * sub-rectangle of the element, so we must apply both the uniform scale
 * and the centering offset; otherwise the overlay drifts off the actual
 * pixels.
 */

export function denormalizeBbox(
  bbox: readonly [number, number, number, number],
  geometry: { videoW: number; videoH: number; elW: number; elH: number },
): [number, number, number, number] {
  const scale = Math.min(
    geometry.elW / geometry.videoW,
    geometry.elH / geometry.videoH,
  );
  const renderedW = geometry.videoW * scale;
  const renderedH = geometry.videoH * scale;
  const offsetX = (geometry.elW - renderedW) / 2;
  const offsetY = (geometry.elH - renderedH) / 2;
  const [x1, y1, x2, y2] = bbox;
  return [
    offsetX + x1 * renderedW,
    offsetY + y1 * renderedH,
    offsetX + x2 * renderedW,
    offsetY + y2 * renderedH,
  ];
}
