/**
 * Hand masking — JS counterpart of `crop_dataset.hand_mask`.
 *
 * Why this exists: the dataset is shot against a plain green screen, a webcam
 * sees a room, and HOG is a gradient descriptor. Background edges inside the
 * crop box compete directly with finger edges. Measured on the unmasked model,
 * adding texture to the outer ring of a crop changes the predicted letter
 * 66.7 % of the time — which is exactly the "same pose, random letter"
 * behaviour reported from live testing.
 *
 * The mask is the convex hull of the 21 landmarks, dilated, with everything
 * outside set flat so it contributes no gradients.
 *
 * The edge is FEATHERED. A hard boundary is itself a strong gradient, so HOG
 * keys on it, and then a one-pixel difference in rasterisation flips the
 * prediction -- an earlier hard-edged version disagreed with Python on 11 of 25
 * crops while only 2.3 % of pixels differed. Both sides now define the mask as
 * a signed distance to the hull with a linear falloff, so they agree by
 * construction rather than by luck.
 */

export const MASK_DILATE = 0.16;    // hard radius, fraction of box side
export const MASK_FEATHER = 0.08;   // soft falloff beyond it
export const MASK_FILL = 128;

/** Convex hull via Andrew's monotone chain. Points are [x, y] pairs. */
export function convexHull(points) {
  const pts = points.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (pts.length < 3) return pts;
  const cross = (o, a, b) =>
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower = [];
  for (const p of pts) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0)
      lower.pop();
    lower.push(p);
  }
  const upper = [];
  for (let i = pts.length - 1; i >= 0; i--) {
    const p = pts[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0)
      upper.pop();
    upper.push(p);
  }
  lower.pop(); upper.pop();
  return lower.concat(upper);
}

function pointInPoly(x, y, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

function distToPoly(x, y, poly) {
  let best = Infinity;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    const dx = xj - xi, dy = yj - yi;
    const len2 = dx * dx + dy * dy;
    let t = len2 ? ((x - xi) * dx + (y - yi) * dy) / len2 : 0;
    t = Math.max(0, Math.min(1, t));
    const px = xi + t * dx, py = yi + t * dy;
    const d = Math.hypot(x - px, y - py);
    if (d < best) best = d;
  }
  return best;
}

/**
 * Apply the hand mask in place to a `size`x`size` grayscale array.
 *
 * @param {Float64Array} gray   row-major, length size*size
 * @param {Array<[number,number]>} ptsPx landmarks in VIDEO pixel coordinates
 * @param {{cx:number, cy:number, side:number}} box the crop box, video pixels
 * @param {number} size crop side (64)
 */
export function applyHandMask(gray, ptsPx, box, size,
                              { dilate = MASK_DILATE, feather = MASK_FEATHER,
                                fill = MASK_FILL } = {}) {
  const scale = size / box.side;
  const ox = box.cx - box.side / 2, oy = box.cy - box.side / 2;
  const local = ptsPx.map(([x, y]) => [(x - ox) * scale, (y - oy) * scale]);
  const hull = convexHull(local);
  if (hull.length < 3) return gray;

  const r = dilate * size;
  const f = Math.max(1e-6, feather * size);

  for (let yy = 0; yy < size; yy++) {
    for (let xx = 0; xx < size; xx++) {
      const i = yy * size + xx;
      const d = distToPoly(xx, yy, hull);
      const sd = pointInPoly(xx, yy, hull) ? d : -d;     // + inside, matching cv2
      const a = Math.min(1, Math.max(0, (sd + r + f) / f));
      if (a < 1) gray[i] = gray[i] * a + fill * (1 - a);
    }
  }
  return gray;
}
