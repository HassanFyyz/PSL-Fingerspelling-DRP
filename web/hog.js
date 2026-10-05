/**
 * Histogram of Oriented Gradients — direct port of `ref_hog` in export_model.py,
 * which is itself verified to match `skimage.feature.hog` to ~1e-7.
 *
 * Why HOG and not the raw image: measured leave-one-signer-out, raw pixels
 * classify letters at 0.74x the majority baseline (i.e. worse than guessing)
 * while HOG reaches 1.31x. HOG encodes local gradient *orientation* and
 * discards absolute intensity, so it keeps finger edge structure and throws
 * away lighting, skin tone and clothing — which are signer identity.
 *
 * Any drift between this and the Python version silently breaks the classifier
 * with no visible error, so web/test_parity.mjs checks them against each other
 * on real crops.
 */

export const ORIENTATIONS = 9;
export const PIXELS_PER_CELL = 8;
export const CELLS_PER_BLOCK = 2;
export const EPS = 1e-5;

/**
 * @param {Float64Array|number[]} gray row-major grayscale, length size*size, 0..255
 * @param {number} size image side in pixels (64)
 * @returns {Float32Array} feature vector
 */
export function hog(gray, size,
                    { orientations = ORIENTATIONS, ppc = PIXELS_PER_CELL,
                      cpb = CELLS_PER_BLOCK, eps = EPS } = {}) {
  const R = size, C = size;

  // Central-difference gradients with zeroed borders, matching
  // skimage's _hog_channel_gradient.
  const gRow = new Float64Array(R * C);
  const gCol = new Float64Array(R * C);
  for (let r = 1; r < R - 1; r++) {
    for (let c = 0; c < C; c++) {
      gRow[r * C + c] = gray[(r + 1) * C + c] - gray[(r - 1) * C + c];
    }
  }
  for (let r = 0; r < R; r++) {
    for (let c = 1; c < C - 1; c++) {
      gCol[r * C + c] = gray[r * C + (c + 1)] - gray[r * C + (c - 1)];
    }
  }

  const nr = Math.floor(R / ppc), nc = Math.floor(C / ppc);
  const hist = new Float64Array(nr * nc * orientations);
  const step = 180.0 / orientations;
  const cellArea = ppc * ppc;

  for (let r = 0; r < nr * ppc; r++) {
    const ci = Math.floor(r / ppc);
    for (let c = 0; c < nc * ppc; c++) {
      const idx = r * C + c;
      const gr = gRow[idx], gc = gCol[idx];
      const mag = Math.hypot(gr, gc);
      if (mag === 0) continue;
      // unsigned orientation in [0,180), matching np.rad2deg(arctan2) % 180
      let ori = Math.atan2(gr, gc) * 180 / Math.PI;
      ori = ((ori % 180) + 180) % 180;
      let k = Math.floor(ori / step);
      if (k >= orientations) k = orientations - 1;   // guard ori == 180-epsilon
      const cj = Math.floor(c / ppc);
      hist[(ci * nc + cj) * orientations + k] += mag / cellArea;
    }
  }

  const bR = nr - cpb + 1, bC = nc - cpb + 1;
  const perBlock = cpb * cpb * orientations;
  const out = new Float32Array(bR * bC * perBlock);
  const buf = new Float64Array(perBlock);
  let o = 0;

  for (let i = 0; i < bR; i++) {
    for (let j = 0; j < bC; j++) {
      let n = 0, sum = 0;
      for (let bi = 0; bi < cpb; bi++) {
        for (let bj = 0; bj < cpb; bj++) {
          const base = ((i + bi) * nc + (j + bj)) * orientations;
          for (let k = 0; k < orientations; k++) {
            const v = hist[base + k];
            buf[n++] = v;
            sum += v * v;
          }
        }
      }
      // L2-Hys: L2 normalise, clip at 0.2, renormalise
      let d = Math.sqrt(sum + eps * eps);
      sum = 0;
      for (let t = 0; t < perBlock; t++) {
        let v = buf[t] / d;
        if (v > 0.2) v = 0.2;
        buf[t] = v;
        sum += v * v;
      }
      d = Math.sqrt(sum + eps * eps);
      for (let t = 0; t < perBlock; t++) out[o++] = buf[t] / d;
    }
  }
  return out;
}
