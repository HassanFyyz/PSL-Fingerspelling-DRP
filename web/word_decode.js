/**
 * Lexicon decoding, shared by both letter classifiers (HOG+LogReg and
 * hand-geometry+SVM). Operates purely on a per-hold probability-like
 * distribution over classes, so it does not care how that distribution was
 * produced -- softmax logits or vote fractions both work the same way here.
 */

export function top(classes, probs, k = 3) {
  return Array.from(probs)
    .map((p, i) => ({ letter: classes[i], p }))
    .sort((a, b) => b.p - a.p)
    .slice(0, k);
}

/** argmax letter sequence -- what the model literally predicted. */
export function rawWord(classes, probSeq) {
  return probSeq.map(p => classes[p.indexOf(Math.max(...p))]).join("");
}

/**
 * Best lexicon match, scored against the full probability distributions.
 *
 * Only words whose length equals the number of detected holds are considered
 * (within `slack`): segmentation is the reliable half of this pipeline, so
 * hold count is much better evidence than any individual letter.
 */
export function decodeWord(classIndex, lexicon, probSeq, { slack = 1, topN = 5 } = {}) {
  const n = probSeq.length;
  if (!n) return [];
  const scored = [];
  for (const [word, letters] of Object.entries(lexicon)) {
    if (Math.abs(letters.length - n) > slack) continue;
    const L = Math.min(letters.length, n);
    let logp = 0, ok = true;
    for (let i = 0; i < L; i++) {
      const ci = classIndex.get(letters[i]);
      if (ci === undefined) { ok = false; break; }
      logp += Math.log(Math.max(probSeq[i][ci], 1e-12));
    }
    if (!ok) continue;
    scored.push({ word, letters: letters.join(""),
                  score: logp / L - 0.7 * Math.abs(letters.length - n) });
  }
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, topN);
}

/* ---------------------------------------------------------------------------
 * Alignment decoder (port of decode_aligned in word_decode.py).
 *
 * decodeWord above compares hold i with letter i, so one missed or spurious
 * hold shifts every later letter. Here a hold may match no letter (cost `ins`)
 * and a letter may match no hold (cost `dele`), and the best monotonic
 * alignment is found by dynamic programming. No hold-count filter.
 *
 * Costs are the ones every leave-one-signer-out fold chose for vote share
 * (alignment_decoder.ipynb, section 5): unaided word top-1 51.9% against 23.6%
 * for decodeWord. The likelihood-ratio half of the Python decoder needs real
 * probabilities and is NOT applied here: vote share is not one.
 * ------------------------------------------------------------------------- */
export const ALIGN = { ins: 0.0, dele: -0.5, floor: 1e-6 };

/** A dictionary word is only SHOWN when it is supported by what was signed:
 *  at least MIN_HOLDS holds, and at least MIN_AGREE of the word's letters
 *  aligned to a hold that has that letter in its top 3. Measured on the 218
 *  dataset clips (unaided holds, leave-one-signer-out): a word is shown on 50%
 *  of clips and is right 79% of the time when shown; always showing the best
 *  word is right 51% of the time. The threshold was chosen on those same
 *  clips, so treat 79% as optimistic. */
export const MIN_HOLDS = 3;
export const MIN_AGREE = 0.6;

/** Best alignment of holds to one word. gain[i][c] = score of matching hold i
 *  to class c. Returns { score, pairs: [[holdIndex, letterIndex], ...] }. */
export function alignWord(gain, idx, { ins = ALIGN.ins, dele = ALIGN.dele } = {}) {
  const n = gain.length, m = idx.length, W = m + 1;
  const D = new Float64Array((n + 1) * W), B = new Uint8Array((n + 1) * W);
  for (let j = 1; j <= m; j++) { D[j] = D[j - 1] + dele; B[j] = 2; }
  for (let i = 1; i <= n; i++) {
    D[i * W] = D[(i - 1) * W] + ins; B[i * W] = 1;
    const g = gain[i - 1];
    for (let j = 1; j <= m; j++) {
      const match = D[(i - 1) * W + j - 1] + g[idx[j - 1]];
      const skipHold = D[(i - 1) * W + j] + ins;
      const skipLetter = D[i * W + j - 1] + dele;
      let best = match, b = 0;
      if (skipHold > best) { best = skipHold; b = 1; }
      if (skipLetter > best) { best = skipLetter; b = 2; }
      D[i * W + j] = best; B[i * W + j] = b;
    }
  }
  const pairs = [];
  let i = n, j = m;
  while (i > 0 || j > 0) {
    const b = B[i * W + j];
    if (b === 0) { pairs.push([i - 1, j - 1]); i--; j--; }
    else if (b === 1) i--;
    else j--;
  }
  pairs.reverse();
  return { score: D[n * W + m], pairs };
}

/**
 * Rank every lexicon word by its best alignment to the holds.
 * Each result: { word, letters, score, agree, nLetters, support, shown }
 *   agree   = aligned letters that are in their hold's top 3
 *   support = agree / nLetters
 *   shown   = whether the page should present this word (see MIN_AGREE)
 */
export function decodeAligned(classIndex, lexicon, probSeq, opts = {}) {
  const n = probSeq.length;
  if (!n) return [];
  const { ins = ALIGN.ins, dele = ALIGN.dele, floor = ALIGN.floor } = opts;
  const k = probSeq[0].length, logK = Math.log(k);
  const gain = [], rank = [];
  for (const p of probSeq) {
    let sum = 0;
    for (let c = 0; c < k; c++) sum += p[c];
    sum = Math.max(sum, 1e-12);
    const g = new Float64Array(k), r = new Int32Array(k);
    for (let c = 0; c < k; c++) {
      g[c] = Math.log(Math.max(p[c] / sum, floor)) + logK;   // uniform prior
      let above = 0;
      for (let d = 0; d < k; d++) if (p[d] > p[c]) above++;
      r[c] = above;
    }
    gain.push(g); rank.push(r);
  }
  const out = [];
  for (const [word, letters] of Object.entries(lexicon)) {
    const idx = letters.map(l => classIndex.get(l));
    if (idx.some(v => v === undefined)) continue;
    const { score, pairs } = alignWord(gain, idx, { ins, dele });
    let agree = 0;
    for (const [i, j] of pairs) if (rank[i][idx[j]] < 3) agree++;
    const support = agree / letters.length;
    out.push({ word, letters: letters.join(""), score, agree, nLetters: letters.length,
               support, shown: n >= MIN_HOLDS && support >= MIN_AGREE });
  }
  out.sort((a, b) => b.score - a.score);
  return out;
}
