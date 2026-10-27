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
