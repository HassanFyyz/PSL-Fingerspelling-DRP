# Static-letter model: word accuracy (22 Nov 2026)

**Question.** The scope is the 30 static letters (decided 15 Nov). The shipped browser
model still had all 41 letters as classes, and words had not been re-tested. Option A,
chosen 22 Nov: train on the 30 static letters, keep all 54 dictionary words, and let a
moving letter in a word be a **slot** the decoder fills with any hold (score 0, no
evidence either way) or skips. Before this change the decoder dropped any word with a
letter the model has no class for, which would have removed 44 of the 54 words.

**Test.** The 218 dataset clips, holds found by Algorithm 2 without being told the word,
each signer held out in turn (four train, one test). Word labels come from the video
names, not from the pseudo-labels, so **this test does not depend on the letter-sheet
labels**. Same protocol and seed as `model_alternatives.ipynb` section 4 and
`alignment_decoder.ipynb`; the 40-letter rows reproduce those numbers exactly
(74.8 / 88.2 and 51.9 / 70.7).

- **Offline:** Platt probabilities, likelihood ratio, (ins, dele) tuned on the training signers.
- **Browser:** vote share, ins 0, dele −0.5, no prior: the page's settings, decoded by
  `web/word_decode.js` itself (`decode_clips.mjs`). "Shown" = the page's support test
  (≥ 3 holds, ≥ 60% of the readable letters in their hold's top 3).

## Results (top-1 %, mean of five signers, 95% signer-bootstrap interval)

| | offline top-1 | offline top-3 | browser top-1 | browser top-3 | browser: shown / right when shown |
|---|---|---|---|---|---|
| **40 letters** (before) | **74.8** (55.9–85.9) | 88.2 | **51.9** (37.8–61.3) | 70.7 | 50.5% / 79.1% |
| **30 static + slots** (option A, shipped) | 68.3 (54.1–78.5) | 84.2 | 40.7 (30.7–48.7) | 59.9 | 55.5% / 61.2% |
| 30 static + one pooled "moving letter" class | 72.0 (53.9–83.4) | 89.6 | 40.0 (32.3–48.2) | 63.3 | 66.5% / 53.1% |

Split by word (171 clips of the 44 words with a moving letter, 47 clips of the 10 fully static words):

| | offline: moving-letter words | offline: static words | browser: moving-letter words | browser: static words |
|---|---|---|---|---|
| 40 letters | 73.2 | 80.0 | 52.1 | 51.4 |
| 30 static + slots | 62.7 | **88.0** | 35.2 | **61.4** |
| 30 static + pooled class | 67.4 | **88.0** | 41.5 | 34.6 |

Offline, slots vs 40 letters: −6.5 points, paired p = 0.14; pooled vs 40 letters: −2.9, p = 0.08.
Five signers, so neither difference is significant, but the browser drop (−11 points top-1,
and a shown word right 61% instead of 79% of the time) is large.

## Reading

* **The static-letter model reads static letters better and words worse.** The 10 fully
  static words gain (offline 80 → 88, browser 51 → 61). The 44 words with a moving letter
  lose more than that, because the 40-letter model was using the moving letters' held hand
  shape as evidence even though it cannot follow the movement. A slot throws that evidence
  away.
* **Pooling the 11 moving letters into one class recovers most of the loss offline**
  (72.0 vs 74.8) without the model ever naming a moving letter. In the browser it does not
  help: the vote share gives the large pooled class too much weight, and shown words are
  right only 53% of the time. Not shipped.
* **The browser decoder is the bigger gap.** For every model the offline decoder (real
  probabilities + likelihood ratio) is 20–30 points above the browser's vote share. Getting
  probabilities into the browser (README next actions) is worth more than any change here.
* The shown-word threshold was set on these same clips for the 40-letter model. Requiring
  2 or 3 agreeing letters changes little (see `browser_summary.csv`).

## Files

- `run.py`: 40 letters vs 30 static + slots, offline and browser. `python results/2026-11-22_static_words/run.py` (about a minute).
- `pooled.py`: the pooled-class variant (reads `offline_per_clip.csv` from `run.py`).
- `decode_clips.mjs`: runs `web/word_decode.js` on dumped clip scores.
- `offline_per_clip.csv`, `browser_per_clip.csv`, `*_pooled.csv`: one row per clip.
- `offline_summary.csv`, `browser_summary.csv`, `browser_summary_pooled.csv`: the tables above.
