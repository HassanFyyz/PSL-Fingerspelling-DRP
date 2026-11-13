# Live segmentation demo

Camera → MediaPipe hand keypoints → SOFI Algorithm 2 (causal) → letter boundaries, running
entirely in the browser.

## Two pages

- **`index.html`** — the live demo: camera → segmentation → letter → word.
- **`capture.html`** — calibration capture: record yourself signing letters, to train a model
  on your own hand. Opens at `http://localhost:8000/web/capture.html`.

## Run

```bash
python -m http.server 8000      # from the repo root
# open http://localhost:8000/web/
```

Opening `index.html` directly off the filesystem will **not** work — ES modules and
`getUserMedia` both need a real origin. `localhost` counts as a secure context, so no HTTPS
certificate is needed for local development.

## What it does

- Runs MediaPipe Hands in the tab. **No video leaves the client**; only the 21 keypoints per
  frame are processed, and only the cropped hold frames would ever be sent to a backend.
- Measures the actual camera framerate before building the segmenter, because the parameters
  are durations, not frame counts (see below).
- Draws the live OFI trace with confirmed holds marked.
- Captures the cropped hand at each detected hold, in signing order — exactly the input a
  letter classifier will consume.

## Recognition

The page predicts a **word**, end to end: camera → keypoints → holds → hand geometry (joint
angles + palm-normalised distances) → RBF-SVM (one-vs-one) → lexicon decode.

This replaced an earlier HOG-image + logistic-regression model that scored only 12.9 % LOSO; see
`hand_geometry.py` for why geometry features plus a kernel SVM do much better than an image crop
plus a linear model on this little data.

Measured leave-one-signer-out (40-letter folds only — the two calibration-session folds hold
only 2 letters each and are excluded as not comparable):

| | result | chance / baseline |
|---|---|---|
| per letter | 26.4 % (interval 22.6–30.4 %) | 10.0 % majority (40 classes) |
| word, unaided holds, old decoder | 23.6 % top-1 | 1.9 % (1 in 54) |
| word, unaided holds, alignment decoder (what the page runs) | 51.9 % top-1, 70.7 % top-3 | |

The letter figure excludes the calibration rows from training and test. The word figures are
measured with the segmenter finding the holds itself; see `end_to_end_words.ipynb` and
`alignment_decoder.ipynb`. With Platt probabilities and a likelihood ratio the same decoder
reaches 75 % offline, but the browser model exports vote share, not probabilities.

**What the page shows.** The top line is the letter read for each hold, unaltered. A
dictionary word is suggested only when at least 3 holds were signed and at least 60 % of the
word's letters match a hold's top 3; otherwise the page says there is no close match. On the
dataset clips a word is then shown about half the time and is right about 79 % of the time
when shown (threshold chosen on those clips, so optimistic). `node web/test_align.mjs` checks
the decoder's behaviour.

**Calibration.** The shipped model includes the developer's own recordings: ا (alif) and ک
(kaf) from two October sessions and ب (be) from 12 November. Live, right hand, those three
letters were read correctly on 61 of 62 holds across three tests and two rooms. Only ک
needed the calibration (0/5 without it); ا and ب are read correctly from the dataset alone.
A second, uncalibrated person got ک 0/3. So this is three letters for one person, not a
general accuracy claim — see `calibration/README.md`.

**If Start camera does nothing,** the browser is holding an older cached script. Press
Ctrl+Shift+R, or open the page in an Incognito window. The page now reports this itself
within five seconds.

The shipped model is trained on **everyone available** — all five dataset signers plus the
calibration sessions — so its training accuracy is meaningless. The honest number for a new
person at a webcam is the leave-one-signer-out one above.

### Files

| File | Purpose |
|---|---|
| `hand_geometry.js` | Joint angles + normalised distances, ported from `hand_geometry.py` |
| `svm_classifier.js` | Weight loading, libsvm one-vs-one decode, lexicon decoder (current model) |
| `word_decode.js` | Lexicon scoring shared by both classifiers |
| `model_geom/` | `geom_model.json` (manifest + lexicon) + SVM weight `.bin` files (~600 KB) |
| `test_svm.mjs` | JS vs Python parity on the OVO decode, checked against `SVC.predict()` |
| `hog.js`, `classifier.js`, `model/` | the earlier HOG + logistic-regression model, kept for reference, no longer loaded by `index.html` |
| `test_classifier.mjs` | JS vs Python parity for the HOG model |
| `test_decode.mjs` | End-to-end word decode, no browser needed |
| `capture.html` | Calibration capture; exports landmarks (+ legacy masked crops) as JSON for `calibrate.py` |

Regenerate the model with `python export_geometry_model.py`, then `python web/gen_svm_fixture.py
> web/svm_fixture.json` and `node web/test_svm.mjs`. The decode was also checked directly against
`sklearn.svm.SVC.predict()` in pure Python before any JS was written — see the top of
`export_geometry_model.py`. A silent numerical or indexing drift here would make every prediction
wrong with no error message, which is why it is checked rather than assumed.

The classifier works from MediaPipe landmarks, not an image crop, so there is no "model view" to
inspect — the colour hold thumbnails are a convenience display only.

## Files

| File | Purpose |
|---|---|
| `index.html` | the demo page |
| `sofi_streaming.js` | JS port of `sofi_streaming.py`, byte-identical output |
| `test_parity.mjs` | replays real clips through the JS port and diffs against Python |

## Three live-only behaviours that do not exist offline

Dataset clips are pre-trimmed to a signer actively fingerspelling. A camera also sees an empty
frame, a hand arriving, and a hand resting — and the unmodified algorithm reports holds for all
three. The first live test produced **47 holds in one session at 5 fps**. The fixes:

| Guard | What it stops |
|---|---|
| `MIN_VIABLE_FPS` + parameter floors | at 5 fps the durations round to `smooth=1, run=1`, so nearly every frame is a "minimum". Below the threshold the page reports nothing rather than nonsense |
| `GAP_TOLERANCE_SEC` | the motion spikes as a hand enters and leaves leave a trough between them that looks exactly like a hold. Only *sustained* absence is suppressed — short dropouts are detector blinks and are kept, because rejecting those costs real accuracy (MAE 1.23 → 1.45) |
| `MIN_CONTRAST` | an idle hand yields ~1.5 spurious holds/sec. A hold is a dip *between* transitions, so the surrounding peak must exceed it by a ratio. 1.10 keeps 98.5 % of real holds, rejects 93 % of idle ones |

The page also runs detection on a downscaled (320×240) copy of the frame and reports which
delegate (GPU/CPU) it got — feeding MediaPipe the full 960×720 capture every tick is what
produced 5 fps in the first place.

## Two things to know before changing anything

**The algorithm is causal but delayed.** A hold is confirmed `delay` frames after it happened
— 300 ms at 50 fps. That is the lookahead Algorithm 2 needs (centred smoothing, the ±`run`
minimum test, and `min_sep` before a deeper minimum can be ruled out), not lag in this page.
Removing the delay changes the algorithm.

**Parameters are durations, never frame counts.** The offline values (`smooth=13, run=7,
min_sep=15`) were tuned on 50 fps video. Reusing them as frame counts at webcam rates degrades
segment-count MAE from ~1.2 to 2.6 at 30 fps and 4.6 at 15 fps — worse than the Algorithm 1
baseline the whole project is trying to beat. `streaming_segmentation.ipynb` has the
measurement.

## Parity test

```bash
node web/test_parity.mjs <fixture.json>
```

The fixture is generated by the "Browser parity" cell in `streaming_segmentation.ipynb`, which
also runs this test automatically. Any change to `sofi_streaming.js` or `sofi_streaming.py`
should be followed by re-running that cell.
