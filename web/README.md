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

The page now predicts a **word**, end to end: camera → keypoints → holds → hand crop →
HOG → logistic regression → lexicon decode.

Measured leave-one-signer-out:

| | result | chance / baseline |
|---|---|---|
| per letter | 13.0 % | 9.8 % majority (40 classes) |
| **word top-1** | **33.1 %** | ~1.9 % (54-word lexicon) |
| word top-3 | 55.0 % | ~5.6 % |

Per-signer word top-1: signer1-F 46.9 %, signer2-F 34.7 %, signer3-F 46.7 %, signer4-L 14.3 %,
signer5-R 23.1 % — front-facing signers do roughly twice as well as tilted ones, exactly as
`angle_evaluation.ipynb` predicts.

**Individual letters are usually wrong, and the page says so.** The word guess survives that
because it scores the *whole* sequence against the vocabulary using the full per-letter
probability distributions rather than the argmax letters, so runner-up guesses still contribute.
Candidate words are filtered to those matching the detected hold count (±1), which is a real
constraint rather than a trick: segmentation is the reliable half of this pipeline (MAE ~1.2),
so hold count is far better evidence than any single letter.

The shipped model is trained on **all five signers** — for a demo you want every scrap of data —
so its training accuracy is meaningless. The honest number for a new person at a webcam is the
leave-one-signer-out one above.

### Files

| File | Purpose |
|---|---|
| `hog.js` | HOG, ported from `export_model.ref_hog`, itself verified against skimage to ~1e-7 |
| `classifier.js` | Weight loading, softmax, lexicon decoder |
| `model/` | `letter_model.json` (manifest + lexicon), `letter_w.bin`, `letter_b.bin` (276 KB) |
| `test_classifier.mjs` | JS vs Python parity on real crops |
| `test_decode.mjs` | End-to-end word decode, no browser needed |
| `capture.html` | Calibration capture; exports landmarks + masked crops as JSON for `calibrate.py` |

Regenerate the model with `python export_model.py`, then re-run both tests. A silent numerical
drift between `hog.js` and the Python HOG would make every prediction wrong with no error
message, which is why it is checked rather than assumed.

## Model view and crop export

Two controls exist to support the classifier work:

- **Show model view** toggles the hold thumbnails between the colour camera crop and the
  **64×64 grayscale square crop a model actually receives**. Worth checking: if the model view
  is off-centre or cut off, no classifier will work however good it is.
- **Save crops** downloads those crops as PNGs, in exactly the format `crop_dataset.py`
  produces. This is how you would seed a per-user calibration set.

The crop geometry here is kept **identical to `crop_dataset.py`** — square box (so resizing
never distorts the hand's aspect ratio), `BOX_PAD = 0.30`, grayscale via BT.601 luma to match
`cv2.COLOR_BGR2GRAY`. If those two drift apart, a model trained on the dataset sees a different
distribution at inference than it did in training, which is the kind of bug that shows up only
as unexplained poor accuracy.

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
