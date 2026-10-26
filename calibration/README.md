# Calibration recordings

Sessions recorded with `web/capture.html`, to train a letter model on a specific
person's own hand rather than on the five dataset signers.

Why this exists: the dataset model scores 25.9% on an unseen signer in the same
studio, and worse again on a new person with a different camera in a different
room — which is what the live demo actually faces. Adding a handful of the
user's own examples closes most of that gap.

## Sessions

| file | date | letters | takes | frames |
|---|---|---|---|---|
| `calibration_2026-10-22.json` | 22 Oct 2026 | ا, ک | 10 (5 each) | 375 |

## Format

Each take holds one letter, recorded over a 2.5s countdown, with the hand
lowered and re-formed between takes. Per frame:

- `pts` — 21 MediaPipe landmarks as `[x, y, z]` in **isotropic video pixels**
  (`x*width, y*height, z*width`). Isotropic matters: the `hand_geometry.py`
  features are angles, and anisotropic coordinates skew them.
- `crop` — base64 of a 64×64 grayscale background-masked hand crop, for the
  first 20 frames of each take. Geometry comes from `pts`; the crops are only
  needed if comparing against the HOG image model.

## Results so far (`calibration_2026-10-22.json`)

Choosing among all 40 letters, chance 2.5%, takes held out whole:

- **70%** before calibration — dataset signers only
- **100%** after calibration — dataset plus the user's other takes of that letter

Three caveats recorded at the time:

1. **Not leakage.** Swapping the labels (every alif relabelled kaf and vice
   versa) made the model follow the swapped labels 10/10 and the true letter
   0/10, so the held-out take really was held out.
2. **But the 100% is flattered.** That same test showed 5 user takes override
   1,345 dataset examples. All ten takes came from one sitting, so a held-out
   take closely resembles the other four — partly answering "does this match
   the rest of this session?" rather than "is this alif?".
3. **The gain was mostly on kaf.** Retrained with zero user examples of the
   test letter: alif 5/5, kaf 2/5. Alif already worked from the dataset alone.

**Open test:** record 2–3 takes per letter in a *different session* — another
day, different light — and score them against this one. That removes the
same-session advantage and is what would make the result quotable.

## Usage

```bash
python calibrate.py calibration/calibration_2026-10-22.json
```
