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
| `calibration_2026-10-26.json` | 26 Oct 2026 | ا, ک | 10 (5 each) | 371 |

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

**Open test — now answered.** A second session, `calibration_2026-10-26.json`,
recorded 4 days later, was scored against a model trained on the dataset plus
only the 22 Oct session — the 26 Oct frames never entered training:

- **50%** no calibration — dataset signers only (alif 5/5, kaf 0/5; it defaults
  to alif every time)
- **100%** cross-session — dataset + 22 Oct, scored on the unseen 26 Oct session
- **100%** same-session upper bound — dataset + 22 Oct + 26 Oct's own sibling
  takes

Cross-session matches the same-session ceiling, so the 22 Oct calibration
generalises to a different day/light/seat rather than memorising one sitting.
Run via `python cross_session_test.py calibration/calibration_2026-10-26.json`.

## Adversarial check: is it reading hand shape or cheating?

To test this directly, a third recording deliberately swapped the labels:
takes labeled ا were actually signed as ک and vice versa (confirmed frame by
frame -- every frame of every "mislabeled" take measured as the swapped
shape, not a partial or noisy transition). Scored against the model trained
on the dataset + 22 Oct (which never saw this file):

- Against the **typed label**: 0/10 correct.
- Against the **true shape actually made**: **10/10 correct.**

The base model ignored the wrong label and correctly read the real hand
shape every time -- it is reading geometry, not some session-specific
artifact. Retraining on this file's own (mislabeled) other takes flips that
exactly: 10/10 against the wrong label, 0/10 against the true shape, because
that is what supervised learning does when fed wrong labels. This is not
leakage, it is calibration having no defense against a bad take -- which is
why `add_calibration.py` now checks each take against the current model
before merging it (see below).

## In the training data

The 22 Oct session has been merged into `hold_landmarks.npy` / `hold_meta.csv`
as signer **`user-hassan`**, via:

```bash
python add_calibration.py calibration/calibration_2026-10-22.json --signer user-hassan
```

**One row per take, not one per frame.** Ten takes became ten rows, each the
median frame of its take. The 375 recorded frames are near-duplicates within
their takes, so counting them individually would treat 10 real examples as 375
and make one person 22% of the training data for two letters. At one row per
take it is 0.7%, which is honest.

If the user's examples should outweigh the dataset — reasonable for personal
calibration — use sample weights at training time rather than duplicating rows.
Padding the row count inflates every accuracy computed from this file.

Effect on the headline number: leave-one-signer-out over the five dataset
signers went 26.5% → 26.9%, i.e. unchanged within noise, which is what 10 extra
rows should do.

**Note for evaluation:** LOSO now has a sixth fold containing only 2 letters.
It is not comparable to the 40-letter folds and should be filtered out when
quoting overall accuracy.

## Mislabel check before merging

`add_calibration.py` now scores every take against a model trained on the
current dataset (never including this session) before merging it. If the
model disagrees with a take's label, that take is flagged and the merge is
refused by default:

```bash
python add_calibration.py calibration/some_session.json --signer whoever
# ... if a take is flagged:
python add_calibration.py calibration/some_session.json --signer whoever --drop-suspicious  # merge the rest
python add_calibration.py calibration/some_session.json --signer whoever --force            # merge anyway
python add_calibration.py calibration/some_session.json --signer whoever --no-check          # skip the check
```

This exists because a deliberately mislabeled take (see the adversarial check
above) would otherwise have been merged silently and taught the model the
wrong thing for that letter.

## Usage

```bash
python calibrate.py calibration/calibration_2026-10-22.json
python cross_session_test.py calibration/calibration_2026-10-26.json
```
