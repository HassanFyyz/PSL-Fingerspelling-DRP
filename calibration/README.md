# Calibration recordings

## 13 November mixed Abdullah/Hassan session

`calibration_2026-11-13.json` contains 261 takes across 26 letters from Abdullah
and Hassan. Take-level signer identity was not recorded, so it is merged under
the source-group ID `user-mixed-abdullah-hassan-2026-11-13`; that ID does not
represent a separate person. Each take adds one median-frame landmark row.
All 261 supplied takes were kept. The raw file's ten MediaPipe Left tags
(all for ق) are preserved; the participants report right-handed signing.

Standalone hamza ء was deliberately included as a new class, expanding the
classifier to 41 letters. Training now has 1,636 rows, including the existing
1,375. Photos decoded from embedded early-frame crops are for inspection;
the SVM learns angles and distances from landmarks.

Same-session whole-take five-fold validation: 83/261 before, 241/261 (92.3%)
after. Restricting to the 251 takes from existing classes: 33.1% → 92.8%.
Original-dataset signer LOSO with calibration added only to training:
26.8% before these takes → 28.0% after. The dataset-only baseline remains
26.4%. These are different protocols; neither proves new-person accuracy.
No counted live result after this update has been recorded yet. Abdullah
reports strong recognition in a later setting with changed lighting and the
same camera; the number of attempts and errors was not recorded.

See `results/2026-11-13_calibration_audit/README.md` for the full evaluation,
snapshots, pictures and before/after live servers. The new session is now
merged: do not score it with `score_session.py`'s already-merged baseline and
call that an unseen-session result. Use the frozen experiment inputs instead.

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

Both sessions have been merged into `hold_landmarks.npy` / `hold_meta.csv`:

| session | signer id | rows |
|---|---|---|
| 22 Oct | `user-hassan` | 10 |
| 26 Oct | `user-hassan-session2` | 10 |
| 12 Nov (file `calibration_2026-11-11.json`, dated in UTC) | `user-hassan-session3` | 5 (be only; this session's alif and kaf takes were recorded but left out, since both letters already have two sessions) |
| 12 Nov evening (`calibration_2026-11-12.json`) | `user-hassan-session4` | 5 (lam only; the session's alif takes were recorded but left out) |

```bash
python add_calibration.py calibration/calibration_2026-10-22.json --signer user-hassan
python add_calibration.py calibration/calibration_2026-10-26.json --signer user-hassan-session2
python add_calibration.py calibration/calibration_2026-11-11.json --signer user-hassan-session3 --force
# then the session-3 alif and kaf rows were removed from hold_meta.csv / hold_landmarks.npy
```

The third session needed `--force`: the mislabel check flagged one kaf take
(model said re 14%, kaf 14%, a near tie). Its crops show the same two-finger
shape as the other four kaf takes, so the label was kept. Scored before merging
(`python score_session.py ...`): be 5/5 with no calibration at all, alif 5/5,
kaf 0/5 with no calibration and 4/5 with only the two October sessions.

Because the 26 Oct session is now in the training file, `cross_session_test.py`
leaves every `user-*` row out of both models except the one earlier session
named by `--calibrated-on` (default `user-hassan`). Before that fix (10 Nov) the
26 Oct takes sat inside the "no calibration" model and the script printed
100/100/100. `calibrate.py` likewise trains its "before" and "after" models on
the dataset signers only.

**One row per take, not one per frame.** Ten takes became ten rows per session,
each the median frame of its take. The 375 recorded frames of a session are
near-duplicates within their takes, so counting them individually would treat 10
real examples as 375 and make one person 22% of the training data for two
letters. At one row per take the two sessions together are 1.5%.

If the user's examples should outweigh the dataset — reasonable for personal
calibration — use sample weights at training time rather than duplicating rows.
Padding the row count inflates every accuracy computed from this file.

Effect on the headline number: leave-one-signer-out over the five dataset
signers went 26.5% → 26.9%, i.e. unchanged within noise, which is what 10 extra
rows should do.

**Note for evaluation:** the file now has two extra signers holding only 2
letters each. They are not comparable to the 40-letter folds.
`evaluation.load_dataset()` drops `user-*` rows by default, and the headline
numbers are computed that way.

**How far the 100% goes.** 10/10 has a 95% interval of 69-100%, the takes within
a session are not independent, and the whole gain is ک going from 0/5 to 5/5.
Treat it as a two-letter pilot.

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
