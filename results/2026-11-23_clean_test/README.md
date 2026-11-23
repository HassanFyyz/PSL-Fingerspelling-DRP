# Step 2: clean test — do the letter sheets help? (23 Nov 2026)

**Why.** Every earlier new-person score used the letter-sheet (video-hold) labels as the
answer key, and some of those labels are wrong (e.g. م crops that show the neighbouring ل).
A model that learned the sheets' mistakes is rewarded by such a test, so it cannot tell
whether the sheets help. Here the answer key is only correctly labelled data.

**Data** (30 static letters only):

| group | what | rows | letters |
|---|---|---|---|
| K | Kamran's images, one letter per folder; signer(s) unknown, so one group | 2,360 | 30 |
| C | our calibration takes (Hassan, Abdullah; five sessions), one row per take | 291 | 30 |
| S | video holds behind the letter sheets, five dataset signers | 1,028 | 29 (no ء) |

**Design.** Two rounds; the test group is never in training.

| option | round A: test K | round B: test C |
|---|---|---|
| sheets only | train S | train S |
| clean only | train C | train K |
| clean + sheets (what we ship now) | train S + C | train S + K |

**Decision rule, fixed before the first run** (`clean_eval.py` docstring): shipped SVM recipe
primary; remove the sheets if *clean only* beats *clean + sheets* by ≥ 3 points in both
rounds, keep them if the reverse, otherwise undecided. ExtraTrees must agree or the result
is undecided.

## Results (accuracy %, macro recall in brackets)

| option | A, SVM | A, ExtraTrees | B, SVM | B, ExtraTrees |
|---|---|---|---|---|
| sheets only | 28.4 (29.9) | 32.3 (33.9) | 35.1 (35.7) | 45.0 (47.0) |
| clean only | 28.5 (29.9) | 30.9 (32.2) | **51.9** (53.6) | **68.7** (70.0) |
| clean + sheets | **33.7** (34.9) | **38.2** (39.7) | 50.9 (52.7) | 67.4 (68.7) |

Clean only minus clean + sheets: round A −5.2 (SVM) / −7.3 (ET) points; round B +1.0 / +1.4.
95% intervals over letters (`comparisons.csv`): A, ET −13.7 to −1.2; the other three cross 0.
Clean only minus sheets only, round B: +16.8 (SVM, 2.9 to 33.2) and +23.7 (ET, 10.0 to 36.3).

**Verdict by the rule: undecided** (`verdict.txt`). The rounds disagree.

## Reading

* **Clean data is worth more per row.** 1,028 sheet rows from five people score the same on
  Kamran's images as 291 calibration takes from two people (28.4 vs 28.5%). With 2,360 clean
  rows (round B) the clean model beats the sheets by 17–24 points.
* **The sheets help only when the clean data is small.** In round A the clean training set
  is two people, and adding the sheets' five people helps (+5 to +7). In round B the clean
  set is large and adding the sheets does nothing or slightly hurts (−1). The likely reason
  is people, not labels: the sheets add signer variety. Recording more people (friends)
  tests this directly: if it is right, the sheets stop helping round A too.
* **Letters.** The sheets help ا, و and ڈ in both rounds and hurt ص and ء in both (ء has no
  sheet examples, so the sheets only dilute it; ص is a candidate for wrong sheet labels).
* **ک is 0% across clean sources in both directions.** Our calibration ک and Kamran's ک
  appear to be different signs, or one source is mislabelled. Earlier, the official ک image
  was read as ڑ. Needs a check against the official sign before either is trusted.
* **ExtraTrees is well ahead of the SVM on clean answer keys** (+4.5 points in round A,
  +16.8 in round B). On the pseudo-labelled video test it was within 1 point (#24), so the
  earlier "keep the SVM" conclusion was drawn on the noisy test. Not acted on yet: the
  browser runs the SVM.
* Our calibration takes are easier to read than Kamran's images (round B scores are higher
  in every row): deliberate signing, and a single frame per take.

## Files

- `../../clean_eval.py`: the test. `PSL_KAMRAN_DIR=~/Desktop/train python clean_eval.py` (15 s).
- `scores.csv`, `comparisons.csv`, `per_letter_recall.csv`, `confusions.csv`, `verdict.txt`.
  Aggregates only: no row of Kamran's data is written here.
