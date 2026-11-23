# Step 2: clean test — do the letter sheets help? (23 Nov 2026, corrected the same day)

> **Corrected.** The first version of this test included 1,149 of Kamran's static images whose
> landmarks are of the idle hand, not the signing hand (`results/2026-11-23_kamran_idle_hand/`).
> All numbers below are re-run without them; the first version's numbers are at the end.

**Why.** Every earlier new-person score used the letter-sheet (video-hold) labels as the
answer key, and some of those labels are wrong (e.g. م crops that show the neighbouring ل).
A model that learned the sheets' mistakes is rewarded by such a test, so it cannot tell
whether the sheets help. Here the answer key is only correctly labelled data.

**Data** (30 static letters only):

| group | what | rows | letters |
|---|---|---|---|
| K | Kamran's images, one letter per folder, several signers (not identified), idle-hand rows removed | 1,211 | 30 |
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
rounds, keep them if the reverse, otherwise undecided. ExtraTrees must agree.

## Results (accuracy %)

| option | A, SVM | A, ExtraTrees | B, SVM | B, ExtraTrees |
|---|---|---|---|---|
| sheets only | 53.4 | 59.4 | 35.0 | 45.0 |
| clean only | 53.7 | 58.2 | **57.7** | **70.8** |
| clean + sheets | **63.8** | **72.0** | 53.3 | 65.6 |

Clean only minus clean + sheets: round A −10.2 (SVM) / −13.8 (ET); round B **+4.5 / +5.2**.
Clean only minus sheets only, round B: +22.7 (SVM, letter interval 9.0 to 37.3), +25.8 (ET, 13.0 to 37.3).

**Verdict by the rule: undecided** — the rounds point in opposite directions, each by more than 3 points.

## Reading

* **When the clean data covers several people (Kamran's set), the sheets hurt.** Round B: adding
  them costs 4.5–5.2 points, and the sheets alone are 23–26 points worse than his images alone.
* **When the clean data is only our two people, the sheets help a lot.** Round A: +10 to +14
  points. They bring five more signers. 1,028 sheet rows alone score the same as our 291 takes
  alone (53.4 vs 53.7%).
* So the sheets' value is **people variety**, and clean data from enough people replaces it.
  For the public model (which cannot contain Kamran's images) the sheets are its only variety;
  for the private model they are a cost.
* **ExtraTrees is well ahead of the SVM on clean answer keys**: +8.2 (A, clean + sheets) and
  +13.1 (B, clean only). On the pseudo-labelled test it was within 1 point (#24). The browser runs the SVM.
* ک and ح are not label problems: see `results/2026-11-23_kamran_idle_hand/` and README #29.

## First version (with idle-hand rows), for the record

Accuracy, A SVM / A ET / B SVM / B ET: sheets only 28.4 / 32.3 / 35.1 / 45.0; clean only
28.5 / 30.9 / 51.9 / 68.7; clean + sheets 33.7 / 38.2 / 50.9 / 67.4. Also undecided.

## Files

- `../../clean_eval.py`: `PSL_KAMRAN_DIR=~/Desktop/train python clean_eval.py` (15 s).
- `scores.csv`, `comparisons.csv`, `per_letter_recall.csv`, `confusions.csv`, `verdict.txt`.
  Aggregates only: no row of Kamran's data is written here.
