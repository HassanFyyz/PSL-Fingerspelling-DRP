# Step 2: clean test — do the letter sheets help? (23 Nov 2026, final)

> **Data history.** First run: Kamran's landmarks included 1,348 idle hands. Second run: those
> rows dropped (1,211 static images). **This run: the idle-hand images re-detected with two hands
> (`python image_dataset.py`), 2,322 static images.** See `results/2026-11-23_kamran_idle_hand/`.
> Earlier numbers are at the end.

**Why.** Every earlier new-person score used the letter-sheet (video-hold) labels as the
answer key, and some of those labels are wrong (e.g. م crops that show the neighbouring ل).
A model that learned the sheets' mistakes is rewarded by such a test, so it cannot tell
whether the sheets help. Here the answer key is only correctly labelled data.

**Data** (30 static letters only):

| group | what | rows | letters |
|---|---|---|---|
| K | Kamran's images, one letter per folder, several signers (not identified) | 2,322 | 30 (54–119 each) |
| C | our calibration takes (Hassan, Abdullah; five sessions), one row per take | 291 | 30 |
| S | video holds behind the letter sheets, five dataset signers | 1,028 | 29 (no ء) |

**Design.** Each clean group is the test group in turn and never in training (new people
recorded later add rounds automatically, `clean_eval.rounds`).

| option | round A: test K | round B: test C |
|---|---|---|
| sheets only | train S | train S |
| clean only | train C | train K |
| clean + sheets (what we ship now) | train S + C | train S + K |

**Decision rule, fixed before the first run** (`clean_eval.py` docstring): shipped SVM recipe
primary; remove the sheets if *clean only* beats *clean + sheets* by ≥ 3 points in every
round, keep them if the reverse, otherwise undecided. ExtraTrees must agree.

## Results (accuracy %)

| option | A, SVM | A, ExtraTrees | B, SVM | B, ExtraTrees |
|---|---|---|---|---|
| sheets only | 52.6 | 60.9 | 35.0 | 45.0 |
| clean only | 56.3 | 57.7 | **64.3** | **69.8** |
| clean + sheets | **67.8** | **74.7** | 57.7 | 68.0 |

Clean only minus clean + sheets: round A −11.5 (SVM) / −17.0 (ET); round B +6.5 / +1.7.
Clean only minus sheets only, round B: +29.2 (SVM, letter interval 17.0 to 43.0), +24.7 (ET, 11.7 to 36.7).

**Verdict by the rule: undecided** — round A favours the sheets, round B (for the SVM) clean data alone.

## Reading

* **When the clean data covers several people (Kamran's set), the sheets do not help:** adding
  them costs the SVM 6.5 points and ExtraTrees 1.7; the sheets alone are 25–29 points worse
  than his images alone.
* **When the clean data is only our two people, the sheets help a lot:** +11.5 to +17 points.
  They bring five more signers. So their value is **people variety**, and clean data from
  enough people replaces it. For the public model (no Kamran) the sheets are its only variety.
* **More clean data helps:** doubling Kamran's usable images (1,211 → 2,322) raised the SVM
  trained on them from 57.7% to 64.3% on our takes.
* **ExtraTrees is well ahead of the SVM on clean answer keys**: +6.9 (A, clean + sheets),
  +5.5 (B, clean only). On the pseudo-labelled test it was within 1 point (#24).

## Earlier runs, for the record (A SVM / A ET / B SVM / B ET)

| option | with idle hands | idle hands dropped | **recovered (this run)** |
|---|---|---|---|
| sheets only | 28.4 / 32.3 / 35.1 / 45.0 | 53.4 / 59.4 / 35.0 / 45.0 | 52.6 / 60.9 / 35.0 / 45.0 |
| clean only | 28.5 / 30.9 / 51.9 / 68.7 | 53.7 / 58.2 / 57.7 / 70.8 | 56.3 / 57.7 / 64.3 / 69.8 |
| clean + sheets | 33.7 / 38.2 / 50.9 / 67.4 | 63.8 / 72.0 / 53.3 / 65.6 | 67.8 / 74.7 / 57.7 / 68.0 |

## Files

- `../../clean_eval.py`: `PSL_KAMRAN_DIR=~/Desktop/train python clean_eval.py` (15 s).
- `scores.csv`, `comparisons.csv`, `per_letter_recall.csv`, `confusions.csv`, `verdict.txt`.
  Aggregates only: no row of Kamran's data is written here.
