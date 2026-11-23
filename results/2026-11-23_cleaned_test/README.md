# Step 3, point 3: the clean test with cleaned sheets (23 Nov 2026)

Same groups, rounds and models as `results/2026-11-23_clean_test/`, with two new training options:
**clean + cleaned sheets** (the crops `flag_sheets.py` drops removed) and **clean + random-dropped**
(the same number of random crops removed, 10 draws): the control, so that "cleaning helps"
cannot just mean "fewer rows". Rules fixed before running (`cleaned_eval.py` docstring):

- Round A drops a crop only if its own judge *and* the other-signer check agree (414 crops),
  because round A's judge alone would flag 54% of correctly labelled images (decided 23 Nov).
  The first-fixed rule (judge alone, 578 crops) is reported too. Round B: its own judge (513 crops).
- Verdict: keep only cleaned sheets if they beat clean only by ≥ 3 points in both rounds; remove
  the sheets if clean only wins by ≥ 3 in both; otherwise undecided. SVM and ExtraTrees must agree.

## Results (accuracy %)

| option | A, SVM | A, ExtraTrees | B, SVM | B, ExtraTrees |
|---|---|---|---|---|
| clean only | 28.5 | 30.9 | 51.9 | 68.7 |
| clean + all sheets | 33.7 | **38.2** | 50.9 | 67.4 |
| **clean + cleaned sheets** | **33.8** | 35.9 | **56.7** | **70.1** |
| clean + random-dropped (mean of 10; min–max) | 31.6 (30.0–32.7) | 36.0 (34.9–37.1) | 49.6 (46.7–53.3) | 67.0 (64.6–68.7) |
| *A as first fixed: cleaned / random* | *34.5 / 31.1* | *34.9 / 35.0* | | |

**Verdict: undecided.** SVM: keep only the cleaned sheets (+5.3 and +4.8 points over clean only).
ExtraTrees: undecided (+5.0 and +1.4). The models disagree, so the rule says undecided.

## Reading

* **With the reliable judge (round B), cleaning finds real errors.** Cleaned sheets beat all
  sheets (SVM +5.8, ET +2.7) and beat dropping the same number at random by more than the random
  draws' whole range (SVM 56.7 vs at most 53.3; ET 70.1 vs at most 68.7). Removing the flagged
  half of the sheets turns them from no help into a help.
* **With the weak judge (round A), cleaning is no better than random for ExtraTrees** (35.9 vs
  36.0) and only slightly better for the SVM. A judge trained on two people cannot tell a wrong
  label from an unfamiliar signer.
* So the quality of the cleaning depends on the judge. The good judge is the one trained on
  Kamran's images; round A cannot use it, because round A tests on his images. That is a limit of
  this test, not of the cleaning.
* ExtraTrees copes with noisy labels better than the SVM: it gains less from cleaning and loses
  less from the noise (round A, all sheets, is its best score).

## What this suggests for the shipped model (not yet done)

Clean the sheets with a judge trained on Kamran's images and our calibration takes, then train on
clean data + cleaned sheets. Round B supports this; nothing contradicts it, but by the rule it is
not yet decided. The visual check of the flags (point 4) and new people (step 4) can settle it.

## Files

- `../../cleaned_eval.py`: `PSL_KAMRAN_DIR=~/Desktop/train python cleaned_eval.py` (about 1.5 minutes)
- `scores.csv`, `verdict.txt`
