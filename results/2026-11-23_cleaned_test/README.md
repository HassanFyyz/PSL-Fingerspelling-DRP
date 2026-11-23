# Step 3, point 3: the clean test with cleaned sheets (23 Nov 2026, final)

> **Data history** as in `results/2026-11-23_clean_test/`: this run uses Kamran's re-detected
> images (2,322 static). Earlier numbers are at the end.

Same groups, rounds and models as `results/2026-11-23_clean_test/`, with two new training options:
**clean + cleaned sheets** (the crops `flag_sheets.py` drops removed) and **clean + random-dropped**
(the same number of random crops removed, 10 draws): the control, so that "cleaning helps"
cannot just mean "fewer rows". Rules fixed before running (`cleaned_eval.py` docstring):

- Round A drops a crop only if its own judge *and* the other-signer check agree (414 crops).
  The judge-alone rule (578 crops) is reported too. Round B: its own judge (497 crops).
- Verdict: keep only cleaned sheets if they beat clean only by ≥ 3 points in both rounds; remove
  the sheets if clean only wins by ≥ 3 in both; otherwise undecided. SVM and ExtraTrees must agree.

## Results (accuracy %)

| option | A, SVM | A, ExtraTrees | B, SVM | B, ExtraTrees |
|---|---|---|---|---|
| clean only | 56.3 | 57.7 | **64.3** | **69.8** |
| clean + all sheets | 67.8 | **74.7** | 57.7 | 68.0 |
| clean + cleaned sheets | **70.8** | 71.5 | 63.2 | 65.3 |
| clean + random-dropped (mean of 10; min–max) | 64.1 (61.8–66.0) | 70.1 (66.7–71.9) | 59.2 (54.3–61.5) | 68.0 (66.0–69.4) |
| *A, judge alone: cleaned / random* | *71.1 / 62.9* | *69.6 / 68.2* | | |

**Verdict: undecided** for both models (`verdict.txt`).

## Reading

* **For the SVM (the model the browser runs), cleaning clearly works.** Cleaned sheets beat all
  sheets in both rounds (+3.0, +5.5) and beat every random-drop draw (70.8 vs at most 66.0;
  63.2 vs at most 61.5). The flags remove real errors, as the blind review confirmed (#30).
* **For ExtraTrees it does not:** cleaned sheets score below all sheets in both rounds and no
  better than dropping at random (inside the random range in A, below it in B). ExtraTrees copes with label noise and gains more from the extra
  rows and people than it loses to wrong labels.
* **Best training set per case** (accuracy, SVM / ExtraTrees):
  - only our two people as clean data (the public model's situation): SVM with **cleaned
    sheets** 70.8; ExtraTrees with **all sheets** 74.7;
  - Kamran's set as clean data (the private model's situation): **no sheets** for both, 64.3 / 69.8.

## Earlier runs, for the record (A SVM / A ET / B SVM / B ET)

| option | with idle hands | idle hands dropped |
|---|---|---|
| cleaned sheets | 33.8 / 35.9 / 56.7 / 70.1 | 63.5 / 67.2 / 58.1 / 69.4 |
| random-dropped | 31.6 / 36.0 / 49.6 / 67.0 | 59.2 / 67.7 / 55.3 / 67.4 |

## Files

- `../../cleaned_eval.py`: `PSL_KAMRAN_DIR=~/Desktop/train python cleaned_eval.py` (about 1.5 minutes)
- `scores.csv`, `verdict.txt`
