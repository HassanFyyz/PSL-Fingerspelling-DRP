# Step 3, point 3: the clean test with cleaned sheets (23 Nov 2026, corrected the same day)

> **Corrected.** Re-run after removing Kamran's idle-hand rows
> (`results/2026-11-23_kamran_idle_hand/`). The first version's numbers are at the end.

Same groups, rounds and models as `results/2026-11-23_clean_test/`, with two new training options:
**clean + cleaned sheets** (the crops `flag_sheets.py` drops removed) and **clean + random-dropped**
(the same number of random crops removed, 10 draws): the control, so that "cleaning helps"
cannot just mean "fewer rows". Rules fixed before running (`cleaned_eval.py` docstring):

- Round A drops a crop only if its own judge *and* the other-signer check agree (414 crops).
  The judge-alone rule (578 crops) is reported too. Round B: its own judge (506 crops).
- Verdict: keep only cleaned sheets if they beat clean only by ≥ 3 points in both rounds; remove
  the sheets if clean only wins by ≥ 3 in both; otherwise undecided. SVM and ExtraTrees must agree.

## Results (accuracy %)

| option | A, SVM | A, ExtraTrees | B, SVM | B, ExtraTrees |
|---|---|---|---|---|
| clean only | 53.7 | 58.2 | 57.7 | **70.8** |
| clean + all sheets | **63.8** | **72.0** | 53.3 | 65.6 |
| clean + cleaned sheets | 63.5 | 67.2 | **58.1** | 69.4 |
| clean + random-dropped (mean of 10; min–max) | 59.2 (56.5–61.1) | 67.7 (64.5–69.0) | 55.3 (52.6–58.1) | 67.4 (65.6–70.5) |
| *A, judge alone: cleaned / random* | *64.7 / 58.3* | *64.7 / 66.1* | | |

**Verdict: undecided** for both models (`verdict.txt`).

## Reading

* **Cleaning removes most of the sheets' harm, but does not make them useful once Kamran's
  data is in.** Round B: all sheets cost 4.5 / 5.2 points against clean only; cleaned sheets
  are level with clean only (+0.4 SVM, −1.4 ET) and better than all sheets (+4.8 / +3.8).
* **Cleaning beats random dropping for the SVM in both rounds, not for ExtraTrees**, whose
  cleaned scores sit inside the random draws' range. ExtraTrees copes with label noise well
  enough that removing the flagged half costs it about as much as removing any half.
* **When the clean data is only our two people, the sheets help even uncleaned** (round A,
  ExtraTrees: 72.0 with all sheets, 67.2 cleaned). There, variety of people outweighs label noise.
* Practical reading (not the rule's verdict): with several clean signers, train without the
  sheets or with cleaned ones; with only our two people, keep them.

## First version (with idle-hand rows), for the record

A SVM / A ET / B SVM / B ET: clean only 28.5 / 30.9 / 51.9 / 68.7; all sheets 33.7 / 38.2 /
50.9 / 67.4; cleaned 33.8 / 35.9 / 56.7 / 70.1; random 31.6 / 36.0 / 49.6 / 67.0. Undecided.

## Files

- `../../cleaned_eval.py`: `PSL_KAMRAN_DIR=~/Desktop/train python cleaned_eval.py` (about 1.5 minutes)
- `scores.csv`, `verdict.txt`
