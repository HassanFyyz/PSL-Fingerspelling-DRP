# Step 3, point 1: flagging bad letter-sheet crops (23 Nov 2026, corrected the same day)

> **Corrected.** Re-run after removing Kamran's idle-hand rows
> (`results/2026-11-23_kamran_idle_hand/`). The flags changed little (round B: 506 dropped
> instead of 513); the judge-reliability numbers changed a lot.

`flag_sheets.py` judges each of the 1,028 static-letter sheet crops (video holds). Thresholds
were fixed before any score was looked at:

- **clean**: an ExtraTrees model trained on clean data only does not have the crop's label in its top 3
- **neighbour**: the previous or next letter of the crop's word is in that model's top 3 with at
  least twice the label's probability (the timing slip, e.g. a م crop showing the ل before it)
- **other signers**: a model trained on the other four sheet signers does not have the label in
  its top 3. Reported only; it does not decide drops.

A crop is **dropped** if it fails *clean* or *neighbour*. The clean judge never uses the group its
round tests: round A's judge is trained on our calibration takes, round B's on Kamran's images.
Every crop's word position was verified against `word_letter_mapping.json` (1,345 of 1,345).

## Results

| | round A judge (our 291 takes) | round B judge (Kamran's 1,211 images) |
|---|---|---|
| crops dropped | 578 (56%) | 506 (49%) |
| of which the judge's top guess is the previous / next letter of the word | 32% | **44%** (223 crops) |
| same rate among crops kept | 4% | 4% |
| judge reliability: correctly labelled clean examples it would flag | 17.8% (on Kamran's images) | 8.6% (on our takes) |

- **The timing slip is real and common.** 223 crops, about one sheet crop in five, look most like
  the letter next to them in the word; among kept crops that happens 4% of the time.
- **Two independent judges agree** on 394 crops (38%): the Kamran judge and the other-signer
  judge share no training data.
- **Not one signer's fault.** Round B drop rates are 43–55% for every signer.
- **Letters.** Highest round-B drop rates: چ 100% (8 crops), ح 85%, ث 75% (4), and 60% for
  ظ خ غ ل پ. The sheet ح crops are visibly wrong (fists, open palms; the official ح is two
  fingers sideways, thumb tucked). Lowest: ذ ض 20%, ط 25%.
- Some flags are blurred or in-between frames rather than wrong labels; those are still bad
  training examples.
- Before the correction, round A's judge looked useless (it "flagged" 54% of correct labels).
  That came from the idle-hand rows in the test, not from the judge.

## Decision for point 3 (made 23 Nov, before the re-test)

In round A a crop is dropped only when **two judges agree** (round A's judge and the
other-signer check; neither uses Kamran's images). It was chosen while round A's judge looked
unreliable; it stands as decided, and the judge-alone rule is reported as well. Round B keeps
its own judge. Re-test: `results/2026-11-23_cleaned_test/`.

## Files

- `../../flag_sheets.py` — `PSL_KAMRAN_DIR=~/Desktop/train python flag_sheets.py` (12 s)
- `flags.csv` — one row per sheet crop: signer, video, word, position, neighbours, each judge's
  rank / top guess / flags (sheet crops are public data; nothing of Kamran's is written)
- `by_signer.csv`, `by_letter.csv`, `summary.csv`
