# Step 3, point 1: flagging bad letter-sheet crops (23 Nov 2026)

`flag_sheets.py` judges each of the 1,028 static-letter sheet crops (video holds). Thresholds
were fixed before any score was looked at:

- **clean**: an ExtraTrees model trained on clean data only does not have the crop's label in its top 3
- **neighbour**: the previous or next letter of the crop's word is in that model's top 3 with at
  least twice the label's probability (the timing slip, e.g. a م crop showing the ل before it)
- **other signers**: a model trained on the other four sheet signers does not have the label in
  its top 3. Reported only; it does not decide drops.

A crop is **dropped** if it fails *clean* or *neighbour*. To keep the clean test (`clean_eval.py`)
honest, the clean judge never uses the group its round tests: round A's judge is trained on our
calibration takes, round B's on Kamran's images. Every crop's word position was verified against
`word_letter_mapping.json` (1,345 of 1,345 match).

## Results

| | round A judge (our 291 takes) | round B judge (Kamran's 2,360 images) |
|---|---|---|
| crops dropped | 578 (56%) | 513 (50%) |
| of which the judge's top guess is the previous / next letter of the word | 32% | **44%** (224 crops) |
| same rate among crops kept | 4% | 4% |
| **judge reliability**: correctly labelled clean examples it would flag | **54%** (on Kamran's images) | **11%** (on our takes) |

- **The timing slip is real and common.** With the reliable judge (B), 224 crops, about one sheet
  crop in five, look most like the letter next to them in the word. Among kept crops that happens
  4% of the time.
- **Two independent judges agree.** 398 crops (39%) are flagged by both the Kamran judge and the
  other-signer judge, which share no training data. 90% of the other-signer flags are also B drops.
- **Round A's judge is too weak to clean with.** Trained on two people, it would flag 54% of
  correctly labelled images. Its drops are mostly noise. (This diagnostic was measured on
  Kamran's images, the group round A tests, so it is reported but must not tune anything.)
- **Not one signer's fault.** Drop rates are 44–56% for every signer (B); signer4-L is not the worst.
- **Letters.** Highest B drop rates: چ 100% (8 crops), ح 81%, ث 75% (4), ل 61%, ڈ 59%, ی 58%.
  For **ح** the Kamran judge rejects 81% but the other sheet signers only 42%: like ک (step 2),
  the sheets and Kamran may sign ح differently. Lowest: ذ ض 20%, ط 25%.
- Some flags are not wrong labels but blurred or in-between frames: the crop is still a bad
  training example, so dropping it is the intended effect.

## Decision needed before point 3 (the re-test)

The rule fixed in advance drops crops on the judge of each round. For round A that judge is
unreliable. Before any re-test score is computed, choose round A's cleaning:

1. **as fixed** (round A judge alone), or
2. **two judges must agree**: round A judge *and* the other-signer check (neither uses Kamran's images).

Point 3 will report both either way; the choice is which one counts.

## Files

- `../../flag_sheets.py` — `PSL_KAMRAN_DIR=~/Desktop/train python flag_sheets.py` (12 s)
- `flags.csv` — one row per sheet crop: signer, video, word, position, neighbours, each judge's
  rank / top guess / flags (sheet crops are public data; nothing of Kamran's is written)
- `by_signer.csv`, `by_letter.csv`, `summary.csv`
