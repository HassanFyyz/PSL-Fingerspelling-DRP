# Step 3, point 4: blind visual check of the sheet-crop flags (23 Nov 2026)

**Question.** Do the crops the round-B judge drops (`results/2026-11-23_sheet_flags/`) really
show the wrong letter? A person who knows the signs answers this without any model score.

**Design (blind).** 80 sheet crops: 40 the judge dropped and 40 it kept, shuffled. Each crop is
shown beside two letters, its label and the judge's best other letter, on random sides, each
with the official PSL image and one of our calibration takes. The reviewer picks left / right /
neither / can't tell. Nothing on the page says which crops were flagged.

**Which crops.** Only tiles of `letter_sheets/*/sheet.png` that could be tied to exactly one hold
(`cut_tiles.py` cuts the tiles and reads their captions by OCR; 117 of 325 static tiles matched
one hold, the rest share signer, letter and position with another word). 8 of 8 matches were
checked by eye against the caption. The sheets show each letter's most trusted holds, so this
sample is, if anything, easier on the labels than the full set.

**How to run it.**
1. Open `review_page.html` in a browser (it is git-ignored: it embeds the official screenshots;
   rebuild with `build_review.py` from the cut tiles and the official images).
2. Answer all 80 (about 10–15 minutes), press **Export answers**, save `flag_review_answers.csv` here.
3. `python results/2026-11-23_flag_review/score_review.py results/2026-11-23_flag_review/flag_review_answers.csv`

**Reading the result.** If the reviewer calls far more of the dropped crops wrong than the kept
ones, the flags pick out real errors, and cleaning the sheets with this judge is justified
independently of any test score.

## Result (reviewed 23 Nov, all 80 answered)

| crops | reviewer: label right | reviewer: label wrong | can't tell | label wrong % (95% interval) |
|---|---|---|---|---|
| **dropped by the judge** (40) | 6 | **34** (20 the judge's letter, 14 neither) | 0 | **85%** (71–93%) |
| kept by the judge (40) | 28 | 7 (1 the judge's letter, 6 neither) | 5 | 20% (10–36%) |

Fisher exact test, dropped vs kept: p = 1×10⁻⁸.

- **The flags are real.** A person who knows the signs, not told which crops were flagged, called
  85% of the dropped crops wrong. When the judge named a different letter, the reviewer agreed
  with it in half of the dropped crops (20 of 40).
- **The kept crops are not clean either:** about one in five is still wrong by the reviewer's
  eye, so cleaned sheets carry roughly 20% label noise.
- **Most sheet crops are wrong.** Weighting by how many crops the judge drops (49%) and keeps
  (51%) gives about 0.49 × 85% + 0.51 × 20% ≈ **52% of static sheet crops wrong or unusable**.
  This sample comes from the most trusted crops of each letter (the sheet tiles), so the full set
  is probably no better.
- This settles point 4: cleaning with the Kamran-trained judge removes mostly real errors,
  independently of any test score.

## Files

- `review_key.csv`: the 80 items (signer, video, position, label, alternative, side, flagged)
- `build_review.py`, `cut_tiles.py`, `score_review.py`
- `flag_review_answers.csv` (the reviewer's answers), `review_scored.csv`, `review_summary.csv`
