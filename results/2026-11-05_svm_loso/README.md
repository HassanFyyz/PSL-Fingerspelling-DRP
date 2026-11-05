# Geometry + SVM: per-letter LOSO evaluation (2026-11-05)

## Protocol
- **Model:** the shipped recipe from `export_geometry_model.py`: 74 geometry features (`hand_geometry.py`), `StandardScaler`, `SVC(C=10, gamma="scale", class_weight="balanced")`.
- **Data:** `hold_landmarks.npy` / `hold_meta.csv`, restricted to the 5 dataset signers (1,345 holds, 40 letters). Hassan's 20 calibration holds (`user-hassan*`, letters ا and ک only) are excluded from both training and test.
- **Split:** leave-one-signer-out. Train on 4 signers, test on the 5th, repeat for all 5, and pool the predictions.
- **Labels are pseudo-labels** (word spelling plus segmentation), not hand-checked, and each hold is a single frame. Recall here means agreement with those labels, so part of every "error" is a wrong label. Even within one signer, a hold's nearest neighbour has the same label only 20% of the time.
- The SVM is deterministic, so a rerun gives the same numbers.

## Results
**Pooled accuracy: 26.4%** (mean of folds 26.5%). Always guessing the most common letter scores 10%. This matches the manifest's 27%, which also had Hassan's holds in training.

| Test signer | Holds | Letters present | Accuracy |
|---|---|---|---|
| signer1-F | 311 | 40 | 21.9% |
| signer2-F | 302 | 40 | 32.5% |
| signer3-F | 279 | 39 | 29.0% |
| signer4-L | 295 | 38 | 22.0% |
| signer5-R | 158 | 36 | 27.2% |

- **Per-letter recall, precision and top confusions:** `per_letter_loso_5signers.csv`.
- **Full 40×40 matrix:** `confusion_loso_5signers.csv` (rows = true letter, columns = predicted).
- **Weakest letters:**
  - Six letters score 0: ص, ڑ, خ, پ, ش, چ. Each has only 4–8 examples, too few to rank.
  - Among letters with at least 20 examples: ح .08, ئ .14, گ .14, ن .15, م .16, ٹ .20, ی .21, ر .23.

## Live comparison
Live: Abdullah, right hand, current SVM app, 10–13 short holds per letter.

| Letter | Live | Offline recall (n) | Offline main confusions | Weak offline too? |
|---|---|---|---|---|
| گ | 12/12 | .14 (21) | ا, ر, ن | Yes, yet perfect live |
| ا | 10/11 | .27 (130) | ل, ر, س | Moderate; most-predicted letter |
| و | ~11/12 | .36 (77) | آ, س, م | Mid |
| ل | 11/13 | .35 (75) | ا, ف, ی | Mid |
| س | 10/12 | .37 (52) | و, ک, ر | Mid |
| ک | 8/11 (→ ہ ×3) | .33 (93) | آ ×27, ی, س | Mid. ہ isn't an offline confusion; the live model also has Hassan's ک |
| ر | 8/12 (→ آ) | .23 (120) | آ ×13, ی, ا | Yes, same confusion |
| ث | ~6/12 (→ و ×4, ی, ھ) | .25 (4) | ک, م, و | Only 4 examples; 1 of 3 errors went to و |
| ت | ~5/11 (→ ش, ک) | .16 (19) | ل, ب, ا | Yes; ش and ک aren't the offline confusions |
| م | 0/10 (→ mostly س) | .16 (57) | ا, ل, ر | Yes, but not toward س |
| ح | 0/10 (→ mostly ر) | .08 (26) | ا, ب, ئ | Yes; ر is its nearest letter by geometry |

Live scores are much higher than offline. Live holds are deliberate, held still, and averaged over many frames; offline holds are single pseudo-labelled frames from fluent signing. Compare letters by rank, not by the raw numbers.

## م and ح (`closeness.py`)
**م: 57 examples from all 5 signers (14/13/12/11/7), but inconsistent.**
- Its examples are the 5th most spread out of 40 letters.
- Each signer's average م differs from another's by up to 14.5, against a median gap of 5.6 between different letters.
- Several of signer4-L's م frames have the index finger up (see `letter_sheets/U0645_م.png`).
- **Not close to س:** by median, س is the 19th–21st nearest of 39 letters.
- **The live م → س confusion is not explained by the training data.** Hypothesis, not tested: dataset م is too scattered to form a region the model can learn, and many dataset س examples are closed fists.

**ح: 26 examples (6/6/6/6/2), also inconsistent.**
- Its examples are the 7th most spread out.
- Its contact sheet mixes open palms with curled fists.
- **ر is its 1st–2nd nearest letter by median**, so the live ح → ر confusion has a real geometric basis.

## Parity fixture
`gen_parity_fixture.py` repeats the fixture-export cells of `streaming_segmentation.ipynb` without running the notebook. It used 218 of 353 clips as candidates and took every 9th clip at 50 and 30 fps, giving 50 cases.

With the new stillness and movement gates off (the default), `node web/test_parity.mjs <fixture>` passed: 50/50 exact match over 9,958 frames.

## Rerun
Run from the repo root. The parity step needs `hand_keypoints_features.csv` in the repo root (gitignored, symlinked locally).

```
python3 results/2026-11-05_svm_loso/loso_eval.py . results/2026-11-05_svm_loso
python3 results/2026-11-05_svm_loso/closeness.py .
python3 results/2026-11-05_svm_loso/gen_parity_fixture.py /tmp/sofi_fixture.json
node web/test_parity.mjs /tmp/sofi_fixture.json
```

The fixture is written outside the repo, and `.gitignore` ignores `*fixture*.json` anyway.
