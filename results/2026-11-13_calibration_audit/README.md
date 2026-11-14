# Calibration recorded on 13 November 2026

The new recordings improve recognition on other takes from the same sitting
much more than they improve recognition on the five original dataset signers.
These are different tests; the 92.3% result is not a new universal accuracy.

## Recording and training data

Source: `calibration/calibration_2026-11-13.json`, copied without modification
from Abdullah's Desktop. SHA-256:
`758bdf5612da8b04e83b1af88f0eca56b7881256f9e5cea522c173f85b07cbe5`.

There are 261 takes, 34,029 frames, 26 labelled letters and 5,220 embedded
64x64 crops. There are ten takes of each letter except پ, which has eleven.
The capture's timestamp is 11:02 UTC (16:02 in Pakistan).

Abdullah confirmed that he and Hassan recorded the takes, usually five each,
but did not record who made which take. No signer identities were inferred
from the images or take order. The training metadata uses
`user-mixed-abdullah-hassan-2026-11-13` as a **source group**, not a distinct
person. It must not be used as a new-person LOSO fold.

Both participants report using their right hand. The JSON contains 251 Right
and ten Left take labels from MediaPipe; all ten Left tags are for ق.
Those tags are preserved in the capture and audit. Mirror-invariant features
do not require rejecting these takes. The tag is not ground truth about the
physical hand used.

At Abdullah's request, standalone hamza ء is added as a 41st class. It was
not in the earlier 40 classes, and the 54-word lexicon is unchanged. No label
mapping or merging with ئ/ؤ/آ was performed.

All 261 supplied takes are included, with one float32 median-frame landmark
sample per take, following `add_calibration.pick_frames(..., n=1)`. No label
was removed because the old model disagreed with it. Agreement with a weak
model is not a reliable label-quality filter. The supplied labels have not
been independently verified against an authoritative PSL annotation.

The combined data has 1,636 rows: 1,345 original pseudo-labelled holds, the
30 prior Hassan calibration rows, and these 261 takes. Every prior row is
unchanged. م and ن are absent from this recording; no new examples for them
were added.

## Pictures

`crops/Uxxxx_<letter>/take_NNN/frame_NNNN.png` contains every embedded crop.
`contact_sheets/` has one image per letter with one crop per take. The three
`overview_*.png` files show all letters compactly.

The capture saves crops only for the first 20 frames. The contact sheets show
the middle of those saved crops (usually frame 10); training uses the middle
landmark frame of the full take (roughly frame 45-75). No crop exists for the
actual median frame in any of these 261 takes. These pictures therefore help
inspect the recorded signs but are not pictures of the exact training frame.
The geometry SVM is trained on landmark features, not these PNGs.

## Same-session calibration test

Settings are fixed: 74 `hand_features` values, training-only StandardScaler,
RBF SVC, C=10, gamma=scale, balanced class weights, no Platt probabilities.
No new hyperparameters were selected.

Before: the exact frozen JavaScript SVM and exported weights score all take
frames. This was checked directly through `SvmClassifier.predict`, not merely
by fitting another Python model. A Python refit has slightly different
coefficients on this machine but gives the same 83 correct takes.

After: five folds made by round-robin take order within each letter. Entire
takes are held out. For each fold, the model trains on the 1,375 prior rows
plus one median frame from each remaining take. Prediction averages OVO vote
shares over **all** frames of a held-out take, following the browser's rule.
Frames from a tested take never enter training. This is same-session take
validation, with both recorders represented in training. It is not
leave-one-person-out or cross-day validation.

| Test population | Before | After |
|---|---:|---:|
| All 261 takes, including new hamza class | 83/261 = 31.8% | 241/261 = 92.3% |
| Only the 251 takes whose labels already existed | 83/251 = 33.1% | 233/251 = 92.8% |
| Hamza ء only | unsupported by old model | 8/10 |

For example: ح 0/10 → 10/10, ث 2/10 → 10/10, ر 0/10 → 8/10,
ت 3/10 → 6/10, and پ 0/11 → 8/11. The full table is in
`take_per_letter.csv`; individual outputs and confusion matrices are saved.
The after result is based on folds, not on testing the final all-data model
against its training takes.

## Transfer to original dataset signers

For each of the five original signers, hold out that person's dataset rows.
Train the same recipe in three conditions: the other four dataset signers
alone; those plus the 30 earlier Hassan rows; and those plus all 261 new takes.
No original test signer is included in any training condition. The augmented
classifier has 41 classes; these dataset tests cover only the original 40.

| Held-out signer | Dataset only | Prior calibration | + new session |
|---|---:|---:|---:|
| signer1-F | 21.9% | 23.5% | 23.5% |
| signer2-F | 32.5% | 31.8% | 35.8% |
| signer3-F | 29.0% | 29.4% | 31.5% |
| signer4-L | 22.0% | 22.0% | 21.4% |
| signer5-R | 27.2% | 28.5% | 27.8% |
| Pooled | 355/1345 = 26.4% | 361/1345 = 26.8% | 376/1345 = 28.0% |
| Mean of signer accuracies | 26.5% | 27.0% | 28.0% |
| Macro recall over the 40 dataset letters | 23.7% | 23.8% | 24.9% |

Relative to the prior-calibration condition, the pooled gain is 1.12 percentage
points. An exploratory paired bootstrap over the five test signers gives a
95% interval of -0.49 to +2.81 points. This is compatible with no transfer gain.
One signer is unchanged, two improve, and two decline slightly. Results are
agreement with original pseudo-labels, not independently annotated letters.

The usual `evaluation.load_dataset()` still excludes **all** `user-*` rows.
An unchanged dataset-only notebook will therefore still report 26.4% after
this merge. The 28.0% test deliberately adds calibration to training and must
be identified as an augmented-training protocol. It must not replace the
26.4% baseline without that explanation.

## Files and reproduction

`training_before/` and `model_before/` preserve the former data and browser
weights. `training_after/` and `model_after/` hold the evaluated deployment
candidate. `installation.json` records hashes for the files installed in the
repo root and `web/model_geom/`. `index_before.html` preserves the former UI.
Base revision: `5c79628`. These files accompany the calibration-data update;
the before/after snapshots preserve the inputs used for this experiment.

Run from the repo root (OUT is this result folder):

```bash
OUT=results/2026-11-13_calibration_audit
node "$OUT/score_shipped.mjs" "$PWD" calibration/calibration_2026-11-13.json /tmp/psl_before_check.json "$OUT/model_before"
.venv/bin/python "$OUT/evaluate_session.py" "$PWD" calibration/calibration_2026-11-13.json "$OUT"
node web/test_svm.mjs "$OUT/candidate_fixture.json"
node web/test_align.mjs
```

`evaluate_session.py` reads frozen `training_before`, so rerunning it after
the merge cannot silently turn the new session into baseline training data.
It replaces its own experiment outputs, not the root data or browser weights.
`apply_candidate.py` installs its saved candidate, checks the current files
against known hashes, and does not overwrite the frozen backups. The standard
`export_geometry_model.py` has an older fold-selection rule that would include
this 26-letter mixed group as an extra test fold; its metadata evaluation is
not the augmented five-signer protocol reported here.

Environment: Python 3.14.7, NumPy 2.5.4, pandas 3.1.0, SciPy 1.18.1,
scikit-learn 1.9.1, Node 25.8.2. Versions and raw results are in `summary.json`.
The final exported SVM passes 1,636 parity cases with zero mismatches;
`test_align.mjs` passes all eight checks. Both comparison-server profiles
were tested over local HTTP: BEFORE serves 40 classes, AFTER serves 41.

## Live before/after test

Run these in two separate terminals from the repo root:

```bash
python3 results/2026-11-13_calibration_audit/serve_comparison.py --profile before --port 8001
python3 results/2026-11-13_calibration_audit/serve_comparison.py --profile after --port 8000
```

Open http://localhost:8001/web/ for BEFORE and http://localhost:8000/web/ for
AFTER. The page heading identifies the profile. Each server serves the matching
weights with caching disabled; no model files need swapping. The BEFORE page
uses the saved earlier HTML. Test one camera page at a time so two inference
loops do not distort processing speed.

Record signer, conditions, intended letter, predicted letter, and spurious or
missed holds. Keep signing, distance and lighting comparable, press Reset
between attempts, and move between repeated holds. Compare existing letters
first; report ء separately because BEFORE cannot output it. There is no
counted post-update live result yet. Abdullah reports strong recognition in
a later setting with different lighting, using the same person and camera;
that observation has no scored attempt counts. The offline same-session
result does not include live segmentation errors.

For the next fair generalisation test, record a fresh session for each person
with their name recorded, keep it out of training, and score the frozen before
and after models. A genuinely new person needs a separate test. The current
file cannot determine whether Abdullah's calibration transfers to Hassan,
because take-level identities were not written down. Comparing with the papers
also needs matching units, metrics and split rules; the 92.3% take score is not
directly comparable with a paper's frame accuracy or character error rate.
