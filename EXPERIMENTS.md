# Experiment log

One row per result. Protocol unless stated: leave-one-signer-out over the five dataset
signers (calibration rows excluded), 74 geometry features, numbers are mean of the five folds.
Intervals are 95%, from resampling signers (`evaluation.bootstrap_by_signer`).

| date | change | config | seed | result | command |
|---|---|---|---|---|---|
| 2026-11-10 | Baseline re-run, shipped SVM | `SVC(C=10, gamma="scale", class_weight="balanced")` | none (deterministic) | letters 26.5% mean of folds, 26.4% pooled, CI 22.6–30.4; majority 10.0%; macro recall 23.7% | `headline_statistics.ipynb` |
| 2026-11-10 | Baseline, nested-tuned | C in {1,10,100}, gamma in {scale,0.01,0.1}, chosen on training signers | none | letters 25.9% mean of folds, CI 22.7–29.6 | `headline_statistics.ipynb` |
| 2026-11-10 | Shuffled-label null | labels permuted within signer, 30 runs | 0 | mean 3.9%, max 4.9% | `headline_statistics.ipynb` |
| 2026-11-10 | Fix: cross-session test trained on its own test session | `user-hassan-session2` rows now excluded from both models | Platt, unseeded | before fix on HEAD: 100/100/100%. After: no calibration 5/10, cross-session 10/10, same-session 10/10 (two letters; 10/10 has a 95% interval of 69–100%) | `python cross_session_test.py calibration/calibration_2026-10-26.json` |
| 2026-11-10 | Fix: `calibrate.py` "before" model included merged calibration rows | `user-*` rows excluded | Platt, unseeded | 22 Oct session: before 7/10 in three runs and 6/10 in one (Platt scaling is unseeded), after 10/10 | `python calibrate.py calibration/calibration_2026-10-22.json` |
| 2026-11-10 | Extraction check: `end_to_end.detect_pts` against `hold_landmarks.npy` | 150 pseudo-labelled hold frames re-detected | 0 | max difference 0.0 px in x, y and z, so the cache is `[x*w, y*h, z*w]` from image-mode detection at `minimum_frame` | `end_to_end_words.ipynb` |
| 2026-11-10 | Free segmentation on the 218 clips | `sofi_dynamic.segment_clip` defaults, MediaPipe at each minimum | none | 1,362 holds; count exact 32.1%, off by one 40.4%, off by 2+ 27.5%, MAE 1.19 | `python end_to_end.py` |
| 2026-11-10 | **End-to-end word accuracy** (current decoder, vote share as in the browser) | same SVM; test holds from free segmentation instead of pseudo-labels | none (deterministic) | **free: top-1 23.6% (CI 18.0–29.1), top-3 34.5% (26.2–39.8).** Oracle count: top-1 55.0% (38.1–70.0), top-3 80.3% (65.0–89.6). By count error: exact 70.0% top-1, off by one 2–3%, off by 2+ 0% | `end_to_end_words.ipynb` |
| 2026-11-10 | Same, Platt probabilities | `probability=True` | 0–4 | free top-1 24.7% mean (range 1.2), top-3 40.7%; oracle top-1 53.5% (range 2.1), top-3 75.2% | `end_to_end_words.ipynb` |
