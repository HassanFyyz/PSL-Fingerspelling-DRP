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
| 2026-11-10 | Fix: `calibrate.py` "before" model included merged calibration rows | `user-*` rows excluded | Platt, unseeded | 22 Oct session: before 6/10, after 10/10 (the README's earlier 7/10 predates the fix and is within Platt run-to-run variation) | `python calibrate.py calibration/calibration_2026-10-22.json` |
