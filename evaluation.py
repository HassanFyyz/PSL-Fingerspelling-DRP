"""
Shared evaluation helpers: one definition of the folds, the baseline and the
confidence intervals, so every notebook reports numbers the same way.

Rules these helpers enforce
---------------------------
* The headline protocol is leave-one-signer-out over the FIVE DATASET SIGNERS.
  Calibration rows (`user-*`) are dropped by `load_dataset()` unless asked for:
  they cover two letters, and a fold of two letters is not a 40-class test.
* With five signers there are five independent test units, not 1,345. The
  intervals here resample SIGNERS (`bootstrap_by_signer`) or use the spread of
  the five fold scores (`fold_t_interval`). An interval that resamples holds
  would be several times too narrow.
* The majority baseline is the most common letter in the TRAINING signers,
  scored on the held-out signer -- not the pooled class frequency.
"""
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from hand_geometry import hand_features

PTS_PATH = "hold_landmarks.npy"
META_PATH = "hold_meta.csv"


def load_dataset(include_users=False, pts_path=PTS_PATH, meta_path=META_PATH):
    """(P, M): landmarks (n, 21, 3) and metadata, dataset signers only by default."""
    P = np.load(pts_path)
    M = pd.read_csv(meta_path)
    if not include_users:
        keep = ~M.signer.str.startswith("user-").to_numpy()
        P, M = P[keep], M[keep].reset_index(drop=True)
    return P, M


def svm(C=10, gamma="scale", probability=False, seed=0):
    """The shipped recipe (export_geometry_model.py)."""
    return make_pipeline(StandardScaler(),
                         SVC(C=C, gamma=gamma, class_weight="balanced",
                             probability=probability, random_state=seed))


def loso_predict(F, y, signers, make_model=svm, proba=False):
    """Leave-one-signer-out predictions for every row.

    Returns `pred` (n,), and when `proba` a dict signer -> (classes, P) holding
    the held-out signer's class probabilities, rows in the order of that
    signer's rows in F.
    """
    pred = np.empty(len(y), dtype=object)
    probs = {}
    for s in sorted(set(signers)):
        tr, te = signers != s, signers == s
        m = make_model().fit(F[tr], y[tr])
        pred[te] = m.predict(F[te])
        if proba:
            probs[s] = (list(m.classes_), m.predict_proba(F[te]))
    return (pred, probs) if proba else pred


def majority_correct(y, signers):
    """1.0 where the training-fold majority letter equals the true letter."""
    out = np.zeros(len(y))
    for s in sorted(set(signers)):
        tr, te = signers != s, signers == s
        out[te] = y[te] == Counter(y[tr]).most_common(1)[0][0]
    return out


def per_fold(values, signers):
    """Mean of `values` within each signer, as a Series indexed by signer."""
    return pd.Series({s: float(np.mean(values[signers == s])) for s in sorted(set(signers))})


def fold_t_interval(fold_scores, level=0.95):
    """(mean, half_width) of a t-interval over the fold scores."""
    a = np.asarray(fold_scores, dtype=float)
    half = stats.t.ppf(0.5 + level / 2, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a))
    return float(a.mean()), float(half)


def bootstrap_by_signer(values, signers, B=20000, seed=0, level=0.95):
    """Percentile interval for the pooled mean of `values`, resampling signers.

    `values` is one number per test unit (hold or clip); `signers` says which
    signer each belongs to. Whole signers are drawn with replacement.
    """
    rng = np.random.default_rng(seed)
    names = sorted(set(signers))
    sums = np.array([values[signers == s].sum() for s in names], dtype=float)
    ns = np.array([(signers == s).sum() for s in names], dtype=float)
    draw = rng.integers(0, len(names), size=(B, len(names)))
    means = sums[draw].sum(1) / ns[draw].sum(1)
    lo, hi = np.percentile(means, [50 * (1 - level), 50 * (1 + level)])
    return float(lo), float(hi)


def permutation_null(F, y, signers, make_model=svm, runs=30, seed=0):
    """Mean-of-folds LOSO accuracy with labels shuffled WITHIN each signer.

    Shuffling within signer keeps every signer's class frequencies, so the null
    answers "what does this pipeline score when the features carry no letter
    information?" rather than something easier to beat.
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(runs):
        yp = y.copy()
        for s in set(signers):
            i = np.where(signers == s)[0]
            yp[i] = rng.permutation(yp[i])
        pred = loso_predict(F, yp, signers, make_model)
        out.append(per_fold((pred == yp).astype(float), signers).mean())
    return np.array(out)


def summarise(name, correct, signers, baseline=None):
    """One row: pooled %, mean of folds +- t half-width, signer-bootstrap CI."""
    correct = np.asarray(correct, dtype=float)
    f = per_fold(correct, signers)
    mean, half = fold_t_interval(f)
    lo, hi = bootstrap_by_signer(correct, signers)
    row = {"metric": name, "n": len(correct), "pooled %": 100 * correct.mean(),
           "mean of folds %": 100 * mean, "t half-width": 100 * half,
           "boot lo %": 100 * lo, "boot hi %": 100 * hi}
    if baseline is not None:
        d = per_fold(correct - np.asarray(baseline, dtype=float), signers)
        row["vs baseline (pts)"] = 100 * d.mean()
        row["paired t p"] = float(stats.ttest_1samp(d, 0).pvalue)
    return row


def geometry_features(P):
    """hand_features(P), kept here so notebooks import one module."""
    return hand_features(P)
