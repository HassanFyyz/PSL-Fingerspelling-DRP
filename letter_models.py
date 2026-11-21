"""
Every letter classifier compared in `train_val_test_models.ipynb`, behind one
interface, so models that need different inputs can be scored by the same loop.

    m = make()                                  # a fresh, unfitted model
    m.fit(P, y, groups=signers)                 # P: (n, 21, 3) landmarks in isotropic pixels
    m.predict_proba(P)  ->  (n, len(m.classes_))
    m.predict(P)

Each model computes its own features from the landmarks:
    "geom"  the 74 angle and distance features of hand_geometry.py (what ships)
    "rich"  289 features: angles, all 210 pairwise distances, palm-frame coordinates
    "pts"   the landmarks themselves (the graph network)

`groups` (the signer of each row) is ignored by most models. Two use it:
`CleanedLabels` to hold signers out while judging labels, and
`SignerNormalised`, which is TRANSDUCTIVE and is marked as such wherever it is reported.
"""
import numpy as np
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import models
import torch_models as tm
from hand_geometry import hand_features

FEATURES = {"geom": hand_features, "rich": tm.rich_features, "pts": lambda P: P}


class OnFeatures:
    """A scikit-learn style model applied to one feature set."""

    def __init__(self, feat, model):
        self.feat, self.model = feat, model

    def fit(self, P, y, groups=None, sample_weight=None):
        X = FEATURES[self.feat](P)
        if sample_weight is None:
            self.model.fit(X, y)
        else:
            self.model.fit(X, y, sample_weight=sample_weight)
        self.classes_ = self.model.classes_
        return self

    def predict_proba(self, P):
        return self.model.predict_proba(FEATURES[self.feat](P))

    def predict(self, P):
        return self.classes_[self.predict_proba(P).argmax(1)]


class SvmVoteShare(OnFeatures):
    """The shipped SVM scored the way the browser scores it: the share of
    one-vs-one contests each letter wins, not Platt probabilities."""

    def __init__(self, seed=0):
        super().__init__("geom", models.svm(seed))

    def predict_proba(self, P):
        svc = self.model[-1]
        Z = self.model[:-1].transform(hand_features(P))
        n = len(self.classes_)
        out = []
        for s in range(0, len(Z), 512):
            margins = svc.decision_function(Z[s:s + 512]) if svc.decision_function_shape == "ovo" \
                else _ovo_margins(svc, Z[s:s + 512])
            votes = np.zeros((len(margins), n))
            p = 0
            for i in range(n):
                for j in range(i + 1, n):
                    win = margins[:, p] > 0
                    votes[:, i] += win
                    votes[:, j] += ~win
                    p += 1
            out.append(votes / (n - 1))
        return np.concatenate(out)


def _ovo_margins(svc, Z):
    shape = svc.decision_function_shape
    svc.decision_function_shape = "ovo"
    try:
        return svc.decision_function(Z)
    finally:
        svc.decision_function_shape = shape


class MeanOf:
    """Average of the members' probabilities (arithmetic, or geometric with geo=True)."""

    def __init__(self, members, geo=False):
        self.members, self.geo = members, geo

    def fit(self, P, y, groups=None, sample_weight=None):
        for m in self.members:
            m.fit(P, y, groups=groups, sample_weight=sample_weight)
        self.classes_ = self.members[0].classes_
        for m in self.members[1:]:
            assert list(m.classes_) == list(self.classes_)
        return self

    def predict_proba(self, P):
        ps = [m.predict_proba(P) for m in self.members]
        if self.geo:
            g = np.exp(np.mean([np.log(p + 1e-4) for p in ps], axis=0))
            return g / g.sum(1, keepdims=True)
        return np.mean(ps, axis=0)

    def predict(self, P):
        return self.classes_[self.predict_proba(P).argmax(1)]


class CleanedLabels:
    """Drop training rows whose label a model trained on OTHER signers ranks
    below `keep_rank`, then fit `final` on what is left.

    The dataset labels are inferred from the videos, so some are wrong. This
    removes the rows the other signers most disagree with. It needs at least
    two signers in the training data; rows of a signer seen alone are kept.
    """

    def __init__(self, final, keep_rank=5, seed=0):
        self.final, self.keep_rank, self.seed = final, keep_rank, seed

    def fit(self, P, y, groups=None, sample_weight=None):
        X = hand_features(P)
        g = np.asarray(groups)
        keep = np.ones(len(y), dtype=bool)
        if len(set(g)) > 1:
            for s in sorted(set(g)):
                a, b = g != s, g == s
                judge = models.extra_trees(self.seed).fit(X[a], y[a])
                pr = judge.predict_proba(X[b])
                col = {c: i for i, c in enumerate(judge.classes_)}
                for k, row in zip(np.where(b)[0], pr):
                    if y[k] in col and not str(g[k]).startswith("user-"):   # calibration labels are real
                        keep[k] = (row > row[col[y[k]]]).sum() < self.keep_rank
        self.kept_ = keep
        self.final.fit(P[keep], y[keep], groups=g[keep])
        self.classes_ = self.final.classes_
        return self

    def predict_proba(self, P):
        return self.final.predict_proba(P)

    def predict(self, P):
        return self.final.predict(P)


class SignerNormalised:
    """TRANSDUCTIVE. Standardises each signer's features by that signer's own
    mean and spread, including the test signer's, whose rows must therefore be
    passed to predict_proba together as one batch. It uses no test labels, but
    it does use the test signer's unlabelled holds, so it answers "how well can
    we do after watching a new person for a while", not "on their first hold".
    """

    def __init__(self, feat, model):
        self.feat, self.model = feat, model

    @staticmethod
    def _norm(X):
        return (X - X.mean(0)) / (X.std(0) + 1e-6)

    def fit(self, P, y, groups=None, sample_weight=None):
        X = FEATURES[self.feat](P).astype(float)
        g = np.asarray(groups)
        for s in set(g):
            X[g == s] = self._norm(X[g == s])
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def predict_proba(self, P):
        return self.model.predict_proba(self._norm(FEATURES[self.feat](P).astype(float)))

    def predict(self, P):
        return self.classes_[self.predict_proba(P).argmax(1)]


# ─── factories ───────────────────────────────────────────────────────────────
def _et(seed=0, **k):
    return models.extra_trees(seed, **k)


def _torch_mlp(seed=0):
    # no class weighting, light regularisation; three initialisations averaged
    return tm.TorchMLP(balanced=False, noise=0.1, dropout=0.3, n_models=3, epochs=80, seed=seed)


def _logreg():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced"))


def _histgb(seed=0):
    return HistGradientBoostingClassifier(max_iter=80, learning_rate=0.1, max_depth=4, l2_regularization=1.0,
                                          class_weight="balanced", random_state=seed)


# name -> (group, factory). "earlier" = already compared in model_alternatives.ipynb.
CANDIDATES = {
    # scored by one-vs-one votes, as SVC.predict and the browser do (Platt probabilities rank differently)
    "SVM-RBF (shipped)":                   ("earlier", lambda s=0: SvmVoteShare(s)),
    "ExtraTrees":                          ("earlier", lambda s=0: OnFeatures("geom", _et(s))),
    "MLP 128, alpha=3":                    ("earlier", lambda s=0: OnFeatures("geom", models.mlp(s))),
    "LDA, shrinkage":                      ("earlier", lambda s=0: OnFeatures("geom", models.lda(s))),
    "kNN, k=15":                           ("earlier", lambda s=0: OnFeatures("geom", models.knn(s))),
    "ensemble: SVM+MLP+ExtraTrees+LDA":    ("earlier", lambda s=0: OnFeatures("geom", models.ensemble(s))),
    "Random forest":                       ("new", lambda s=0: OnFeatures("geom", RandomForestClassifier(
        500, class_weight="balanced_subsample", min_samples_leaf=2, n_jobs=-1, random_state=s))),
    "Gradient boosting (HistGB)":          ("new", lambda s=0: OnFeatures("geom", _histgb(s))),
    "Logistic regression":                 ("new", lambda s=0: OnFeatures("geom", _logreg())),
    "ExtraTrees, rich features":           ("new", lambda s=0: OnFeatures("rich", _et(s))),
    "LDA, rich features":                  ("new", lambda s=0: OnFeatures("rich", models.lda(s))),
    "ExtraTrees, cleaned labels":          ("new", lambda s=0: CleanedLabels(OnFeatures("geom", _et(s)), 5, s)),
    "Neural net (PyTorch MLP)":            ("new", lambda s=0: OnFeatures("geom", _torch_mlp(s))),
    "Neural net (PyTorch MLP), rich":      ("new", lambda s=0: OnFeatures("rich", _torch_mlp(s))),
    "Graph network on the hand skeleton":  ("new", lambda s=0: OnFeatures("pts", tm.HandGCN(
        balanced=False, n_models=2, epochs=60, seed=s))),
    "ExtraTrees + neural net":             ("new", lambda s=0: MeanOf([OnFeatures("rich", _et(s)),
                                                                       OnFeatures("geom", _torch_mlp(s))])),
    "5-model blend (geometric mean)":      ("new", lambda s=0: MeanOf([
        OnFeatures("geom", _et(s)), OnFeatures("rich", _et(s)), OnFeatures("geom", models.lda(s)),
        OnFeatures("rich", models.lda(s)), OnFeatures("geom", models.mlp(s))], geo=True)),
}

# Transductive: sees the test signer's unlabelled holds. Reported separately.
TRANSDUCTIVE = {
    "ExtraTrees rich, per-signer normalised": lambda s=0: SignerNormalised("rich", _et(s)),
    "LDA, per-signer normalised":             lambda s=0: SignerNormalised("geom", models.lda(s)),
}
