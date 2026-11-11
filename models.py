"""
Letter classifiers compared against the shipped RBF SVM, as factories taking a
seed, so every notebook builds them the same way.

All take the 74 geometry features and expose fit / predict / predict_proba /
classes_. Settings were fixed before looking at held-out results; where a
family has a setting worth tuning, `model_alternatives.ipynb` chooses it inside
the training signers.

Note on the SVM: `predict` and `argmax(predict_proba)` disagree. `predict` uses
the class-weighted one-vs-one vote; Platt probabilities are fitted afterwards
and largely undo the class weighting. Tables that rank by probability therefore
show a different SVM accuracy from tables that use `predict`.
"""
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def svm(seed=0, C=10, gamma="scale"):
    return make_pipeline(StandardScaler(), SVC(C=C, gamma=gamma, class_weight="balanced",
                                               probability=True, random_state=seed))


def extra_trees(seed=0, n=500, leaf=2):
    return ExtraTreesClassifier(n, class_weight="balanced_subsample", min_samples_leaf=leaf,
                                n_jobs=-1, random_state=seed)


def mlp(seed=0, hidden=128, alpha=3.0):
    """One hidden layer with heavy L2. alpha is the setting that matters."""
    return make_pipeline(StandardScaler(), MLPClassifier((hidden,), alpha=alpha, max_iter=600,
                                                         random_state=seed))


def lda(seed=0):
    return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))


def knn(seed=0, k=15):
    return make_pipeline(StandardScaler(), KNeighborsClassifier(k, weights="distance"))


class MeanEnsemble:
    """Average of the members' class probabilities."""

    def __init__(self, members):
        self.members = members

    def fit(self, X, y):
        for m in self.members:
            m.fit(X, y)
        self.classes_ = self.members[0].classes_
        for m in self.members[1:]:
            assert list(m.classes_) == list(self.classes_)
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.members], axis=0)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


def ensemble(seed=0):
    return MeanEnsemble([svm(seed), mlp(seed), extra_trees(seed), lda(seed)])


CANDIDATES = {
    "SVM-RBF (shipped)": svm,
    "ExtraTrees": extra_trees,
    "MLP 128, alpha=3": mlp,
    "LDA, shrinkage": lda,
    "kNN, k=15": knn,
    "ensemble: SVM+MLP+ExtraTrees+LDA": ensemble,
}
