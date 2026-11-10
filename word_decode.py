"""
Lexicon decoding in Python, mirroring web/word_decode.js, so word accuracy can
be measured offline with the same logic the live page runs.

`decode_current` is a line-for-line port of `decodeWord` in web/word_decode.js:
hold i is compared with letter i, and only words within `slack` letters of the
hold count are considered.
"""
import numpy as np

EPS = 1e-12


def vote_share(model, F):
    """Per-class fraction of one-vs-one wins, as web/svm_classifier.js computes it.

    `model` is a fitted Pipeline(StandardScaler, SVC). Returns (n, n_classes).
    Deterministic, unlike `predict_proba`, whose Platt scaling is fitted on
    random internal folds.
    """
    svc = model.named_steps["svc"]
    X = model.named_steps["standardscaler"].transform(F)
    k = len(svc.classes_)
    D = svc.decision_function(X) if svc.decision_function_shape == "ovo" else None
    if D is None:
        shape, svc.decision_function_shape = svc.decision_function_shape, "ovo"
        D = svc.decision_function(X)
        svc.decision_function_shape = shape
    V = np.zeros((len(X), k))
    p = 0
    for i in range(k):
        for j in range(i + 1, k):
            win = D[:, p] > 0
            V[win, i] += 1
            V[~win, j] += 1
            p += 1
    return V / (k - 1)


def decode_current(probs, classes, lexicon, slack=1, pen=0.7):
    """Ranked list of words. Port of decodeWord in web/word_decode.js."""
    n = len(probs)
    if n == 0:
        return []
    ci = {c: i for i, c in enumerate(classes)}
    scored = []
    for word, letters in lexicon.items():
        if abs(len(letters) - n) > slack:
            continue
        L = min(len(letters), n)
        if any(l not in ci for l in letters[:L]):
            continue
        logp = sum(np.log(max(probs[i][ci[letters[i]]], EPS)) for i in range(L))
        scored.append((logp / L - pen * abs(len(letters) - n), word))
    scored.sort(key=lambda t: -t[0])
    return [w for _, w in scored]
