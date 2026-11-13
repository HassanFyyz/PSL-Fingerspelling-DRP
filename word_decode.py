"""
Lexicon decoding in Python, mirroring web/word_decode.js, so word accuracy can
be measured offline with the same logic the live page runs.

`decode_current` is a line-for-line port of `decodeWord` in web/word_decode.js:
hold i is compared with letter i, and only words within `slack` letters of the
hold count are considered.

`decode_aligned` is the replacement. Two changes, each measured separately in
alignment_decoder.ipynb:

1. ALIGNMENT. The segmenter finds the right number of holds on about a third of
   clips. Comparing hold i with letter i means one missed or spurious hold
   shifts every later letter, and top-1 falls from 70% to 2-3%. Here a hold may
   be skipped (it was a transition, not a letter) and a letter may be skipped
   (its hold was missed), each at a fixed cost, and the best monotonic
   alignment is found by dynamic programming.

2. LIKELIHOOD RATIO. The classifier outputs P(letter | hold). Multiplying those
   along a word counts the letter prior once per letter, so words made of
   common letters win. Dividing by the prior gives a quantity proportional to
   P(hold | letter), which is what a word model should multiply. A matched hold
   then scores log P(letter | hold) - log P(letter): positive when the hold is
   evidence FOR that letter, negative when against. A skipped hold scores `ins`
   and a skipped letter `dele`, on the same scale, so no length normalisation
   and no hold-count filter are needed.

This needs real probabilities. The browser's vote share is not one, and dividing
it by the prior makes things worse; use `prior=None` with vote share.
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


def decode_aligned(probs, classes, lexicon, prior=None, ins=-1.0, dele=-1.0, floor=1e-6):
    """Ranked list of words by best alignment of holds to letters.

    probs    (n_holds, n_classes) class scores; rows are renormalised to sum 1
    prior    {letter: frequency} from the TRAINING signers, or None for uniform
    ins      score of a hold matched to no letter (<= 0)
    dele     score of a letter matched to no hold (<= 0)
    """
    n = len(probs)
    k = len(classes)
    ci = {c: i for i, c in enumerate(classes)}
    if n:
        P = np.asarray(probs, dtype=float)
        P = P / np.maximum(P.sum(1, keepdims=True), EPS)
        pr = (np.array([max(prior.get(c, 0.0), 1.0 / (10 * k)) for c in classes])
              if prior is not None else np.full(k, 1.0 / k))
        G = np.log(np.maximum(P, floor)) - np.log(pr)      # gain of matching hold i to class c
    scored = []
    for word, letters in lexicon.items():
        if any(l not in ci for l in letters):
            continue
        idx = [ci[l] for l in letters]
        m = len(idx)
        prev = np.arange(m + 1) * dele                      # no holds used yet
        for i in range(n):
            cur = np.empty(m + 1)
            cur[0] = prev[0] + ins
            g = G[i]
            for j in range(1, m + 1):
                cur[j] = max(prev[j - 1] + g[idx[j - 1]], prev[j] + ins, cur[j - 1] + dele)
            prev = cur
        scored.append((prev[m], word))
    scored.sort(key=lambda t: -t[0])
    return [w for _, w in scored]


GRID_INS = (0.0, -0.5, -1.0, -2.0)
GRID_DELE = (-0.5, -1.0, -2.0, -3.0)


def tune_aligned(clips, lexicon, which="free", use_prior=True,
                 grid_ins=GRID_INS, grid_dele=GRID_DELE):
    """(ins, dele) with the best top-1 on `clips`, ties broken by top-3.

    Pass clips the test signer is not part of (end_to_end.nested_clips "tune").
    """
    best = None
    for ins in grid_ins:
        for dele in grid_dele:
            t1 = t3 = 0
            for c in clips:
                r = decode_aligned(c[which], c["classes"], lexicon,
                                   c["prior"] if use_prior else None, ins, dele)
                t1 += r[:1] == [c["word"]]
                t3 += c["word"] in r[:3]
            if best is None or (t1, t3) > best[0]:
                best = ((t1, t3), ins, dele)
    return best[1], best[2]
