"""
Can the model recognise YOUR letters, choosing from all 40?

Usage:
    python calibrate.py calibration_2026-10-22.json

You record yourself signing a few letters (web/capture.html). The model keeps all
40 letter classes from the dataset -- you are not reducing the problem. The
question is whether, when you sign alif, it picks alif out of 40 options.
Chance is 2.5%.

It reports two numbers:

  BEFORE  trained on the 5 dataset signers only, tested on your takes.
          This is what the live demo does today, and it is the number that has
          been disappointing.

  AFTER   trained on the dataset PLUS your other takes of that letter, tested on
          a take of yours it has never seen.

The gap between them is the value of calibration. Takes are held out whole,
never split into frames -- frames inside one take are near-duplicates, and
splitting them randomly is what made the dataset look like 74.6% when the honest
figure was 6.8%.

Landmark geometry (joint angles + palm-normalised distances) is used rather than
image crops, because those features are invariant to image resolution. Your
webcam and the 240x350 dataset videos can therefore be mixed directly.
"""
import base64
import json
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

from hand_geometry import hand_features

DATASET_PTS = "hold_landmarks.npy"
DATASET_META = "hold_meta.csv"


def load_user(path):
    d = json.load(open(path, encoding="utf-8"))
    pts, letters, takes = [], [], []
    for t in d["takes"]:
        tid = f"{t['letter']}#{t['take']}"
        for fr in t["frames"]:
            pts.append(np.array(fr["pts"], dtype=np.float32))
            letters.append(t["letter"])
            takes.append(tid)
    return np.stack(pts), np.array(letters), np.array(takes), d


def fit(F, y, C=10, gamma="scale"):
    m = make_pipeline(StandardScaler(),
                      SVC(C=C, gamma=gamma, class_weight="balanced", probability=True))
    m.fit(F, y)
    return m


def take_vote(model, F):
    """One prediction per take: average the per-frame probabilities."""
    P = model.predict_proba(F).mean(0)
    cls = model.named_steps["svc"].classes_
    order = np.argsort(P)[::-1]
    return cls[order[0]], [(cls[i], P[i]) for i in order[:3]]


def main(path):
    Pd = np.load(DATASET_PTS)
    Md = pd.read_csv(DATASET_META)
    Fd, yd = hand_features(Pd), Md.letter.to_numpy()

    Pu, yu, takes, meta = load_user(path)
    Fu = hand_features(Pu)
    user_letters = sorted(set(yu))

    print(f"dataset : {len(Fd)} holds, {len(set(yd))} letters, {Md.signer.nunique()} signers")
    print(f"you     : {len(Fu)} frames, {len(set(takes))} takes, letters {' '.join(user_letters)}")
    print(f"          recorded at {meta.get('fps','?')} fps, hand {meta.get('handedness')}")
    for L in user_letters:
        print(f"            {L}: {len(set(takes[yu == L]))} takes")

    missing = [L for L in user_letters if L not in set(yd)]
    if missing:
        print(f"\nWARNING: {' '.join(missing)} not present in the dataset; "
              f"only your own examples will teach it.")

    n_classes = len(set(yd) | set(yu))
    print(f"\nthe model chooses between {n_classes} letters -- chance is {100/n_classes:.1f}%")

    # ---------- BEFORE: dataset only ----------
    base = fit(Fd, yd)
    before = {}
    for t in sorted(set(takes)):
        m = takes == t
        pred, top3 = take_vote(base, Fu[m])
        before[t] = (yu[m][0], pred, top3)

    # ---------- AFTER: dataset + your other takes ----------
    after = {}
    for t in sorted(set(takes)):
        te = takes == t
        tr = ~te
        F = np.vstack([Fd, Fu[tr]])
        y = np.concatenate([yd, yu[tr]])
        model = fit(F, y)
        pred, top3 = take_vote(model, Fu[te])
        after[t] = (yu[te][0], pred, top3)

    def summarise(res, tag):
        ok = sum(1 for truth, pred, _ in res.values() if truth == pred)
        print(f"\n{tag}: {ok}/{len(res)} takes correct ({100*ok/len(res):.0f}%)")
        for t in sorted(res):
            truth, pred, top3 = res[t]
            mark = "OK " if truth == pred else "   "
            alts = ", ".join(f"{L} {100*p:.0f}%" for L, p in top3)
            print(f"  {mark} {t:<10} you signed {truth}  ->  model said {pred}      [top3: {alts}]")
        return ok / len(res)

    a = summarise(before, "BEFORE calibration (dataset signers only)")
    b = summarise(after, "AFTER calibration (dataset + your other takes)")

    print("\n" + "=" * 62)
    print(f"  before : {100*a:.0f}%      after : {100*b:.0f}%      chance : {100/n_classes:.1f}%")
    print("=" * 62)

    per = defaultdict(lambda: [0, 0])
    for truth, pred, _ in after.values():
        per[truth][0] += truth == pred
        per[truth][1] += 1
    print("\nafter calibration, per letter:")
    for L in sorted(per):
        c, n = per[L]
        print(f"   {L}: {c}/{n} takes correct")

    print("\n--- what this means ---")
    if b > a:
        print(f"  Calibration helped: {100*a:.0f}% -> {100*b:.0f}%. Adding even a few examples of")
        print("  your own hand moves the model, which is the result that matters.")
    elif b == a == 1.0:
        print("  Both perfect. The letters you picked may be easy to tell apart;")
        print("  try a couple of similar-looking ones to make it a harder test.")
    else:
        print(f"  Calibration did not help here ({100*a:.0f}% -> {100*b:.0f}%). With few takes")
        print("  that can be noise. Record more takes before concluding anything.")
    print(f"\n  Every prediction above is a choice among {n_classes} letters, so each correct")
    print(f"  one beats {100/n_classes:.1f}% chance. The model can still say anything on a")
    print("  pose you have not trained -- that is expected and is not what this tests.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
