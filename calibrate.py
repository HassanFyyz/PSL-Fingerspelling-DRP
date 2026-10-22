"""
Train and evaluate a letter model on your own recorded data.

Usage:
    python calibrate.py calibration_2026-10-22.json

Input comes from web/capture.html, which records takes of you signing letters and
saves landmarks plus masked hand crops.

Two things this is careful about
--------------------------------
**Takes are held out, not frames.** The ~70 frames inside one take are
near-duplicates of each other. Splitting them randomly puts near-identical images
on both sides and reports an accuracy several times too high -- the same mistake
that made the dataset look like 74.6% when the honest figure was 6.8%. Here, whole
takes are held out, so the model is always tested on a take it has never seen.

**Few classes make accuracy look good for free.** With 2 letters, a coin flip scores
50%, and two dissimilar letters will score near 100% whatever the method. Accuracy is
therefore reported against chance, and a separation score is reported alongside it,
which stays meaningful at any number of classes.
"""
import base64
import json
import sys
from itertools import combinations

import numpy as np

from hand_geometry import hand_features


def load(path):
    """Returns (landmark array, crops, letters, take ids)."""
    d = json.load(open(path, encoding="utf-8"))
    pts, crops, letters, takes = [], [], [], []
    size = d.get("img_size", 64)
    for t in d["takes"]:
        tid = f"{t['letter']}#{t['take']}"
        for fr in t["frames"]:
            pts.append(np.array(fr["pts"], dtype=np.float32))
            letters.append(t["letter"])
            takes.append(tid)
            if fr.get("crop"):
                crops.append((len(pts) - 1,
                              np.frombuffer(base64.b64decode(fr["crop"]), dtype=np.uint8)
                                .reshape(size, size)))
    return (np.stack(pts), crops, np.array(letters), np.array(takes), d)


def leave_one_take_out(F, y, takes):
    """Accuracy with whole takes held out. Returns (accuracy, per-letter dict)."""
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    correct, total = 0, 0
    per_letter = {}
    for held in sorted(set(takes)):
        te = takes == held
        tr = ~te
        if len(set(y[tr])) < 2:
            continue
        m = make_pipeline(StandardScaler(),
                          SVC(C=10, gamma="scale", class_weight="balanced"))
        m.fit(F[tr], y[tr])
        pred = m.predict(F[te])
        # one vote per take: the take's majority prediction
        vals, counts = np.unique(pred, return_counts=True)
        take_pred = vals[counts.argmax()]
        truth = y[te][0]
        ok = take_pred == truth
        correct += ok
        total += 1
        per_letter.setdefault(truth, [0, 0])
        per_letter[truth][0] += ok
        per_letter[truth][1] += 1
    return (correct / max(total, 1)), per_letter, total


def separation(F, y, takes):
    """Mean between-letter distance over mean within-letter distance.

    Computed between TAKE centroids, so it is not inflated by duplicate frames.
    Above 1 means takes of the same letter are closer to each other than to other
    letters. Unlike accuracy, this does not get easier as classes are removed.
    """
    cents, labs = [], []
    for t in sorted(set(takes)):
        m = takes == t
        cents.append(F[m].mean(0))
        labs.append(y[m][0])
    C = np.array(cents)
    labs = np.array(labs)
    C = (C - C.mean(0)) / (C.std(0) + 1e-8)
    within, between = [], []
    for i, j in combinations(range(len(C)), 2):
        d = float(np.linalg.norm(C[i] - C[j]))
        (within if labs[i] == labs[j] else between).append(d)
    if not within or not between:
        return float("nan"), len(C)
    return float(np.mean(between) / np.mean(within)), len(C)


def main(path):
    pts, crops, y, takes, meta = load(path)
    letters = sorted(set(y))
    n_takes = len(set(takes))
    print(f"loaded {len(pts)} frames, {n_takes} takes, {len(letters)} letters: {' '.join(letters)}")
    print(f"recorded at {meta.get('fps','?')} fps, handedness {meta.get('handedness')}")
    for L in letters:
        print(f"   {L}: {len(set(takes[y==L]))} takes, {int((y==L).sum())} frames")

    if n_takes < 4:
        print("\nNot enough takes to evaluate. Record at least 2 per letter.")
        return

    F = hand_features(pts)
    print(f"\ngeometry features: {F.shape[1]} per frame")

    acc, per_letter, total = leave_one_take_out(F, y, takes)
    sep, ncent = separation(F, y, takes)
    chance = 1.0 / len(letters)

    print("\n--- leave-one-TAKE-out (each take held out in turn) ---")
    print(f"  accuracy     : {100*acc:.1f}%   ({total} takes tested)")
    print(f"  chance       : {100*chance:.1f}%   ({len(letters)} letters)")
    print(f"  separation   : {sep:.2f}   (>1 means same-letter takes cluster together)")
    for L, (c, n) in sorted(per_letter.items()):
        print(f"     {L}: {c}/{n} takes correct")

    print("\n--- how to read this ---")
    if len(letters) <= 3:
        print(f"  With only {len(letters)} letters, accuracy is NOT evidence the approach works:")
        print(f"  chance alone is {100*chance:.0f}%, and dissimilar letters score near 100%")
        print("  whatever the method. What this run actually tells you is whether the")
        print("  CAPTURE is sound: hand detected throughout, crops framed correctly, and")
        print("  takes of the same letter landing near each other.")
        print(f"  The separation score ({sep:.2f}) is the meaningful number here, and it")
        print("  stays comparable as you add letters.")
        print("\n  Record ~8 letters for a result that says something about accuracy.")
    else:
        print(f"  {len(letters)} letters, chance {100*chance:.1f}%. Accuracy is now meaningful.")
        print(f"  For reference, the dataset model scores 25.9% on an unseen signer")
        print("  across all 40 letters.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
