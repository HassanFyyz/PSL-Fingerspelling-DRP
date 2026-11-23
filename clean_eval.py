"""
Step 2: a clean test. Do the letter sheets help or hurt?

The video holds ("sheets") are labelled by where a letter should fall in a word
video, so some labels are wrong (e.g. م crops that show the neighbouring ل).
Every earlier new-person score was measured against those same labels, so it
cannot tell whether the sheets help. This test uses only correctly labelled data
as the answer key:

  K  Kamran's images: one letter per folder (signer(s) unknown, one group)
  C  our calibration takes: each take is one letter signed on purpose
  S  the video holds behind the letter sheets (five dataset signers)

All static letters only (splits.DYNAMIC_LETTERS removed everywhere).

Training options, each scored in two rounds; a group is never in training when
it is the test group:

  option            round A: test K        round B: test C
  sheets only       train S                train S
  clean only        train C                train K
  clean + sheets    train S + C            train S + K      (= what we ship now)

DECISION RULE (fixed 23 Nov, before the first run). Primary model: the shipped
SVM recipe. Compare "clean only" with "clean + sheets" on accuracy:
  * clean only ahead by >= 3 points in BOTH rounds  -> take the sheets out of training
  * clean + sheets ahead by >= 3 points in BOTH rounds -> keep them
  * anything else -> undecided: try cleaned sheets (bad crops dropped) and more people
ExtraTrees is reported as a check; if it disagrees with the SVM the result is
called undecided whatever the SVM says.

Run:  PSL_KAMRAN_DIR=~/Desktop/train python clean_eval.py
      (writes results/2026-11-23_clean_test/; Kamran's landmarks never leave his folder)
"""
import os
import sys

import numpy as np
import pandas as pd

import image_dataset
import models
from hand_geometry import hand_features
from splits import DYNAMIC_LETTERS

OUT = os.path.join("results", "2026-11-23_clean_test")
MARGIN = 3.0                                   # points, see DECISION RULE

OPTIONS = {"sheets only": {"S"}, "clean only": {"C", "K"}, "clean + sheets": {"S", "C", "K"}}
ROUNDS = {"A: test Kamran": "K", "B: test calibration": "C"}
MODELS = {"SVM": models.svm, "ExtraTrees": models.extra_trees}


def groups():
    """{'S'|'C'|'K': (features, letters, source)} for the static letters."""
    P = np.load("hold_landmarks.npy")
    M = pd.read_csv("hold_meta.csv")
    out = {}
    cal = M.signer.str.startswith("user-").to_numpy()
    for name, mask in (("S", ~cal), ("C", cal)):
        out[name] = (hand_features(P[mask]), M.letter.to_numpy()[mask], M.signer.to_numpy()[mask])
    PK, MK = image_dataset.load()
    keep = ((MK.detected == 1) & ~MK.duplicated("sha1")).to_numpy()
    out["K"] = (hand_features(PK[keep]), MK.letter.to_numpy()[keep], MK.split.to_numpy()[keep])
    for name, (F, y, src) in out.items():
        st = ~np.isin(y, DYNAMIC_LETTERS)
        out[name] = (F[st], y[st], src[st])
    return out


def stack(G, names):
    names = sorted(names)
    return (np.vstack([G[n][0] for n in names]), np.concatenate([G[n][1] for n in names]))


def letter_recall(y, pred):
    """Per-letter recall, for the letters present in the test group."""
    return pd.Series(pred == y).groupby(y).mean()


def boot_diff(ra, rb, B=20000, seed=0):
    """95% interval for mean(ra - rb) over letters, resampling letters."""
    d = (ra - rb).dropna().to_numpy()
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return np.percentile(m, [2.5, 97.5])


def run(seed=0, verbose=True):
    G = groups()
    if verbose:
        for n, (F, y, _) in G.items():
            print(f"  {n}: {len(y)} rows, {len(set(y))} letters")
    rows, recalls, preds = [], {}, []
    for rname, test in ROUNDS.items():
        Fte, yte = G[test][0], G[test][1]
        for oname, opt in OPTIONS.items():
            train = opt - {test}
            Ftr, ytr = stack(G, train)
            assert test not in train
            for mname, make in MODELS.items():
                m = make(seed).fit(Ftr, ytr)
                p = m.predict(Fte)
                r = letter_recall(yte, p)
                recalls[(rname, oname, mname)] = r
                rows.append({"round": rname, "option": oname, "model": mname,
                             "train": "+".join(sorted(train)), "train rows": len(ytr),
                             "test rows": len(yte),
                             "accuracy %": 100 * (p == yte).mean(),
                             "macro recall %": 100 * r.mean()})
                preds.append(pd.DataFrame({"round": rname, "option": oname, "model": mname,
                                           "letter": yte, "pred": p}))
                if verbose:
                    print(f"  {rname:21s} {oname:15s} {mname:10s} {rows[-1]['accuracy %']:5.1f}%")
    T = pd.DataFrame(rows)

    # paired comparisons on the same test rows: clean only vs the other two
    comp = []
    for rname in ROUNDS:
        for mname in MODELS:
            for other in ("clean + sheets", "sheets only"):
                a = T[(T["round"] == rname) & (T.model == mname)].set_index("option")
                ra, rb = recalls[(rname, "clean only", mname)], recalls[(rname, other, mname)]
                lo, hi = boot_diff(ra, rb)
                comp.append({"round": rname, "model": mname, "clean only vs": other,
                             "accuracy diff pts": a.loc["clean only", "accuracy %"] - a.loc[other, "accuracy %"],
                             "macro recall diff pts": 100 * (ra - rb).mean(),
                             "letter CI lo": 100 * lo, "letter CI hi": 100 * hi,
                             "letters better": int((ra > rb).sum()), "letters worse": int((ra < rb).sum())})
    Cmp = pd.DataFrame(comp)
    return T, Cmp, recalls, pd.concat(preds, ignore_index=True)


def decide(Cmp):
    """Apply the DECISION RULE (module docstring) to the comparison table."""
    verdict = {}
    for mname in MODELS:
        d = Cmp[(Cmp.model == mname) & (Cmp["clean only vs"] == "clean + sheets")]["accuracy diff pts"]
        if (d >= MARGIN).all():
            verdict[mname] = "remove the sheets"
        elif (d <= -MARGIN).all():
            verdict[mname] = "keep the sheets"
        else:
            verdict[mname] = "undecided"
    final = verdict["SVM"] if verdict["SVM"] == verdict["ExtraTrees"] else "undecided"
    return verdict, final


def main():
    os.makedirs(OUT, exist_ok=True)
    T, Cmp, recalls, P = run()
    verdict, final = decide(Cmp)
    pd.set_option("display.width", 200)
    print(T.round(1).to_string(index=False))
    print(Cmp.round(1).to_string(index=False))
    print("verdict per model:", verdict, "->", final)
    T.round(2).to_csv(os.path.join(OUT, "scores.csv"), index=False)
    Cmp.round(2).to_csv(os.path.join(OUT, "comparisons.csv"), index=False)
    R = pd.DataFrame({f"{r} | {o} | {m}": v for (r, o, m), v in recalls.items()}).round(3)
    R.to_csv(os.path.join(OUT, "per_letter_recall.csv"))
    # confusions only (letter, predicted letter, count): no row of Kamran's data is written
    conf = P[P.letter != P.pred].groupby(["round", "option", "model", "letter", "pred"]).size()
    conf.rename("count").reset_index().sort_values("count", ascending=False).to_csv(
        os.path.join(OUT, "confusions.csv"), index=False)
    with open(os.path.join(OUT, "verdict.txt"), "w", encoding="utf-8") as f:
        f.write(f"per model: {verdict}\nfinal: {final}\n")
    return T, Cmp, final


if __name__ == "__main__":
    main()
