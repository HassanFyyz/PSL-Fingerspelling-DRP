"""
Step 3, point 3: the clean test again, with the bad sheet crops dropped.

Same groups, rounds and models as clean_eval.py. New training options:

  clean + cleaned sheets    sheets minus the crops flag_sheets.py drops
  clean + random-dropped    sheets minus the SAME NUMBER of random crops (10 draws):
                            the control, so "cleaning helps" cannot just mean "fewer rows"

Which crops are dropped (fixed 23 Nov before any score below was computed):
  round A (test Kamran)       its own judge AND the other-signer check both flag the
                              crop (chosen because round A's judge alone would flag 54%
                              of correctly labelled images). The judge-alone rule
                              ("A, as first fixed") is reported too.
  round B (test calibration)  its own judge, as fixed.

DECISION RULE (fixed before running), shipped SVM and ExtraTrees must agree:
  * keep only the cleaned sheets  if clean + cleaned sheets >= clean only + 3 points in both rounds
  * remove the sheets entirely    if clean only >= clean + cleaned sheets + 3 points in both rounds
  * otherwise undecided (more people needed)
Reported alongside: does cleaning beat all sheets, and does it beat random dropping?

    PSL_KAMRAN_DIR=~/Desktop/train python cleaned_eval.py
"""
import os

import numpy as np
import pandas as pd

import clean_eval as ce
import flag_sheets

OUT = os.path.join("results", "2026-11-23_cleaned_test")
N_RANDOM = 10


def drop_masks(M):
    return {"A": (M.A_drop & M.other_flag).to_numpy(),          # two judges agree (primary)
            "A, as first fixed": M.A_drop.to_numpy(),
            "B": M.B_drop.to_numpy()}


def score(make, seed, Ftr, ytr, Fte, yte):
    p = make(seed).fit(Ftr, ytr).predict(Fte)
    return 100 * (p == yte).mean(), 100 * ce.letter_recall(yte, p).mean()


def run(seed=0):
    G = ce.groups()
    M = flag_sheets.flag(seed)
    masks = drop_masks(M)
    FS, yS = G["S"][0], G["S"][1]
    rows = []
    plan = [("A: test Kamran", "K", "C", "A"), ("A: test Kamran", "K", "C", "A, as first fixed"),
            ("B: test calibration", "C", "K", "B")]
    for rname, test, clean, mkey in plan:
        Fte, yte = G[test][0], G[test][1]
        Fc, yc = G[clean][0], G[clean][1]
        drop = masks[mkey]
        for mname, make in ce.MODELS.items():
            base = {"round": rname, "drop rule": mkey, "model": mname, "test rows": len(yte),
                    "sheet crops dropped": int(drop.sum())}
            opts = {"clean only": (Fc, yc),
                    "clean + all sheets": (np.vstack([Fc, FS]), np.concatenate([yc, yS])),
                    "clean + cleaned sheets": (np.vstack([Fc, FS[~drop]]), np.concatenate([yc, yS[~drop]]))}
            for oname, (Ftr, ytr) in opts.items():
                a, r = score(make, seed, Ftr, ytr, Fte, yte)
                rows.append({**base, "option": oname, "train rows": len(ytr),
                             "accuracy %": a, "macro recall %": r})
            rng = np.random.default_rng(seed)
            acc = []
            for _ in range(N_RANDOM):
                keep = np.ones(len(yS), bool)
                keep[rng.choice(len(yS), int(drop.sum()), replace=False)] = False
                acc.append(score(make, seed, np.vstack([Fc, FS[keep]]),
                                 np.concatenate([yc, yS[keep]]), Fte, yte))
            acc = np.array(acc)
            rows.append({**base, "option": "clean + random-dropped (mean of 10)",
                         "train rows": len(yc) + int((~drop).sum()),
                         "accuracy %": acc[:, 0].mean(), "macro recall %": acc[:, 1].mean(),
                         "random min %": acc[:, 0].min(), "random max %": acc[:, 0].max()})
            print(f"  {rname:21s} {mkey:18s} {mname:10s} done")
    return pd.DataFrame(rows)


def decide(T, margin=ce.MARGIN):
    primary = T[T["drop rule"].isin(["A", "B"])]
    acc = primary.pivot_table(index=["model", "round"], columns="option", values="accuracy %")
    verdict, checks = {}, {}
    for mname in ce.MODELS:
        a = acc.loc[mname]
        d = a["clean + cleaned sheets"] - a["clean only"]
        verdict[mname] = ("keep only the cleaned sheets" if (d >= margin).all() else
                          "remove the sheets" if (d <= -margin).all() else "undecided")
        checks[mname] = {
            "cleaned beats all sheets in both rounds":
                bool((a["clean + cleaned sheets"] > a["clean + all sheets"]).all()),
            "cleaned beats random dropping in both rounds":
                bool((a["clean + cleaned sheets"] > a["clean + random-dropped (mean of 10)"]).all())}
    final = verdict["SVM"] if verdict["SVM"] == verdict["ExtraTrees"] else "undecided"
    return verdict, checks, final, acc


def main():
    os.makedirs(OUT, exist_ok=True)
    T = run()
    verdict, checks, final, acc = decide(T)
    pd.set_option("display.width", 220)
    print(T.round(1).to_string(index=False))
    print(acc.round(1).to_string())
    print("verdict:", verdict, "->", final); print(checks)
    T.round(2).to_csv(os.path.join(OUT, "scores.csv"), index=False)
    with open(os.path.join(OUT, "verdict.txt"), "w", encoding="utf-8") as f:
        f.write(f"per model: {verdict}\nfinal: {final}\nchecks: {checks}\n")


if __name__ == "__main__":
    main()
