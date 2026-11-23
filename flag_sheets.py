"""
Step 3, point 1: flag letter-sheet crops whose label looks wrong.

Each sheet crop (a video hold, static letters only) is judged three ways.
Thresholds were fixed on 23 Nov before any score was looked at:

  clean      a model trained on CLEAN data only (ExtraTrees, seed 0) does not
             have the crop's label in its top 3
  neighbour  the previous or next letter of the crop's word is in that model's
             top 3 with at least twice the label's probability: the timing slip
             seen in the letter sheets (a م crop that shows the ل before it)
  other      a model trained on the other four sheet signers does not have the
             label in its top 3 (the check in letter_models.CleanedLabels, at
             top 3). Reported for comparison; it does not decide drops.

A crop is DROPPED if it fails `clean` or `neighbour`.

No leakage into the clean test (clean_eval.py): the clean judge never uses the
group that round tests.
  round A (tests Kamran's images)   judge trained on our calibration takes
  round B (tests our calibration)   judge trained on Kamran's images

    PSL_KAMRAN_DIR=~/Desktop/train python flag_sheets.py
writes results/2026-11-23_sheet_flags/ (sheet crops are public data; nothing
of Kamran's is written).
"""
import json
import os

import numpy as np
import pandas as pd

import clean_eval as ce
import models
from canonicalize_words import canonicalize, spelling_for_labels
from hand_geometry import hand_features
from splits import DYNAMIC_LETTERS

OUT = os.path.join("results", "2026-11-23_sheet_flags")
TOP = 3                 # label must be within the judge's top TOP
NEIGHBOUR_RATIO = 2.0   # neighbour flagged if p(neighbour) >= this x p(label)
JUDGE = {"A": "C", "B": "K"}   # round -> clean group the judge is trained on


def sheet_rows():
    """Static-letter sheet crops with word, position and neighbours, in the same
    order as clean_eval.groups()['S']."""
    P = np.load("hold_landmarks.npy")
    M = pd.read_csv("hold_meta.csv")
    keep = (~M.signer.str.startswith("user-") & ~M.letter.isin(DYNAMIC_LETTERS)).to_numpy()
    P, M = P[keep], M[keep].reset_index(drop=True)
    W = json.load(open("word_letter_mapping.json", encoding="utf-8"))
    rows = []
    for v, occ, letter in zip(M.video, M.occ, M.letter):
        sp = spelling_for_labels(v.rsplit(".", 1)[0])
        ls = W.get(sp) or W[canonicalize(sp)]
        pos = int(occ.rsplit("|", 1)[1])
        assert ls[pos] == letter, (v, pos, letter)
        rows.append((sp, pos, ls[pos - 1] if pos > 0 else "", ls[pos + 1] if pos + 1 < len(ls) else ""))
    M[["word", "pos", "prev", "next"]] = pd.DataFrame(rows)
    return hand_features(P), M


def judge(model, F, M, prefix):
    """Run one judge over the crops; add its rank / flags as columns."""
    pr = model.predict_proba(F)
    col = {c: i for i, c in enumerate(model.classes_)}
    order = np.argsort(-pr, axis=1)
    rank = np.empty_like(order); rank[np.arange(len(pr))[:, None], order] = np.arange(pr.shape[1])
    out = {f"{prefix}_rank": [], f"{prefix}_top1": [], f"{prefix}_p_label": []}
    nb_flag, nb_letter = [], []
    for k, (letter, prev, nxt) in enumerate(zip(M.letter, M.prev, M.next)):
        c = col.get(letter)
        out[f"{prefix}_rank"].append(int(rank[k, c]) + 1 if c is not None else 99)
        out[f"{prefix}_top1"].append(model.classes_[order[k, 0]])
        pl = pr[k, c] if c is not None else 0.0
        out[f"{prefix}_p_label"].append(round(float(pl), 3))
        best = ""
        for nb in (prev, nxt):
            j = col.get(nb)
            if nb and nb != letter and j is not None and rank[k, j] < TOP \
                    and pr[k, j] >= NEIGHBOUR_RATIO * max(pl, 1e-9):
                if not best or pr[k, j] > pr[k, col[best]]:
                    best = nb
        nb_flag.append(bool(best)); nb_letter.append(best)
    for k, v in out.items():
        M[k] = v
    M[f"{prefix}_flag_clean"] = M[f"{prefix}_rank"] > TOP
    if prefix != "other":
        M[f"{prefix}_flag_neighbour"] = nb_flag
        M[f"{prefix}_neighbour_letter"] = nb_letter
        M[f"{prefix}_drop"] = M[f"{prefix}_flag_clean"] | M[f"{prefix}_flag_neighbour"]
    return M


def flag(seed=0):
    F, M = sheet_rows()
    G = ce.groups()
    assert len(G["S"][1]) == len(M) and (G["S"][1] == M.letter.to_numpy()).all()
    for rnd, group in JUDGE.items():
        m = models.extra_trees(seed).fit(G[group][0], G[group][1])
        M = judge(m, F, M, rnd)
    # other signers: each sheet signer judged by the other four (sheet labels only)
    parts = []
    for s in sorted(M.signer.unique()):
        a, b = (M.signer != s).to_numpy(), (M.signer == s).to_numpy()
        m = models.extra_trees(seed).fit(F[a], M.letter.to_numpy()[a])
        parts.append(judge(m, F[b], M[b].copy(), "other"))
    O = pd.concat(parts).sort_index()
    for c in ("other_rank", "other_top1", "other_p_label", "other_flag_clean"):
        M[c] = O[c]
    M = M.rename(columns={"other_flag_clean": "other_flag"})
    return M


def summary(M):
    by = lambda key: M.groupby(key).agg(
        crops=("letter", "size"),
        A_drop=("A_drop", "sum"), B_drop=("B_drop", "sum"),
        both_drop=("both", "sum"), other_flag=("other_flag", "sum"))
    M = M.assign(both=M.A_drop & M.B_drop)
    sig = by("signer"); let = by("letter")
    for t in (sig, let):
        t["A_drop %"] = (100 * t.A_drop / t.crops).round(1)
        t["B_drop %"] = (100 * t.B_drop / t.crops).round(1)
    agree = {
        "crops": len(M),
        "A: dropped": int(M.A_drop.sum()), "A: by clean check": int(M.A_flag_clean.sum()),
        "A: by neighbour check only": int((M.A_flag_neighbour & ~M.A_flag_clean).sum()),
        "B: dropped": int(M.B_drop.sum()), "B: by clean check": int(M.B_flag_clean.sum()),
        "B: by neighbour check only": int((M.B_flag_neighbour & ~M.B_flag_clean).sum()),
        "dropped in both rounds": int(M.both.sum()),
        "other-signer flags": int(M.other_flag.sum()),
        "other-signer flags also dropped in A": int((M.other_flag & M.A_drop).sum()),
        "other-signer flags also dropped in B": int((M.other_flag & M.B_drop).sum()),
    }
    return sig, let, agree


def judge_reliability(seed=0):
    """DIAGNOSTIC ONLY (thresholds were fixed before this was computed): how often the
    clean check would flag a CORRECTLY labelled example, measured on the other clean
    group. Uses the test group of each round, so it must never be used to set thresholds."""
    G = ce.groups()
    out = {}
    for rnd, group in JUDGE.items():
        other = "K" if group == "C" else "C"
        m = models.extra_trees(seed).fit(G[group][0], G[group][1])
        pr = m.predict_proba(G[other][0]); col = {c: i for i, c in enumerate(m.classes_)}
        y = G[other][1]
        r = np.array([(pr[k] > pr[k, col[l]]).sum() for k, l in enumerate(y)])
        out[f"round {rnd}: judge on {group}, correct {other} labels outside top {TOP}"] = \
            round(100 * float((r >= TOP).mean()), 1)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    M = flag()
    sig, let, agree = summary(M)
    rel = judge_reliability()
    pd.set_option("display.width", 200)
    print(pd.Series(agree).to_string()); print(sig.to_string()); print(rel)
    cols = ["signer", "video", "word", "pos", "prev", "letter", "next",
            "A_rank", "A_top1", "A_p_label", "A_flag_clean", "A_flag_neighbour", "A_neighbour_letter", "A_drop",
            "B_rank", "B_top1", "B_p_label", "B_flag_clean", "B_flag_neighbour", "B_neighbour_letter", "B_drop",
            "other_rank", "other_top1", "other_flag"]
    M[cols].to_csv(os.path.join(OUT, "flags.csv"), index=False)
    sig.to_csv(os.path.join(OUT, "by_signer.csv")); let.to_csv(os.path.join(OUT, "by_letter.csv"))
    pd.Series({**agree, **rel}).to_csv(os.path.join(OUT, "summary.csv"), header=["value"])
    return M


if __name__ == "__main__":
    main()
