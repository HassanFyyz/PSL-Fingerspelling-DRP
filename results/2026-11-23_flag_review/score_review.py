"""
Score the blind review of the sheet-crop flags.

    python results/2026-11-23_flag_review/score_review.py flag_review_answers.csv

For each item the reviewer picked the left letter, the right letter, neither, or "can't tell".
Using review_key.csv (which side was the label, and whether the round-B judge dropped the crop):

  reviewer says the LABEL is wrong  = picked the other letter, or neither
  flag precision                    = share of DROPPED crops the reviewer says are wrong
  kept-crop error rate              = share of KEPT crops the reviewer says are wrong
                                      (wrong labels the flags missed)
"can't tell" answers are counted separately and left out of both rates.

What would confirm the cleaning: flag precision well above the kept-crop error rate. The
judge's flags then pick out wrong or unusable crops, independently of any model score.
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def score(answers_csv):
    K = pd.read_csv(os.path.join(HERE, "review_key.csv"))
    A = pd.read_csv(answers_csv, encoding="utf-8-sig", dtype={"item": int})
    D = K.merge(A[["item", "answer"]], on="item", how="left")
    D["answer"] = D.answer.fillna("")
    D["says_label"] = D.answer == D.label_side
    D["says_wrong"] = D.answer.isin(["left", "right", "neither"]) & ~D.says_label
    D["says_other_letter"] = D.answer.isin(["left", "right"]) & ~D.says_label
    D["unsure"] = D.answer == "unsure"
    D["blank"] = D.answer == ""
    rows = []
    for flagged, name in ((True, "dropped by the judge"), (False, "kept by the judge")):
        d = D[D.flagged == flagged]
        judged = d[~d.unsure & ~d.blank]
        rows.append({"crops": name, "items": len(d), "answered": int((~d.blank).sum()),
                     "can't tell": int(d.unsure.sum()),
                     "reviewer: label right": int(judged.says_label.sum()),
                     "reviewer: label wrong": int(judged.says_wrong.sum()),
                     "  of which the judge's letter": int(judged.says_other_letter.sum()),
                     "wrong %": round(100 * judged.says_wrong.mean(), 1) if len(judged) else None})
    S = pd.DataFrame(rows)
    D.to_csv(os.path.join(HERE, "review_scored.csv"), index=False)
    S.to_csv(os.path.join(HERE, "review_summary.csv"), index=False)
    print(S.to_string(index=False))
    return S


if __name__ == "__main__":
    score(sys.argv[1])
