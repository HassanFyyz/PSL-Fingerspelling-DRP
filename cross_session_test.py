"""
The open test from calibration/README.md: does calibration done on one session
(22 Oct) transfer to a different session (26 Oct) -- a different day, different
light, different seat -- without ever training on that second session?

Usage:
    python cross_session_test.py calibration/calibration_2026-10-26.json
    python cross_session_test.py <test_session.json> --calibrated-on user-hassan

`--calibrated-on` names the signer id of the EARLIER session in hold_meta.csv
(default user-hassan, the 22 Oct session). Every other `user-*` row is left out
of both models. That matters: the 26 Oct session was later merged into the
training data as `user-hassan-session2`, and without this filter its own takes
sat inside the "no calibration" model and the script printed 100/100/100.

Three numbers, all scored on the SAME held-out 26 Oct takes:

  NO CALIBRATION   trained on the 5 dataset signers only. What the live demo
                    does today with no personalisation at all.

  CROSS-SESSION     trained on the dataset PLUS the 22 Oct session (already
                    merged into hold_landmarks.npy / hold_meta.csv as
                    user-hassan). This is the real test: calibration learned
                    on one day, scored on a session it has never seen.

  SAME-SESSION      trained on the dataset + 22 Oct + the 26 Oct take's OWN
                    sibling takes (leave-one-take-out within 26 Oct). Upper
                    bound -- what calibrate.py already reports for a single
                    session, shown here for comparison.
"""
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
    P = model.predict_proba(F).mean(0)
    cls = model.named_steps["svc"].classes_
    order = np.argsort(P)[::-1]
    return cls[order[0]], [(cls[i], P[i]) for i in order[:3]]


def main(path, calibrated_on="user-hassan"):
    Md = pd.read_csv(DATASET_META)
    Pd = np.load(DATASET_PTS)
    Fd_all = hand_features(Pd)

    signer = Md.signer.to_numpy()
    letters = Md.letter.to_numpy()
    is_user = Md.signer.str.startswith("user-").to_numpy()
    is_old_session = signer == calibrated_on
    if not is_old_session.any():
        sys.exit(f"no rows for signer '{calibrated_on}' in {DATASET_META}")
    # base: the five dataset signers only. cal: those plus the ONE earlier
    # session. No other user-* rows in either -- they may be the test session.
    in_cal = ~is_user | is_old_session
    Fd_base, yd_base = Fd_all[~is_user], letters[~is_user]
    Fd_cal, yd_cal = Fd_all[in_cal], letters[in_cal]
    left_out = sorted(set(signer[is_user & ~is_old_session]))
    if left_out:
        print(f"left out of both models: {', '.join(left_out)}")

    Pu, yu, takes, meta = load_user(path)
    Fu = hand_features(Pu)
    user_letters = sorted(set(yu))

    print(f"dataset (5 signers) : {len(Fd_base)} holds")
    print(f"+ earlier session   : {is_old_session.sum()} holds ({calibrated_on})")
    print(f"26 Oct session      : {len(Fu)} frames, {len(set(takes))} takes, letters {' '.join(user_letters)}")
    print(f"                      recorded at {meta.get('fps','?')} fps")

    n_classes = len(set(yd_base) | set(yu))
    print(f"\nchoosing among {n_classes} letters -- chance is {100/n_classes:.1f}%\n")

    model_base = fit(Fd_base, yd_base)
    model_cal = fit(Fd_cal, yd_cal)

    no_cal, cross, same = {}, {}, {}
    for t in sorted(set(takes)):
        te = takes == t
        tr = ~te

        pred, top3 = take_vote(model_base, Fu[te])
        no_cal[t] = (yu[te][0], pred, top3)

        pred, top3 = take_vote(model_cal, Fu[te])
        cross[t] = (yu[te][0], pred, top3)

        F = np.vstack([Fd_cal, Fu[tr]])
        y = np.concatenate([yd_cal, yu[tr]])
        model_same = fit(F, y)
        pred, top3 = take_vote(model_same, Fu[te])
        same[t] = (yu[te][0], pred, top3)

    def summarise(res, tag):
        ok = sum(1 for truth, pred, _ in res.values() if truth == pred)
        print(f"{tag}: {ok}/{len(res)} takes correct ({100*ok/len(res):.0f}%)")
        for t in sorted(res):
            truth, pred, top3 = res[t]
            mark = "OK " if truth == pred else "   "
            alts = ", ".join(f"{L} {100*p:.0f}%" for L, p in top3)
            print(f"  {mark} {t:<10} you signed {truth}  ->  model said {pred}      [top3: {alts}]")
        print()
        return ok / len(res)

    a = summarise(no_cal, "NO CALIBRATION  (dataset signers only)")
    b = summarise(cross, "CROSS-SESSION   (dataset + 22 Oct, scored on 26 Oct)")
    c = summarise(same, "SAME-SESSION    (dataset + 22 Oct + 26 Oct siblings)")

    print("=" * 66)
    print(f"  no calibration : {100*a:.0f}%   cross-session : {100*b:.0f}%   same-session : {100*c:.0f}%")
    print("=" * 66)

    per = defaultdict(lambda: [0, 0])
    for truth, pred, _ in cross.values():
        per[truth][0] += truth == pred
        per[truth][1] += 1
    print("\ncross-session, per letter:")
    for L in sorted(per):
        cnt, n = per[L]
        print(f"   {L}: {cnt}/{n} takes correct")

    print("\n--- what this means ---")
    if b > a:
        print(f"  Calibration from 22 Oct transferred to a new session: {100*a:.0f}% -> {100*b:.0f}%.")
        print("  That is the number worth quoting -- it was never trained on 26 Oct data.")
    elif b == a:
        print(f"  No change from calibration on this new session ({100*a:.0f}% both ways).")
    else:
        print(f"  Calibration from 22 Oct did worse on 26 Oct ({100*a:.0f}% -> {100*b:.0f}%).")
        print("  Possible overfitting to the first session's exact pose/lighting.")
    if c > b:
        print(f"  Same-session calibration ({100*c:.0f}%) beats cross-session ({100*b:.0f}%),")
        print("  as expected -- some of the earlier gain was session-specific.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    on = "user-hassan"
    if "--calibrated-on" in sys.argv:
        on = sys.argv[sys.argv.index("--calibrated-on") + 1]
    main(sys.argv[1], on)
