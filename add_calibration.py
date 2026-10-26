"""
Add a recorded calibration session into the training data.

Usage:
    python add_calibration.py calibration/calibration_2026-10-22.json --signer user-hassan
    python add_calibration.py ... --frames-per-take 3      # mild augmentation
    python add_calibration.py ... --dry-run                # show what would change

Appends to `hold_landmarks.npy` / `hold_meta.csv`, which every experiment reads.

One row per take, by default
----------------------------
The dataset stores ONE row per letter occurrence. A recorded take holds ~37
frames of the same pose, and those frames are near-duplicates: the hand barely
moves during a take, which is the whole reason the segmenter can find it. Adding
all of them would count 10 real examples as 375 and make one person 22% of the
training data for two letters.

So by default each take contributes a single row, taken from the median frame of
that take. That matches the dataset's semantics and keeps leave-one-signer-out
meaningful.

If you want the user's examples to carry MORE weight than the dataset -- which
is a legitimate thing to want for personal calibration -- do it with explicit
sample weights at training time, not by duplicating near-identical rows. Padding
the row count inflates every accuracy estimate computed from this file.

Mislabel check
---------------
Before merging, each take is scored against a model trained on the CURRENT
dataset only (this take's own session is never in that model, so there is no
leakage). If that model's prediction disagrees with the take's stated letter,
the take is flagged: either the label is wrong, or that letter genuinely looks
like another one for this person, and either way it should not be merged
silently. This was not a hypothetical -- a deliberately mislabeled take (alif
recorded under a kaf label) was caught exactly this way: the base model called
it correctly while the label said otherwise. See calibration/README.md.

By default a flagged take blocks the write. Use --drop-suspicious to merge
everything else and skip just the flagged takes, or --force to merge
everything including the flagged ones (only if you are sure the label, not
the model, is right).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from hand_geometry import hand_features
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

PTS_PATH = "hold_landmarks.npy"
META_PATH = "hold_meta.csv"


def load_takes(path):
    d = json.load(open(path, encoding="utf-8"))
    out = []
    for t in d["takes"]:
        pts = np.array([f["pts"] for f in t["frames"]], dtype=np.float32)
        out.append({"letter": t["letter"], "take": t["take"], "pts": pts})
    return out, d


def pick_frames(pts, n):
    """n representative frames from a take, spread across it.

    The middle of a take is the most settled part -- the start and end include
    the hand arriving and leaving -- so a single pick takes the median frame.
    """
    if n <= 1:
        return pts[len(pts) // 2][None]
    lo, hi = int(0.2 * len(pts)), int(0.8 * len(pts))
    idx = np.unique(np.linspace(lo, max(lo + 1, hi - 1), n).round().astype(int))
    return pts[np.clip(idx, 0, len(pts) - 1)]


def check_takes(takes, P, M):
    """Score each take against a model trained on the current dataset only.

    Returns a dict take_key -> (claimed_letter, predicted_letter, top3) for
    every take whose claimed letter the base model disagrees with.
    """
    model = make_pipeline(StandardScaler(),
                          SVC(C=10, gamma="scale", class_weight="balanced", probability=True))
    model.fit(hand_features(P), M.letter.to_numpy())
    classes = model.named_steps["svc"].classes_

    flagged = {}
    for t in takes:
        feats = hand_features(t["pts"])
        probs = model.predict_proba(feats).mean(0)
        order = np.argsort(probs)[::-1]
        pred = classes[order[0]]
        if pred != t["letter"]:
            top3 = [(classes[i], probs[i]) for i in order[:3]]
            flagged[f"{t['letter']}#{t['take']}"] = (t["letter"], pred, top3)
    return flagged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--signer", default=None,
                    help="signer id for these rows (default: user-<date>)")
    ap.add_argument("--frames-per-take", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-check", action="store_true",
                    help="skip the mislabel check entirely")
    ap.add_argument("--drop-suspicious", action="store_true",
                    help="merge everything except flagged takes")
    ap.add_argument("--force", action="store_true",
                    help="merge flagged takes anyway, trusting the label over the model")
    args = ap.parse_args()

    takes, meta = load_takes(args.capture)
    signer = args.signer or f"user-{meta.get('created','')[:10]}"

    P = np.load(PTS_PATH)
    M = pd.read_csv(META_PATH)
    print(f"current dataset : {len(P)} holds, {M.signer.nunique()} signers, "
          f"{M.letter.nunique()} letters")

    if signer in set(M.signer):
        print(f"\n'{signer}' is already present ({int((M.signer==signer).sum())} rows).")
        print("Remove those rows first, or pass a different --signer.")
        return

    if not args.no_check:
        print("\nchecking takes against the current model (not trained on this session)...")
        flagged = check_takes(takes, P, M)
        if flagged:
            print(f"\n{len(flagged)} take(s) disagree with the model trained on everyone else:")
            for key, (claimed, pred, top3) in flagged.items():
                alts = ", ".join(f"{L} {100*p:.0f}%" for L, p in top3)
                print(f"   {key:<10} you labeled '{claimed}'  but the model says '{pred}'   [top3: {alts}]")
            if args.force:
                print("\n--force: merging flagged takes anyway, trusting the label.")
            elif args.drop_suspicious:
                takes = [t for t in takes if f"{t['letter']}#{t['take']}" not in flagged]
                print(f"\n--drop-suspicious: merging the other {len(takes)} take(s), skipping flagged ones.")
                if not takes:
                    print("nothing left to merge.")
                    return
            else:
                print("\nRefusing to merge. Either the label or the recording is wrong for")
                print("the take(s) above -- check it before trusting it into training data.")
                print("Re-run with --drop-suspicious to merge the rest, or --force to merge")
                print("everything anyway.")
                return
        else:
            print("all takes agree with the model. proceeding.")

    new_pts, rows = [], []
    for t in takes:
        for k, p in enumerate(pick_frames(t["pts"], args.frames_per_take)):
            new_pts.append(p)
            rows.append({"signer": signer,
                         "video": f"calib_{t['letter']}_{t['take']}",
                         "letter": t["letter"],
                         "occ": f"{signer}|calib_{t['letter']}_{t['take']}|{k}"})

    newM = pd.DataFrame(rows)
    print(f"adding          : {len(newM)} rows from {len(takes)} takes as '{signer}'")
    for L, n in newM.letter.value_counts().sort_index().items():
        have = int((M.letter == L).sum())
        print(f"   {L}: +{n:2d}   (dataset already has {have})")

    total = len(P) + len(newM)
    print(f"\nafter           : {total} holds, {M.signer.nunique()+1} signers")
    print(f"your share      : {100*len(newM)/total:.1f}% of all rows")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    for f in (PTS_PATH, META_PATH):
        if not os.path.exists(f + ".bak"):
            os.replace(f, f + ".bak") if False else None
    np.save(PTS_PATH + ".bak.npy", P)
    M.to_csv(META_PATH + ".bak.csv", index=False)

    np.save(PTS_PATH, np.concatenate([P, np.stack(new_pts)]))
    pd.concat([M, newM], ignore_index=True).to_csv(META_PATH, index=False)
    print(f"\nwrote {PTS_PATH} and {META_PATH}")
    print(f"backups: {PTS_PATH}.bak.npy, {META_PATH}.bak.csv")
    print("\nNote: leave-one-signer-out now has an extra fold holding only your "
          f"{newM.letter.nunique()} letters, which is not comparable to the "
          "40-letter folds. Filter it out when quoting the headline number.")


if __name__ == "__main__":
    main()
