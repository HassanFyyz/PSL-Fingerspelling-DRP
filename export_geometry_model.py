"""
Train the hand-geometry + RBF-SVM letter classifier and export it for the
browser, replacing the HOG + logistic-regression model export_model.py built.

Why switch: on leave-one-signer-out, angles+distances+SVM-RBF scores 26.5%
against HOG+LogReg's 12.9% (hand_geometry.py). The live demo has been shipping
the weaker model the whole time; this brings the actual best model to the page
people click.

Porting an RBF-kernel multiclass SVM to JS is more than a matrix multiply, so
unlike export_model.py this cannot just fold a scaler into weights. Instead it
exports sklearn's raw SVC internals (support vectors, dual coefficients,
per-pair intercepts, per-class support counts) and the browser replays
libsvm's own one-vs-one decision procedure: for every class pair, a signed
margin from a kernel sum; the class with the most pairwise wins, well above
chance, is 2.5 %, under ties the lowest class index wins.

That decode was NOT assumed -- it was checked line for line against
model.predict() on every training row plus 500 synthetic points before any JS
was written (0 mismatches). The margin-sum tiebreak that seemed natural was
checked too and does NOT match sklearn; plain vote-count argmax does. See
web/test_svm.mjs for the parity harness that re-checks this on every run.

No probability=True / Platt scaling here. That would need porting libsvm's
pairwise coupling algorithm too, for calibrated probabilities this project
does not strictly need: the browser only needs an argmax and a usable
top-3/lexicon-scoring signal, and the fraction of pairwise wins per class
serves that without the extra machinery.
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from hand_geometry import hand_features
from splits import DYNAMIC_LETTERS

PTS_PATH = "hold_landmarks.npy"
META_PATH = "hold_meta.csv"
MAPPING_PATH = "word_letter_mapping.json"
OUT_DIR = "web/model_geom"
# Trained with Kamran's private letter-image set as well. The SVM stores
# training rows (support vectors), so this export carries features of his
# images and must never be committed or published: the folder is git-ignored
# and the page only uses it when it exists locally.
OUT_DIR_KAMRAN = "web/model_geom_kamran"


def training_data(pts_path=PTS_PATH, meta_path=META_PATH, with_kamran=False,
                  with_official=False, with_video=True, static_only=True):
    """(F, y, signers): our holds + calibration takes, plus Kamran's / official images if asked.

    with_video=False drops the pseudo-labelled video holds (signers without the
    'user-' prefix) and keeps only the calibration takes from our own files.
    Kamran's rows: hand found, exact duplicate files dropped, signer 'kamran-images'.
    Official rows: the PSL website screenshots, signer 'official-psl'.
    static_only=True (the default since 22 Nov) drops every row of the 11 dynamic
    letters (splits.DYNAMIC_LETTERS) from every source, so the model has the 30
    static letters as its classes. `--all-letters` restores the old 41-class model.
    """
    F, y, g = _all_rows(pts_path, meta_path, with_kamran, with_official, with_video)
    if static_only:
        keep = ~np.isin(y, DYNAMIC_LETTERS)
        F, y, g = F[keep], y[keep], g[keep]
    return F, y, g


def _all_rows(pts_path, meta_path, with_kamran, with_official, with_video):
    P = np.load(pts_path)
    M = pd.read_csv(meta_path)
    if not with_video:
        keep = M.signer.str.startswith("user-").to_numpy()
        P, M = P[keep], M[keep].reset_index(drop=True)
    F, y, g = hand_features(P), M.letter.to_numpy(), M.signer.to_numpy()
    if with_official:
        import image_dataset
        PO, MO = image_dataset.official()
        F = np.vstack([F, hand_features(PO)])
        y = np.concatenate([y, MO.letter.to_numpy()])
        g = np.concatenate([g, np.array(["official-psl"] * len(MO), dtype=object)])
    if with_kamran:
        import image_dataset
        PK, MK = image_dataset.load()
        keep = ((MK.detected == 1) & ~MK.duplicated("sha1")).to_numpy()
        F = np.vstack([F, hand_features(PK[keep])])
        y = np.concatenate([y, MK.letter.to_numpy()[keep]])
        g = np.concatenate([g, np.array(["kamran-images"] * int(keep.sum()), dtype=object)])
    return F, y, g


def loso_accuracy(F, y, signers, exclude_single_letter_folds=True, extra=None):
    """Leave-one-signer-out, filtering folds that hold too few letters to be
    comparable (a calibration signer with only 2 letters is not a 40-class
    test)."""
    accs = []
    for s in sorted(set(signers)):
        te = signers == s
        tr = ~te
        if exclude_single_letter_folds and len(set(y[te])) < 10:
            continue
        Ftr, ytr = F[tr], y[tr]
        if extra is not None:                  # rows that are always training (never a fold)
            Ftr, ytr = np.vstack([Ftr, extra[0]]), np.concatenate([ytr, extra[1]])
        sc = StandardScaler().fit(Ftr)
        clf = SVC(C=10, gamma="scale", class_weight="balanced")
        clf.fit(sc.transform(Ftr), ytr)
        acc = (clf.predict(sc.transform(F[te])) == y[te]).mean()
        accs.append(acc)
    return float(np.mean(accs))


def video_test_accuracy(F, y, Fv, yv, sv):
    """Train once on (F, y), score each video signer as an outside test; mean of signers."""
    sc = StandardScaler().fit(F)
    clf = SVC(C=10, gamma="scale", class_weight="balanced").fit(sc.transform(F), y)
    pred = clf.predict(sc.transform(Fv))
    return float(np.mean([(pred[sv == s] == yv[sv == s]).mean() for s in sorted(set(sv))]))


def build(pts_path=PTS_PATH, meta_path=META_PATH, mapping_path=MAPPING_PATH, out_dir=None,
          with_kamran=False, with_official=False, with_video=True, static_only=True):
    private = with_kamran or with_official
    out_dir = out_dir or (OUT_DIR_KAMRAN if private else OUT_DIR)
    os.makedirs(out_dir, exist_ok=True)
    F, y, signers = training_data(pts_path, meta_path, with_kamran, with_official, with_video,
                                  static_only)
    print(f"{len(F)} training rows, {len(set(y))} letters"
          + (" (static only)" if static_only else " (all letters)"))

    print("computing leave-one-signer-out accuracy (folds with at least 10 letters)...")
    if with_video:
        own = ~np.isin(signers, ["kamran-images", "official-psl"])
        loso = loso_accuracy(F[own], y[own], signers[own],
                             extra=(F[~own], y[~own]) if private else None)
    else:
        # no video hold is in training, so the five video signers are an outside test
        Fv, yv, sv = training_data(pts_path, meta_path, static_only=static_only)
        v = ~pd.Series(sv).str.startswith("user-").to_numpy()
        loso = video_test_accuracy(F, y, Fv[v], yv[v], sv[v])
    print(f"  LOSO: {loso:.3f}")
    # majority baseline under the same protocol: the most common training letter, scored per
    # dataset signer and averaged like `loso`
    vid = ~pd.Series(signers).str.startswith("user-").to_numpy() & \
        ~np.isin(signers, ["kamran-images", "official-psl"])
    common = pd.Series(y).value_counts().index[0]
    majority = float(np.mean([(y[signers == s] == common).mean() for s in sorted(set(signers[vid]))])) \
        if vid.any() else None

    scaler = StandardScaler().fit(F)
    Fs = scaler.transform(F)
    model = SVC(C=10, gamma="scale", class_weight="balanced")
    model.fit(Fs, y)

    classes = [str(c) for c in model.classes_]
    sv = model.support_vectors_.astype(np.float32)
    dual_coef = model.dual_coef_.astype(np.float32)
    intercept = model.intercept_.astype(np.float32)
    n_support = model.n_support_.astype(np.int32)
    gamma = float(model._gamma)

    sv.tofile(os.path.join(out_dir, "geom_sv.bin"))
    dual_coef.tofile(os.path.join(out_dir, "geom_dualcoef.bin"))
    intercept.tofile(os.path.join(out_dir, "geom_intercept.bin"))
    scaler.mean_.astype(np.float32).tofile(os.path.join(out_dir, "geom_scaler_mean.bin"))
    scaler.scale_.astype(np.float32).tofile(os.path.join(out_dir, "geom_scaler_scale.bin"))

    mapping = json.load(open(mapping_path, encoding="utf-8"))
    manifest = {
        "kind": "handgeom+svm-rbf-ovo",
        "n_feat": int(sv.shape[1]),
        "n_sv": int(sv.shape[0]),
        "gamma": gamma,
        "classes": classes,
        "n_classes": len(classes),
        "n_support": n_support.tolist(),
        "n_pairs": int(intercept.shape[0]),
        "train_holds": int(len(F)),
        "train_signers": sorted(set(signers)),
        "handedness": "right",
        "private": bool(private),
        "training_sources": {"video_holds": bool(with_video), "calibration": True,
                             "kamran_images": bool(with_kamran), "official_psl": bool(with_official)},
        "static_only": bool(static_only),
        "dynamic_letters": list(DYNAMIC_LETTERS) if static_only else [],
        "honest_accuracy": {
            "kamran_images": ("Kamran's images are in the training rows of every fold and are never "
                              "a test fold; the test signer is always one of ours.") if with_kamran else None,
            "no_video": ("No video hold is in training; letter_acc is the mean over the five video "
                         "signers used as an outside test, scored against their pseudo-labels.")
                        if not with_video else None,
            "protocol": "leave-one-signer-out; a signer is a fold only if it holds at least "
                        "10 letters (the five dataset signers and the 13 Nov calibration "
                        "session); the smaller calibration sessions are always training. "
                        "letter_majority is the most common letter scored on the dataset "
                        "signers. The shipped model trains on everyone."
                        + (" Static letters only: the 11 dynamic letters are removed from "
                           "training and from the test holds." if static_only else ""),
            "letter_acc": round(loso, 3),
            "letter_majority": None if majority is None else round(majority, 3),
            "note": ("The lexicon keeps all 54 words. A dynamic letter in a word is a slot "
                     "the decoder fills with any hold or skips (word_decode.js); the model "
                     "never outputs it." if static_only else
                     "Same 54-word lexicon; every letter is a class."),
        },
        "lexicon": {w: list(ls) for w, ls in mapping.items()},
    }
    with open(os.path.join(out_dir, "geom_model.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)

    total_kb = (sv.nbytes + dual_coef.nbytes + intercept.nbytes) / 1024
    print(f"\nwrote {out_dir}/geom_model.json")
    print(f"support vectors: {sv.shape[0]} x {sv.shape[1]} ({sv.nbytes/1024:.0f} KB)")
    print(f"dual_coef: {dual_coef.shape} ({dual_coef.nbytes/1024:.0f} KB)")
    print(f"total weight size: {total_kb:.0f} KB")
    return manifest, model, scaler


if __name__ == "__main__":
    import sys
    build(with_kamran="--with-kamran" in sys.argv, with_official="--with-official" in sys.argv,
          with_video="--no-video" not in sys.argv, static_only="--all-letters" not in sys.argv)
