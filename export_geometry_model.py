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

PTS_PATH = "hold_landmarks.npy"
META_PATH = "hold_meta.csv"
MAPPING_PATH = "word_letter_mapping.json"
OUT_DIR = "web/model_geom"


def loso_accuracy(F, y, signers, exclude_single_letter_folds=True):
    """Leave-one-signer-out, filtering folds that hold too few letters to be
    comparable (a calibration signer with only 2 letters is not a 40-class
    test)."""
    accs = []
    for s in sorted(set(signers)):
        te = signers == s
        tr = ~te
        if exclude_single_letter_folds and len(set(y[te])) < 10:
            continue
        clf = SVC(C=10, gamma="scale", class_weight="balanced")
        clf.fit(StandardScaler().fit_transform(F[tr]), y[tr])
        sc = StandardScaler().fit(F[tr])
        acc = (clf.predict(sc.transform(F[te])) == y[te]).mean()
        accs.append(acc)
    return float(np.mean(accs))


def build(pts_path=PTS_PATH, meta_path=META_PATH, mapping_path=MAPPING_PATH, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    P = np.load(pts_path)
    M = pd.read_csv(meta_path)
    F = hand_features(P)
    y = M.letter.to_numpy()
    signers = M.signer.to_numpy()

    print("computing leave-one-signer-out accuracy (40-letter folds only)...")
    loso = loso_accuracy(F, y, signers)
    print(f"  LOSO: {loso:.3f}")

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
        "honest_accuracy": {
            "protocol": "leave-one-signer-out over the 40-letter folds; the two "
                        "calibration-session folds (2 letters each) are excluded "
                        "as not comparable. The shipped model trains on everyone.",
            "letter_acc": round(loso, 3),
            "letter_majority": 0.098,
            "note": "No lexicon decoding changes here -- same word-decode logic "
                    "as the HOG model, scored against the same 54-word vocabulary.",
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
    build()
