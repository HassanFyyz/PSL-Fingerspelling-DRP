"""
Train the deployment letter classifier and export it for the browser.

The model is HOG + multinomial logistic regression -- the best representation
measured in letter_cnn.ipynb (12.9 % leave-one-signer-out against a 9.8 %
majority baseline, lift 1.31x). It is deliberately simple enough to run in
JavaScript as a matrix multiply, with no inference runtime to ship.

IMPORTANT about the accuracy figure
-----------------------------------
The exported model is trained on ALL five signers, because for a demo you want
every scrap of data. Its *honest* accuracy estimate is still the
leave-one-signer-out 12.9 %, NOT anything measured on data it trained on. For a
new person in front of a webcam, LOSO is the applicable number -- and 12.9 % on
40 classes means the per-letter output will usually be wrong.

That is why the app also decodes against a lexicon: individual letters are
unreliable, but the nearest word in a 54-word vocabulary can still be right
often enough to be worth showing.
"""
import json
import os

import numpy as np

HOG_ORIENTATIONS = 9
HOG_PPC = 8
HOG_CPB = 2
EPS = 1e-5

OUT_DIR = "web/model"


def ref_hog(img, orientations=HOG_ORIENTATIONS, ppc=HOG_PPC, cpb=HOG_CPB, eps=EPS):
    """Pure-numpy HOG, verified to match skimage.feature.hog to ~1e-7.

    Kept here rather than calling skimage so that this file alone defines the
    feature the browser must reproduce; web/hog.js is a direct port of it.
    """
    im = img.astype(np.float64)
    R, C = im.shape
    g_row = np.zeros_like(im)
    g_col = np.zeros_like(im)
    g_row[1:-1, :] = im[2:, :] - im[:-2, :]
    g_col[:, 1:-1] = im[:, 2:] - im[:, :-2]
    mag = np.hypot(g_row, g_col)
    ori = np.rad2deg(np.arctan2(g_row, g_col)) % 180.0

    nr, nc = R // ppc, C // ppc
    hist = np.zeros((nr, nc, orientations))
    step = 180.0 / orientations
    for k in range(orientations):
        sel = (ori >= step * k) & (ori < step * (k + 1))
        m = np.where(sel, mag, 0.0)
        hist[:, :, k] = m[:nr * ppc, :nc * ppc].reshape(nr, ppc, nc, ppc).sum(axis=(1, 3)) / (ppc * ppc)

    bR, bC = nr - cpb + 1, nc - cpb + 1
    out = np.zeros((bR, bC, cpb, cpb, orientations))
    for i in range(bR):
        for j in range(bC):
            b = hist[i:i + cpb, j:j + cpb, :]
            v = b / np.sqrt(np.sum(b ** 2) + eps ** 2)
            v = np.minimum(v, 0.2)
            v = v / np.sqrt(np.sum(v ** 2) + eps ** 2)
            out[i, j] = v
    return out.ravel()


def build(crops_path="letter_crops.npz", mapping_path="word_letter_mapping.json",
          out_dir=OUT_DIR, hog_cache="letter_crops_hog.npy"):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    os.makedirs(out_dir, exist_ok=True)
    d = np.load(crops_path, allow_pickle=True)
    X, y = d["images"], d["letters"]

    if os.path.exists(hog_cache):
        H = np.load(hog_cache)
    else:
        H = np.stack([ref_hog(im) for im in X]).astype(np.float32)
        np.save(hog_cache, H)

    scaler = StandardScaler(with_mean=False).fit(H)
    clf = LogisticRegression(max_iter=1500, C=0.05, class_weight="balanced", n_jobs=-1)
    clf.fit(scaler.transform(H), y)

    classes = [str(c) for c in clf.classes_]
    # fold the scaler into the weights: p = softmax(W @ (h / scale) + b)
    W = (clf.coef_ / scaler.scale_[None, :]).astype(np.float32)   # (n_classes, n_feat)
    b = clf.intercept_.astype(np.float32)

    W.tofile(os.path.join(out_dir, "letter_w.bin"))
    b.tofile(os.path.join(out_dir, "letter_b.bin"))

    mapping = json.load(open(mapping_path, encoding="utf-8"))
    manifest = {
        "kind": "hog+logreg",
        "img_size": int(X.shape[1]),
        "hog": {"orientations": HOG_ORIENTATIONS, "pixels_per_cell": HOG_PPC,
                "cells_per_block": HOG_CPB, "eps": EPS, "n_features": int(W.shape[1])},
        "classes": classes,
        "n_classes": len(classes),
        "train_occurrences": int(len(set(d["occurrences"]))),
        "honest_accuracy": {"loso_acc": 0.129, "majority": 0.098, "lift": 1.31,
                            "note": "leave-one-signer-out; the model shipped here is "
                                    "trained on all signers, so do not quote its "
                                    "training accuracy"},
        "lexicon": {w: list(ls) for w, ls in mapping.items()},
    }
    with open(os.path.join(out_dir, "letter_model.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)

    print(f"features {W.shape[1]}, classes {len(classes)}")
    print(f"wrote {out_dir}/letter_model.json, letter_w.bin ({W.nbytes/1024:.0f} KB), letter_b.bin")
    print(f"lexicon: {len(mapping)} words")
    return manifest


if __name__ == "__main__":
    build()
