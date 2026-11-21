"""
Generate a parity fixture for test_svm.mjs: real landmark frames (from the
dataset and both calibration sessions) plus the Python-computed features,
vote shares, and predicted letter, for the JS port to reproduce exactly.

Usage: python web/gen_svm_fixture.py > web/svm_fixture.json
       python web/gen_svm_fixture.py --with-kamran > web/model_geom_kamran/svm_fixture.json
       (the second trains exactly as `export_geometry_model.py --with-kamran`
       and adds 30 of Kamran's images to the cases; it stays git-ignored)
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, ".")
from hand_geometry import hand_features
from export_geometry_model import training_data

WITH_KAMRAN = "--with-kamran" in sys.argv
P = np.load("hold_landmarks.npy")
M = pd.read_csv("hold_meta.csv")
# same flags as export_geometry_model.py; the cases stay drawn from all our rows
F, y, g = training_data(with_kamran=WITH_KAMRAN, with_official="--with-official" in sys.argv,
                        with_video="--no-video" not in sys.argv)

scaler = StandardScaler().fit(F)
model = SVC(C=10, gamma="scale", class_weight="balanced")
model.fit(scaler.transform(F), y)
classes = model.classes_

rng = np.random.default_rng(42)
idx = rng.choice(len(P), size=60, replace=False)
# make sure both calibration signers' alif/kahf rows are included
cal = np.where(M.signer.str.startswith("user-hassan"))[0]
idx = np.unique(np.concatenate([idx, cal]))

pts = [P[i] for i in idx]
true = [M.letter[i] for i in idx]
if WITH_KAMRAN:
    import image_dataset
    PK, MK = image_dataset.load()
    k = rng.choice(np.where(MK.detected == 1)[0], size=30, replace=False)
    pts += [PK[i] for i in k]
    true += [MK.letter[i] for i in k]
preds = model.predict(scaler.transform(hand_features(np.array(pts))))

cases = []
for k in range(len(pts)):
    cases.append({
        "pts": np.asarray(pts[k]).tolist(),   # (21, 3)
        "letter": str(true[k]),
        "pred": str(preds[k]),
    })

print(json.dumps(cases, ensure_ascii=False))
