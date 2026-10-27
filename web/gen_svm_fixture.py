"""
Generate a parity fixture for test_svm.mjs: real landmark frames (from the
dataset and both calibration sessions) plus the Python-computed features,
vote shares, and predicted letter, for the JS port to reproduce exactly.

Usage: python web/gen_svm_fixture.py > web/svm_fixture.json
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, ".")
from hand_geometry import hand_features

P = np.load("hold_landmarks.npy")
M = pd.read_csv("hold_meta.csv")
F = hand_features(P)
y = M.letter.to_numpy()

scaler = StandardScaler().fit(F)
model = SVC(C=10, gamma="scale", class_weight="balanced")
model.fit(scaler.transform(F), y)
classes = model.classes_

rng = np.random.default_rng(42)
idx = rng.choice(len(P), size=60, replace=False)
# make sure both calibration signers' alif/kahf rows are included
cal = np.where(M.signer.str.startswith("user-hassan"))[0]
idx = np.unique(np.concatenate([idx, cal]))

preds = model.predict(scaler.transform(F[idx]))

cases = []
for k, i in enumerate(idx):
    cases.append({
        "pts": P[i].tolist(),          # (21, 3)
        "letter": str(y[i]),
        "pred": str(preds[k]),
    })

print(json.dumps(cases, ensure_ascii=False))
