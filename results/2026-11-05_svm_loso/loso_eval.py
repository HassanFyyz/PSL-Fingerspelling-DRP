"""Leave-one-signer-out per-letter recall + confusion for the shipped geometry+SVM
recipe (export_geometry_model.py: StandardScaler, SVC C=10 gamma=scale
class_weight=balanced), on the 5 dataset signers only. Read-only on the repo.

usage: python3 loso_eval.py <repo> <outdir>
"""
import sys, os
from collections import Counter
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

REPO, OUT = sys.argv[1], sys.argv[2]
sys.path.insert(0, REPO)
from hand_geometry import hand_features, feature_names

P = np.load(os.path.join(REPO, "hold_landmarks.npy"))
M = pd.read_csv(os.path.join(REPO, "hold_meta.csv"))
keep = ~M.signer.str.startswith("user-").to_numpy()
P, M = P[keep], M[keep].reset_index(drop=True)
F = hand_features(P)
y = M.letter.to_numpy(); sg = M.signer.to_numpy()
classes = sorted(set(y))
print(f"holds {len(y)}  signers {sorted(set(sg))}  letters {len(classes)}")

pred = np.empty(len(y), dtype=object)
for s in sorted(set(sg)):
    te, tr = sg == s, sg != s
    sc = StandardScaler().fit(F[tr])
    clf = SVC(C=10, gamma="scale", class_weight="balanced").fit(sc.transform(F[tr]), y[tr])
    pred[te] = clf.predict(sc.transform(F[te]))
    print(f"  fold {s:<10} n={te.sum():>3}  letters={len(set(y[te])):>2}  acc={np.mean(pred[te]==y[te]):.3f}")
fold_acc = [np.mean(pred[sg == s] == y[sg == s]) for s in sorted(set(sg))]
print(f"pooled acc {np.mean(pred==y):.3f}   mean of folds {np.mean(fold_acc):.3f}   "
      f"majority {Counter(y).most_common(1)[0][1]/len(y):.3f}")

cm = pd.crosstab(pd.Series(y, name="true"), pd.Series(pred, name="pred")).reindex(
    index=classes, columns=classes, fill_value=0)
cm.to_csv(os.path.join(OUT, "confusion_loso_5signers.csv"))

rows = []
for c in classes:
    m = y == c
    conf = Counter(pred[m & (pred != c)]).most_common(3)
    rows.append(dict(letter=c, n=int(m.sum()), signers=len(set(sg[m])),
                     per_signer=" ".join(f"{s[6]}:{(sg[m]==s).sum()}" for s in sorted(set(sg))),
                     recall=np.mean(pred[m] == c),
                     predicted_as_this=int((pred == c).sum()),
                     precision=(np.mean(y[pred == c] == c) if (pred == c).any() else np.nan),
                     confusions=", ".join(f"{k}×{v}" for k, v in conf)))
T = pd.DataFrame(rows).sort_values(["recall", "n"])
T.to_csv(os.path.join(OUT, "per_letter_loso_5signers.csv"), index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 100)
print("\nper-letter (sorted by recall):")
print(T.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

# ---- live letters
LIVE = ["گ", "ا", "و", "ل", "س", "ک", "ر", "ث", "ت", "م", "ح"]
print("\nlive letters, offline:")
print(T.set_index("letter").loc[[l for l in LIVE if l in T.letter.values]]
      .to_string(float_format=lambda v: f"{v:.2f}"))

# ---- geometry closeness (features standardised on all 5 signers)
Z = StandardScaler().fit_transform(F)
cent = {c: Z[y == c].mean(0) for c in classes}
def spread(c):  # mean distance of class samples to own centroid
    return np.linalg.norm(Z[y == c] - cent[c], axis=1).mean()
spreads = {c: spread(c) for c in classes}
print(f"\nwithin-class spread (mean dist to own centroid): median over letters "
      f"{np.median(list(spreads.values())):.2f}, range {min(spreads.values()):.2f}-{max(spreads.values()):.2f}")
D = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=2)
np.fill_diagonal(D, np.inf)
names = feature_names()
for c, other in [("م", "س"), ("ح", "ر")]:
    m = y == c
    cd = sorted(((np.linalg.norm(cent[c] - cent[k]), k) for k in classes if k != c))
    rank = [k for _, k in cd].index(other) + 1
    print(f"\n=== {c}  n={m.sum()}  per signer: " +
          " ".join(f"{s}:{(sg[m]==s).sum()}" for s in sorted(set(sg))))
    print(f"  spread {spreads[c]:.2f} (rank {sorted(spreads.values(), reverse=True).index(spreads[c])+1}/40 most spread; "
          f"{other} spread {spreads[other]:.2f})")
    print(f"  nearest centroids: " + ", ".join(f"{k} {d:.2f}" for d, k in cd[:6]) +
          f"   -> {other} is #{rank} at {dict((k,d) for d,k in cd)[other]:.2f}")
    print(f"  median centroid-to-centroid distance overall: "
          f"{np.median([np.linalg.norm(cent[a]-cent[b]) for i,a in enumerate(classes) for b in classes[i+1:]]):.2f}")
    # per-signer centroids of c: are signers consistent with each other?
    sc_ = {s: Z[m & (sg == s)].mean(0) for s in sorted(set(sg[m]))}
    ss = list(sc_)
    print("  between-signer centroid distances for this letter: " +
          ", ".join(f"{a[6]}-{b[6]} {np.linalg.norm(sc_[a]-sc_[b]):.2f}" for i, a in enumerate(ss) for b in ss[i+1:]))
    # nearest neighbour (other signer only, to mimic cross-signer) label of each sample
    idx = np.where(m)[0]
    nn = []
    for i in idx:
        d = D[i].copy(); d[sg == sg[i]] = np.inf
        nn.append(y[np.argmin(d)])
    print(f"  cross-signer 1-NN labels: {Counter(nn).most_common(6)}")
    # within-signer 1-NN
    nnw = []
    for i in idx:
        d = D[i].copy(); d[sg != sg[i]] = np.inf
        nnw.append(y[np.argmin(d)])
    print(f"  same-signer 1-NN labels:  {Counter(nnw).most_common(6)}")
    diff = cent[c] - cent[other]
    top = np.argsort(-np.abs(diff))[:6]
    print(f"  biggest standardised centroid differences {c} vs {other}: " +
          ", ".join(f"{names[j]} {diff[j]:+.2f}" for j in top))
    # same for live signer-free: what does the LOSO SVM call these?
    print(f"  LOSO predictions for {c}: {Counter(pred[m]).most_common(6)}")
    print(f"  per-signer LOSO recall: " + " ".join(
        f"{s[6]}:{np.mean(pred[m & (sg==s)]==c):.2f}" for s in sorted(set(sg[m]))))
    # which videos / words are the c examples from
    print(f"  words: {Counter(M.video[m].str.replace('.MP4','')).most_common(8)}")
