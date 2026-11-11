"""Score a new calibration session three ways, without merging it.
  A  dataset signers only (no calibration at all)
  B  dataset + Hassan's earlier merged sessions (the shipped model's training data)
  C  B + this session's OTHER takes (leave one take out)
One prediction per take: mean probability over its frames. Seed fixed."""
import warnings; warnings.filterwarnings("ignore")
import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, ".")
import numpy as np, pandas as pd
import evaluation as ev

path = sys.argv[1]
d = json.load(open(path, encoding="utf-8"))
print(f"file: {path}\ncreated {d.get('created')}  fps {d.get('fps')}  hand {d.get('handedness')}  takes {len(d['takes'])}")
pts, lab, take = [], [], []
for t in d["takes"]:
    tid = f"{t['letter']}#{t['take']}"
    print(f"   {tid}: {len(t['frames'])} frames, hand {t.get('handedness')}")
    for fr in t["frames"]:
        pts.append(np.array(fr["pts"], dtype=np.float32)); lab.append(t["letter"]); take.append(tid)
Fu = ev.geometry_features(np.stack(pts)); yu = np.array(lab); tk = np.array(take)

P, M = ev.load_dataset(include_users=True); F = ev.geometry_features(P); y = M.letter.to_numpy()
user = M.signer.str.startswith("user-").to_numpy()
print(f"\ndataset signers: {(~user).sum()} holds; earlier calibration rows: {user.sum()} ({sorted(set(y[user]))})")

def vote(m, X):
    p = m.predict_proba(X).mean(0); o = np.argsort(p)[::-1]
    return m.classes_[o[0]], ", ".join(f"{m.classes_[i]} {100*p[i]:.0f}%" for i in o[:3])

mA = ev.svm(probability=True, seed=0).fit(F[~user], y[~user])
mB = ev.svm(probability=True, seed=0).fit(F, y)
res = []
for t in sorted(set(tk)):
    te = tk == t; truth = yu[te][0]
    a, ta = vote(mA, Fu[te]); b, tb = vote(mB, Fu[te])
    # one row per sibling take (median frame), as add_calibration.py would merge them
    sib = [np.where(tk == s)[0] for s in sorted(set(tk)) if s != t]
    Fs = np.stack([Fu[i[len(i) // 2]] for i in sib]); ys = np.array([yu[i[0]] for i in sib])
    mC = ev.svm(probability=True, seed=0).fit(np.vstack([F, Fs]), np.concatenate([y, ys]))
    c, tc = vote(mC, Fu[te])
    res.append((truth, t, a == truth, b == truth, c == truth, a, b, c, tb))
R = pd.DataFrame(res, columns=["letter", "take", "A", "B", "C", "A said", "B said", "C said", "B top3"])
print("\nper take (what each model said):")
print(R[["take", "A said", "B said", "C said", "B top3"]].to_string(index=False))
print("\ntakes correct, by letter:")
print(R.groupby("letter")[["A", "B", "C"]].agg(lambda s: f"{int(s.sum())}/{len(s)}").rename(columns={
    "A": "A dataset only", "B": "B + earlier sessions (shipped)", "C": "C + this session's other takes"}).to_string())
