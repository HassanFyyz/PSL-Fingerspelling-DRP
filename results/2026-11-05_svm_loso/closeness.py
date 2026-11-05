import sys, numpy as np, pandas as pd
from collections import Counter
REPO=sys.argv[1]; sys.path.insert(0,REPO)
from hand_geometry import hand_features
from sklearn.preprocessing import StandardScaler
P=np.load(f"{REPO}/hold_landmarks.npy"); M=pd.read_csv(f"{REPO}/hold_meta.csv")
k=~M.signer.str.startswith("user-").to_numpy(); P,M=P[k],M[k].reset_index(drop=True)
Z=StandardScaler().fit_transform(hand_features(P)); y=M.letter.to_numpy(); sg=M.signer.to_numpy()
cls=sorted(set(y))
# robust per-class "prototype": medoid-ish = median in standardized space
med={c:np.median(Z[y==c],0) for c in cls}
for c,o in [("م","س"),("ح","ر")]:
    for tag,mask in [("all",y==c),("without signer4-L",(y==c)&(sg!="signer4-L"))]:
        Zc=Z[mask]
        near=[min(cls,key=lambda k:np.linalg.norm(z-med[k])) for z in Zc]
        d_o=np.linalg.norm(Zc-med[o],axis=1); d_self=np.linalg.norm(Zc-med[c],axis=1)
        mc=np.median(Zc,0)
        rk=sorted(cls,key=lambda k:np.linalg.norm(mc-med[k]) if k!=c else -1)
        print(f"{c} {tag:>18} n={mask.sum():>2}: nearest class-median per sample {Counter(near).most_common(5)}; "
              f"closer to {o} than to own median: {np.mean(d_o<d_self):.2f}; {o} rank among other letters by median distance: {rk.index(o)}")
