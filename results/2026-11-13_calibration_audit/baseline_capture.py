"""Score a new capture before merging, using the browser's mean OVO vote shares."""
import sys,json,collections,time
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
repo=Path(sys.argv[1]);source=Path(sys.argv[2]);out=Path(sys.argv[3]);out.mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(repo))
from hand_geometry import hand_features

def fit(P,y):
 return make_pipeline(StandardScaler(),SVC(C=10,gamma='scale',class_weight='balanced',decision_function_shape='ovo')).fit(hand_features(P),y)

def vote_share(m,F):
 svc=m.named_steps['svc'];n=len(svc.classes_);shares=[]
 for k in range(0,len(F),512):
  z=m.decision_function(F[k:k+512]);v=np.zeros((len(z),n));p=0
  for i in range(n):
   for j in range(i+1,n):
    yes=z[:,p]>0;v[:,i]+=yes;v[:,j]+=~yes;p+=1
  shares.append(v/(n-1))
 return np.concatenate(shares)

def main():
 d=json.loads(source.read_text());training=out/'training_before' if (out/'training_before').is_dir() else repo
 P=np.load(training/'hold_landmarks.npy');M=pd.read_csv(training/'hold_meta.csv');data=~M.signer.str.startswith('user-').to_numpy()
 U=np.concatenate([np.array([f['pts'] for f in t['frames']],dtype=np.float32) for t in d['takes']]);F=hand_features(U)
 start=0;bounds=[]
 for t in d['takes']:bounds.append((start,start+len(t['frames'])));start+=len(t['frames'])
 results={};rows=[]
 for name,sel in [('dataset_only',data),('current_recipe_refit',np.ones(len(M),dtype=bool))]:
  print('Fitting',name,'rows',sel.sum(),flush=True);m=fit(P[sel],M.letter.to_numpy()[sel]);cls=m.classes_;s=vote_share(m,F)
  if name=='current_recipe_refit':
   model_dir=out/'model_before' if (out/'model_before').is_dir() else repo/'web/model_geom'
   man=json.load(open(model_dir/'geom_model.json'))
   assert m.named_steps['svc'].support_vectors_.shape[0]==man['n_sv']
   assert list(cls)==man['classes']
  pred=[]
  for t,(a,b) in zip(d['takes'],bounds):
   v=s[a:b].mean(0);p=int(np.argmax(v));rank=sorted(range(len(v)),key=lambda i:(-v[i],i));truth=t['letter'];pred.append(str(cls[p]));rows.append({'model':name,'letter':truth,'take':t['take'],'known_class':truth in set(cls),'n_frames':b-a,'handedness':t.get('handedness'),'pred':str(cls[p]),'correct':bool(cls[p]==truth),'top3': ' | '.join(str(cls[i]) for i in rank[:3]),'max_vote_share':float(v[p])})
  print(name,'done',flush=True)
 R=pd.DataFrame(rows);R.to_csv(out/'baseline_per_take.csv',index=False)
 T=R.groupby(['letter','model'],sort=False).agg(n=('correct','size'),correct=('correct','sum'),recall=('correct','mean'),known_class=('known_class','first'))
 T.to_csv(out/'baseline_per_letter.csv');print(T.to_string())
 print('known-class totals',R[R.known_class].groupby('model').correct.agg(['sum','size','mean']).to_string())
 print('No new capture rows entered either training model. New class hamza separately flagged; no signer identity assigned.')
if __name__=='__main__': main()
