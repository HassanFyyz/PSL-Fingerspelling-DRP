"""Re-creates streaming_segmentation.ipynb cells 2, 7 and 20 (fixture export)
without running the notebook. Run from the repo root. usage: gen_parity_fixture.py <out.json>"""
import json, sys, os
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd
from canonicalize_words import spelling_for_labels
from sofi_dynamic import KP_COLS_X, KP_COLS_Y
from sofi_streaming import StreamingSegmenter, TUNED_FPS

df = pd.read_csv("hand_keypoints_features.csv")
mapping = json.load(open("word_letter_mapping.json", encoding="utf-8"))
df["spelling"] = df["video_id"].map({v: spelling_for_labels(v) for v in df["video_id"].unique()})
clips = []
for (sg, vid), sub in df.groupby(["signer_id", "video_id"], sort=True):
    sub = sub.sort_values("frame_id").reset_index(drop=True)
    sp = sub["spelling"].iloc[0]
    if sp not in mapping or sub["hand_detected"].mean() < 0.5 or len(sub) < 30:
        continue
    clips.append({"signer": sg, "video_id": vid, "sub": sub})
print(f"{df[['signer_id','video_id']].drop_duplicates().shape[0]} clips in CSV, {len(clips)} validation clips")

def run_stream(sub, fps, **kw):
    X = sub[KP_COLS_X].to_numpy(float); Y = sub[KP_COLS_Y].to_numpy(float)
    det = sub["hand_detected"].to_numpy()
    seg = StreamingSegmenter(fps=fps, **kw); out = []
    for i in range(len(sub)):
        events = seg.push(None, None) if det[i] == 0 else seg.push(X[i], Y[i])
        out += [h.frame for h in events]
    out += [h.frame for h in seg.finish()]
    return out, seg

cases = []
for i, c in enumerate(clips):
    if i % 9:
        continue
    sub = c["sub"]
    X = sub[KP_COLS_X].to_numpy(float); Y = sub[KP_COLS_Y].to_numpy(float)
    det = sub["hand_detected"].to_numpy()
    frames = [None if det[j] == 0 else [list(map(float, X[j])), list(map(float, Y[j]))] for j in range(len(sub))]
    for fps in (50, 30):
        out, seg = run_stream(sub, fps)
        cases.append({"signer": c["signer"], "video": c["video_id"], "fps": fps,
                      "frames": frames, "expected": [int(v) for v in out],
                      "config": {"smooth": seg.smooth_win, "run": seg.run,
                                 "min_sep": seg.min_sep, "delay": seg.delay}})
json.dump(cases, open(sys.argv[1], "w"), separators=(",", ":"))
print(f"{len(cases)} cases exported, {sum(len(c['expected']) for c in cases)} expected holds")
