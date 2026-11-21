"""
Variant of run.py: the 11 dynamic letters are pooled into ONE class, "*" ("a moving
letter, not identified"), instead of being removed. The model still never names a
moving letter; the lexicon writes every dynamic letter as "*", so the decoder scores
"there is a moving letter here" with real evidence rather than a blind slot.
Same clips, folds and settings as run.py.
"""
import json, os, subprocess, sys
import numpy as np, pandas as pd

sys.path.insert(0, ".")
import end_to_end as e2e, word_decode as wd, evaluation as ev
from splits import DYNAMIC_LETTERS

OUT = os.path.dirname(os.path.abspath(__file__))
DYN = set(DYNAMIC_LETTERS)
POOL = "*"

d = e2e._data()
d["y"] = np.array([POOL if l in DYN else l for l in d["y"]], dtype=object)   # relabel training rows
lex_pool = {w: [POOL if l in DYN else l for l in ls] for w, ls in d["lex"].items()}


def summarise(t, name):
    sg = t.signer.to_numpy()
    r1 = ev.summarise("", t.top1.to_numpy(), sg); r3 = ev.summarise("", t.top3.to_numpy(), sg)
    return {"setting": name, "clips": len(t), "top-1 %": r1["mean of folds %"],
            "CI lo": r1["boot lo %"], "CI hi": r1["boot hi %"], "top-3 %": r3["mean of folds %"]}


dynword = lambda w: any(l in DYN for l in d["lex"][w])
nested, _ = e2e.nested_clips(scores="proba", seed=0)
parts = []
for s, dd in nested.items():
    ins, dele = wd.tune_aligned(dd["tune"], lex_pool, "free", True)
    dec = lambda probs, classes, lx, clip, a=ins, b=dele: wd.decode_aligned(probs, classes, lx, clip["prior"], a, b)
    parts.append(e2e.word_table(dd["test"], lex_pool, dec, "free"))
t = pd.concat(parts, ignore_index=True); t["dynamic_word"] = [dynword(w) for w in t.word]
t.assign(model="pooled").to_csv(os.path.join(OUT, "offline_per_clip_pooled.csv"), index=False)
base = pd.read_csv(os.path.join(OUT, "offline_per_clip.csv"))
base40 = base[base.model == "40 letters (before)"].reset_index(drop=True)
rows = [summarise(t, "offline, 30 static + pooled moving class"),
        summarise(t[t.dynamic_word], "   words with a moving letter"),
        summarise(t[~t.dynamic_word], "   fully static words")]
print(pd.DataFrame(rows).round(1).to_string(index=False))
assert (base40.word == t.word).all()
r = ev.summarise("", t.top1.to_numpy(), t.signer.to_numpy(), baseline=base40.top1.to_numpy())
print(f"offline vs 40 letters: {r['vs baseline (pts)']:+.1f} pts, paired p = {r['paired t p']:.2f}")

clips, _ = e2e.loso_clips(scores="vote")
dump = {"all": [], "static": [{"signer": c["signer"], "word": c["word"], "classes": c["classes"],
                               "probs": np.asarray(c["free"]).tolist()} for c in clips],
        "lexicon": lex_pool, "dynamic_letters": []}
path = os.path.join(OUT, "vote_clips_pooled.json")
json.dump(dump, open(path, "w", encoding="utf-8"), ensure_ascii=False)
res = subprocess.run(["node", os.path.join(OUT, "decode_clips.mjs"), path], capture_output=True, text=True, check=True)
os.remove(path)
B = pd.DataFrame(json.loads(res.stdout)); B["dynamic_word"] = [dynword(w) for w in B.word]
B.assign(model="pooled").to_csv(os.path.join(OUT, "browser_per_clip_pooled.csv"), index=False)
rows = []
for sub, lab in ((B, "browser, 30 static + pooled moving class"), (B[B.dynamic_word], "   words with a moving letter"),
                 (B[~B.dynamic_word], "   fully static words")):
    r = summarise(sub, lab); sh = sub.shown.astype(bool)
    r["shown %"] = 100 * sh.mean(); r["shown right %"] = 100 * sub.top1[sh].mean()
    rows.append(r)
print(pd.DataFrame(rows).round(1).to_string(index=False))
pd.DataFrame(rows).round(1).to_csv(os.path.join(OUT, "browser_summary_pooled.csv"), index=False)
