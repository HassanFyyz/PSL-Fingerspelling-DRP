"""
Landmarks at the holds the segmenter actually finds, for end-to-end evaluation.

Why this exists
---------------
`hold_landmarks.npy` holds one row per PSEUDO-LABELLED hold. Pseudo-labelling
is told the word, so it places exactly as many holds as the word has letters
(pseudo_label.py). Every word-accuracy figure computed from that file therefore
assumes the letter count is known. A live system does not know it: Algorithm 2
has to find the holds unaided, and it gets the count exactly right on about a
third of clips.

This module runs the unaided path on the same 218 clips:

    keypoint CSV -> sofi_dynamic.segment_clip (free local minima)
                 -> MediaPipe on the cropped video at each minimum
                 -> 21 landmarks in isotropic pixels

and caches the result in `freeseg_landmarks.npy` / `freeseg_meta.csv`, in the
same format as the pseudo-labelled cache so the same features and models apply.

Details that matter
-------------------
* Landmarks are `[x*w, y*h, z*w]`, the convention web/index.html and
  web/capture.html use. `verify_against_cache()` re-detects the pseudo-labelled
  hold frames with this code and compares with `hold_landmarks.npy`, because no
  script that produced that file's z column is in the repo.
* If MediaPipe finds no hand at the minimum frame, neighbouring frames inside
  the hold window are tried, nearest first. The live page averages every frame
  in a hold, so one missed frame does not lose the hold there either. A hold
  with no detection in the whole window is recorded with `detected = 0`.
* Left-hand detections are kept. The features are mirror-invariant and the live
  page classifies them too.
"""
import os

import cv2
import numpy as np
import pandas as pd

from crop_dataset import VIDEO_ROOT, make_detector
from sofi_dynamic import DEFAULT_SEG_LEN, segment_clip

CSV_PATH = "hand_keypoints_features.csv"
SEGMENTS_PATH = "pseudo_label_segments.csv"
OUT_PTS = "freeseg_landmarks.npy"
OUT_META = "freeseg_meta.csv"


def detect_pts(detector, img_bgr):
    """((21, 3) isotropic-pixel landmarks, handedness) or (None, None)."""
    import mediapipe as mp
    img = mp.Image(image_format=mp.ImageFormat.SRGB,
                   data=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    res = detector.detect(img)
    if not res.hand_landmarks:
        return None, None
    h, w = img_bgr.shape[:2]
    pts = np.array([[p.x * w, p.y * h, p.z * w] for p in res.hand_landmarks[0]], dtype=np.float32)
    hand = res.handedness[0][0].category_name if res.handedness else None
    return pts, hand


def read_frame(cap, frame):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
    got, img = cap.read()
    return img if got else None


def detect_near(detector, cap, frame, n_frames, reach=DEFAULT_SEG_LEN // 2):
    """Detect at `frame`, falling back to the nearest frame within `reach`."""
    for off in sorted(range(-reach, reach + 1), key=abs):
        f = frame + off
        if f < 0 or f >= n_frames:
            continue
        img = read_frame(cap, f)
        if img is None:
            continue
        pts, hand = detect_pts(detector, img)
        if pts is not None:
            return pts, hand, f
    return None, None, None


def load_clips(csv_path=CSV_PATH, segments_path=SEGMENTS_PATH):
    """{(signer, video): (sub, spelling)} for the clips that have pseudo-labels."""
    seg = pd.read_csv(segments_path)
    spell = {(r.signer_id, r.video_id): r.spelling
             for r in seg.drop_duplicates(["signer_id", "video_id"]).itertuples()}
    df = pd.read_csv(csv_path)
    out = {}
    for key, sub in df.groupby(["signer_id", "video_id"], sort=True):
        if key in spell:
            out[key] = (sub.sort_values("frame_id").reset_index(drop=True), spell[key])
    return out


def build(out_pts=OUT_PTS, out_meta=OUT_META, video_root=VIDEO_ROOT, verbose=True):
    """Run free segmentation + detection on every clip and cache the result."""
    clips = load_clips()
    detector = make_detector()
    pts_all, rows = [], []
    for (sg, vid), (sub, spelling) in clips.items():
        minima = segment_clip(sub)["minima"]
        cap = cv2.VideoCapture(os.path.join(video_root, sg, vid))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or len(sub)
        for k, pos in enumerate(minima):
            frame = int(sub.frame_id.iloc[pos])
            pts, hand, used = detect_near(detector, cap, frame, n_frames)
            rows.append({"signer": sg, "video": vid, "spelling": spelling, "hold": k,
                         "frame": frame, "frame_used": -1 if used is None else used,
                         "detected": int(pts is not None), "hand": hand or ""})
            pts_all.append(pts if pts is not None else np.full((21, 3), np.nan, np.float32))
        cap.release()
    P = np.stack(pts_all)
    M = pd.DataFrame(rows)
    np.save(out_pts, P)
    M.to_csv(out_meta, index=False, encoding="utf-8")
    if verbose:
        print(f"{len(clips)} clips, {len(M)} free holds, {int((M.detected == 0).sum())} with no hand "
              f"found, {int((M.hand == 'Left').sum())} read as a left hand")
        print(f"wrote {out_pts}, {out_meta}")
    return P, M


def verify_against_cache(n=150, seed=0, video_root=VIDEO_ROOT):
    """Re-detect pseudo-labelled hold frames and compare with hold_landmarks.npy.

    Returns a DataFrame with, per sampled hold, the largest coordinate difference
    in pixels for x/y and for z. Small numbers mean this module reproduces the
    cached file, including its z scaling.
    """
    P = np.load("hold_landmarks.npy")
    M = pd.read_csv("hold_meta.csv")
    seg = pd.read_csv(SEGMENTS_PATH)
    seg["occ"] = seg.signer_id + "|" + seg.video_id + "|" + seg.letter_index.astype(str)
    frame_of = dict(zip(seg.occ, seg.minimum_frame))
    idx = np.where(~M.signer.str.startswith("user-").to_numpy())[0]
    idx = np.random.default_rng(seed).choice(idx, size=min(n, len(idx)), replace=False)
    detector = make_detector()
    rows = []
    for i in idx:
        r = M.iloc[i]
        cap = cv2.VideoCapture(os.path.join(video_root, r.signer, r.video))
        img = read_frame(cap, frame_of[r.occ])
        cap.release()
        pts, _ = detect_pts(detector, img) if img is not None else (None, None)
        if pts is None:
            rows.append({"occ": r.occ, "xy_px": np.nan, "z_px": np.nan})
            continue
        d = np.abs(pts - P[i])
        rows.append({"occ": r.occ, "xy_px": float(d[:, :2].max()), "z_px": float(d[:, 2].max())})
    return pd.DataFrame(rows)


_DATA = {}


def _data(free_pts=OUT_PTS, free_meta=OUT_META):
    """Features and metadata for both hold sets, loaded once."""
    key = (free_pts, free_meta)
    if key not in _DATA:
        import json
        import evaluation as ev
        P, M = ev.load_dataset()
        F = ev.geometry_features(P)
        Pf, Mf = np.load(free_pts), pd.read_csv(free_meta)
        ok = Mf.detected.to_numpy() == 1
        Ff = np.zeros((len(Pf), F.shape[1]))
        Ff[ok] = ev.geometry_features(Pf[ok])
        seg = pd.read_csv(SEGMENTS_PATH).drop_duplicates(["signer_id", "video_id"])
        spell = {(r.signer_id, r.video_id): r.spelling for r in seg.itertuples()}
        lex = json.load(open("word_letter_mapping.json", encoding="utf-8"))
        _DATA[key] = dict(F=F, y=M.letter.to_numpy(), sg=M.signer.to_numpy(), vd=M.video.to_numpy(),
                          Ff=Ff, fsg=Mf.signer.to_numpy(), fvd=Mf.video.to_numpy(), fok=ok,
                          spell=spell, lex=lex)
    return _DATA[key]


def clips_for(train_signers, test_signer, scores="vote", seed=0, make_model=None):
    """Clips of `test_signer`, scored by a model fitted on `train_signers` only.

    Each clip carries two score sequences from that one model:
      "oracle"  the pseudo-labelled holds (letter count known)
      "free"    the holds Algorithm 2 found unaided
    and `prior`, the letter frequencies of the training signers.
    """
    import evaluation as ev
    from word_decode import vote_share

    d = _data()
    tr = np.isin(d["sg"], list(train_signers))
    if make_model is not None:
        model = make_model(seed).fit(d["F"][tr], d["y"][tr])
        score = model.predict_proba
    else:
        model = ev.svm(probability=(scores == "proba"), seed=seed).fit(d["F"][tr], d["y"][tr])
        score = model.predict_proba if scores == "proba" else (lambda X: vote_share(model, X))
    classes = list(model.classes_)
    prior = pd.Series(d["y"][tr]).value_counts(normalize=True).to_dict()
    clips = []
    for v in sorted(set(d["vd"][d["sg"] == test_signer])):
        orc = np.where((d["sg"] == test_signer) & (d["vd"] == v))[0]
        fr = np.where((d["fsg"] == test_signer) & (d["fvd"] == v) & d["fok"])[0]
        word = d["spell"][(test_signer, v)]
        clips.append({"signer": test_signer, "video": v, "word": word,
                      "n_true": len(d["lex"][word]), "classes": classes, "prior": prior,
                      "oracle": score(d["F"][orc]),
                      "free": score(d["Ff"][fr]) if len(fr) else np.zeros((0, len(classes)))})
    return clips


def signers():
    return sorted(set(_data()["sg"]))


def loso_clips(scores="vote", seed=0, make_model=None):
    """Every clip, each scored by a model that never saw its signer.

    `scores` is "vote" (one-vs-one vote share, what the browser computes;
    deterministic) or "proba" (Platt probabilities; depends on `seed`).
    `make_model(seed)` may return any estimator with predict_proba; it then
    replaces the SVM and `scores` is ignored. Returns (clips, lexicon).
    """
    S = signers()
    clips = []
    for s in S:
        clips += clips_for([t for t in S if t != s], s, scores, seed, make_model)
    return clips, _data()["lex"]


def nested_clips(scores="proba", seed=0, make_model=None):
    """For tuning a decoder without touching the held-out signer.

    Returns ({signer: {"test": clips, "tune": clips}}, lexicon). "test" is the
    held-out signer scored by the model trained on the other four. "tune" is
    those four signers, each scored by a model trained on the remaining three
    -- so decoder settings can be chosen on clips the letter model had not
    seen, and that the test signer never touches.
    """
    S = signers()
    out = {}
    for s in S:
        train = [t for t in S if t != s]
        tune = []
        for h in train:
            tune += clips_for([t for t in train if t != h], h, scores, seed, make_model)
        out[s] = {"test": clips_for(train, s, scores, seed, make_model), "tune": tune}
    return out, _data()["lex"]


def word_table(clips, lex, decoder, which="free"):
    """DataFrame per clip: top1, top3 under `decoder`.

    `decoder(probs, classes, lex)`, or `decoder(probs, classes, lex, clip)` if
    it has a parameter named `clip` (for decoders that need the training-signer
    prior carried on the clip).
    """
    import inspect
    wants_clip = "clip" in inspect.signature(decoder).parameters
    rows = []
    for c in clips:
        ranked = (decoder(c[which], c["classes"], lex, c) if wants_clip
                  else decoder(c[which], c["classes"], lex))
        rows.append({"signer": c["signer"], "video": c["video"], "word": c["word"],
                     "n_true": c["n_true"], "n_holds": len(c[which]),
                     "top1": float(ranked[:1] == [c["word"]]), "top3": float(c["word"] in ranked[:3])})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    build()
