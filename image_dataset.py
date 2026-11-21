"""
Kamran's PSL letter-image dataset: hand landmarks for every image, cached.

The dataset (PSL_Fingerspelling_Image_Dataset/, from the group's Drive) is NOT
in git and must stay private: `.gitignore` covers the folder, and the landmark
cache this module writes lives inside it, so it is covered too.

Layout: <root>/{train,val}/<letter>/<image>.jpg, full 1920x1080 green-screen
frames, one hand shape per image. Folder names are letters, except kaf, which
is spelled with the Arabic kaf U+0643 and is mapped to the Urdu kaf U+06A9 used
everywhere else in this repo.

Landmarks are detected exactly as for the video holds (end_to_end.detect_pts,
MediaPipe image mode, [x*w, y*h, z*w]), so hand_geometry features and the
existing models apply unchanged. If no hand is found in the full frame, the
centre square is tried, because the hand is small in a full-HD frame.

    P, M = load()          # P (n, 21, 3); M: split, letter, file, detected, sha1
"""
import hashlib
import os

import cv2
import numpy as np
import pandas as pd

ROOT = "PSL_Fingerspelling_Image_Dataset"
CACHE_PTS = os.path.join(ROOT, "landmarks.npy")
CACHE_META = os.path.join(ROOT, "landmarks_meta.csv")
LETTER_FIX = {"ك": "ک"}          # Arabic kaf -> Urdu kaf
IMAGE_EXT = (".jpg", ".jpeg", ".png")


def list_images(root=ROOT):
    rows = []
    for split in ("train", "val"):
        base = os.path.join(root, split)
        for folder in sorted(os.listdir(base)):
            d = os.path.join(base, folder)
            if not os.path.isdir(d):
                continue                            # e.g. "Training statistics.doc"
            for f in sorted(os.listdir(d)):
                if f.lower().endswith(IMAGE_EXT):
                    rows.append((split, LETTER_FIX.get(folder, folder), os.path.join(d, f)))
    return pd.DataFrame(rows, columns=["split", "letter", "file"])


def _read(path):
    # cv2.imread cannot open non-ASCII (Urdu) paths on Windows
    return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)


def _detect(detector, img):
    from end_to_end import detect_pts
    pts, hand = detect_pts(detector, img)
    if pts is not None:
        return pts, hand, "full"
    h, w = img.shape[:2]
    x0 = max(0, (w - h) // 2)
    pts, hand = detect_pts(detector, img[:, x0:x0 + h])
    if pts is not None:
        pts = pts.copy()
        pts[:, 0] += x0
        return pts, hand, "centre"
    return None, None, "none"


def extract(root=ROOT, force=False, verbose=True):
    """Detect landmarks for every image and write the cache. Returns (P, M)."""
    if not force and os.path.exists(CACHE_PTS) and os.path.exists(CACHE_META):
        return load()
    from crop_dataset import make_detector
    det = make_detector()
    M = list_images(root)
    P = np.full((len(M), 21, 3), np.nan, dtype=np.float32)
    hands, how, sha = [], [], []
    for i, f in enumerate(M.file):
        img = _read(f)
        sha.append(hashlib.sha1(open(f, "rb").read()).hexdigest())
        pts, hand, mode = _detect(det, img)
        if pts is not None:
            P[i] = pts
        hands.append(hand)
        how.append(mode)
        if verbose and (i + 1) % 500 == 0:
            print(f"{i + 1}/{len(M)}")
    M["hand"], M["found_in"], M["sha1"] = hands, how, sha
    M["detected"] = (M.found_in != "none").astype(int)
    np.save(CACHE_PTS, P)
    M.to_csv(CACHE_META, index=False, encoding="utf-8")
    return P, M


def load():
    return np.load(CACHE_PTS), pd.read_csv(CACHE_META, encoding="utf-8")


# ─── official PSL reference images ───────────────────────────────────────────
# Screenshots from the PSL website in letter_sheets/<code>_<letter>/official/.
# Someone else's material: git-ignored, local only, like Kamran's set.
OFFICIAL_GLOB = os.path.join("letter_sheets", "*", "official", "*")


def official():
    """(P, M) for the official reference images in which a hand is found."""
    import glob
    from crop_dataset import make_detector
    det = make_detector()
    rows, pts = [], []
    for f in sorted(glob.glob(OFFICIAL_GLOB)):
        if not f.lower().endswith(IMAGE_EXT):
            continue
        letter = os.path.basename(os.path.dirname(os.path.dirname(f))).split("_", 1)[1]
        p, hand, mode = _detect(det, _read(f))
        rows.append((LETTER_FIX.get(letter, letter), f, hand, mode, int(p is not None)))
        pts.append(p if p is not None else np.full((21, 3), np.nan, np.float32))
    M = pd.DataFrame(rows, columns=["letter", "file", "hand", "found_in", "detected"])
    keep = (M.detected == 1).to_numpy()
    return np.array(pts)[keep], M[keep].reset_index(drop=True)
