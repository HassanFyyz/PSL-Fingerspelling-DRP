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
    P, M = load_usable()   # only rows to train / test on (see IDLE HAND below)

IDLE HAND (found 23 Nov). The images are full-body shots: one hand signs at
chest or face height, the other hangs at the signer's side. The original cache
was made with a one-hand detector, which picked the idle hand in 1,348 of 2,902
usable images (46%): every one of them has the hand centre below y = 900 of
1080 with the fingers pointing down, and no image has its hand centre between
y = 700 and 900, so the cut at IDLE_Y is exact. `load_usable()` drops them.
`extract(two_hands=True)` re-detects up to two hands and keeps the higher one:
    PSL_KAMRAN_DIR=~/Desktop/train python image_dataset.py
Run on 23 Nov: hands in 2,952 of 2,952 images, 1,316 idle hands replaced by the
signing hand, 48 still idle (dropped), 2,863 usable.
"""
import hashlib
import os

import cv2
import numpy as np
import pandas as pd

# The folder can live outside the repo: set PSL_KAMRAN_DIR to its path, e.g.
#   PSL_KAMRAN_DIR=~/Desktop/train python clean_eval.py
ROOT = os.path.expanduser(os.environ.get("PSL_KAMRAN_DIR", "PSL_Fingerspelling_Image_Dataset"))
CACHE_PTS = os.path.join(ROOT, "landmarks.npy")
CACHE_META = os.path.join(ROOT, "landmarks_meta.csv")
LETTER_FIX = {"ك": "ک"}          # Arabic kaf -> Urdu kaf
IMAGE_EXT = (".jpg", ".jpeg", ".png")
IDLE_Y = 800            # hand centre lower than this (of 1080 px) = the idle hand


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


def _upper_hand(detector, img_bgr):
    """Like end_to_end.detect_pts, but with up to two hands keep the HIGHER one
    (smaller mean y): the signing hand, not the one hanging at the side."""
    import mediapipe as mp
    res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                   data=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)))
    if not res.hand_landmarks:
        return None, None
    h, w = img_bgr.shape[:2]
    hands = [np.array([[p.x * w, p.y * h, p.z * w] for p in lm], dtype=np.float32)
             for lm in res.hand_landmarks]
    k = int(np.argmin([pts[:, 1].mean() for pts in hands]))
    side = res.handedness[k][0].category_name if res.handedness else None
    return hands[k], side


def _detect(detector, img, two_hands=False):
    from end_to_end import detect_pts
    if two_hands:
        detect_pts = _upper_hand
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


def extract(root=ROOT, force=False, verbose=True, two_hands=False):
    """Detect landmarks for every image and write the cache. Returns (P, M).
    two_hands=True: detect up to two hands and keep the higher (signing) one."""
    if not force and os.path.exists(CACHE_PTS) and os.path.exists(CACHE_META):
        return load()
    from crop_dataset import make_detector
    det = make_detector()
    if two_hands:
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision
        from crop_dataset import MODEL_PATH, ensure_model
        det = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=mpp.BaseOptions(model_asset_path=ensure_model(MODEL_PATH)),
            running_mode=vision.RunningMode.IMAGE, num_hands=2,
            min_hand_detection_confidence=0.3))
    M = list_images(root)
    P = np.full((len(M), 21, 3), np.nan, dtype=np.float32)
    hands, how, sha = [], [], []
    for i, f in enumerate(M.file):
        img = _read(f)
        sha.append(hashlib.sha1(open(f, "rb").read()).hexdigest())
        pts, hand, mode = _detect(det, img, two_hands)
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


def idle_hand(P):
    """True where the detected hand is the idle one hanging at the side."""
    return np.nanmean(P[:, :, 1], axis=1) > IDLE_Y


def usable(P, M):
    """Rows to train and test on: hand found, not an exact duplicate file, not the idle hand."""
    det = (M.detected == 1).to_numpy()
    idle = np.zeros(len(M), bool)
    idle[det] = idle_hand(P[det])
    return det & ~M.duplicated("sha1").to_numpy() & ~idle


def load_usable(verbose=False):
    P, M = load()
    keep = usable(P, M)
    if verbose:
        det = (M.detected == 1).to_numpy()
        dup = M.duplicated("sha1").to_numpy()
        idle = np.zeros(len(M), bool)
        idle[det] = idle_hand(P[det])
        print(f"Kamran's images: {len(M)}; hand found {det.sum()}; duplicate files "
              f"{(det & dup).sum()}; idle hand {(det & ~dup & idle).sum()}; usable {keep.sum()}")
    return P[keep], M[keep].reset_index(drop=True)


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


if __name__ == "__main__":
    # Re-detect with two hands, keeping the signing one. The old cache is kept beside it.
    import shutil
    for f in (CACHE_PTS, CACHE_META):
        if os.path.exists(f):
            shutil.copy(f, f + ".one_hand_backup")
    P, M = extract(force=True, two_hands=True)
    det = (M.detected == 1).to_numpy()
    print(f"hands found {det.sum()} of {len(M)}; still in the idle zone: {int(idle_hand(P[det]).sum())}")
