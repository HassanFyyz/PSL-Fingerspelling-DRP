"""
Build an image dataset of hand crops at the pseudo-labelled hold frames.

This is the input to the appearance-based letter classifier. The motivation is
in letter_classifier.ipynb: landmark geometry predicts *signer identity* at 97 %
while predicting letters below the majority baseline, and letter_reference.ipynb
confirmed by eye that the labels themselves are sound. So the information that
distinguishes PSL letters -- finger configuration and contact -- is present in
the pixels and absent from the 21 landmark coordinates.

A trap worth knowing about
--------------------------
The keypoints in `hand_keypoints_features.csv` are NOT in the coordinate space
of the 240x350 cropped videos. Overlaying them puts the hand in the bottom-right
corner, on the signer's jacket, with one landmark off the bottom edge entirely.
The hand aspect ratio implied by those coordinates is 0.23, where a real hand is
0.45-0.65; assuming a 16:9 source gives 0.60. They were evidently extracted from
the original uncropped recordings.

Nothing else in the project depends on that mapping -- Algorithm 2 uses only
frame-to-frame *displacement*, the pseudo-labels are frame indices, the letter
features are wrist-relative, and the contact sheets render whole frames. But
cropping by those boxes would produce a dataset of jacket lapels.

So this module re-runs MediaPipe directly on the cropped videos. The hand is
near-stationary during a hold, so it detects once at the hold's minimum frame
and reuses that box for the surrounding frames: ~1.4k detections instead of
~14.5k.

Grayscale, deliberately
-----------------------
Colour carries skin tone and clothing, which are signer identity -- the exact
confound that sank the landmark features. Feeding RGB invites the model to
relearn it. Shape is what distinguishes letters, so the crops are grayscale.
"""
import os
import urllib.request

import cv2
import numpy as np
import pandas as pd

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "hand_landmarker.task")

VIDEO_ROOT = "PSL_Fingerspelling_Video_Dataset_Cropped/PSL_Fingerspelling_Video_Dataset_Cropped"

IMG_SIZE = 64
BOX_PAD = 0.30          # fraction of box size added on each side
FRAMES_PER_HOLD = 5     # sampled evenly across the hold's span


def ensure_model(path: str = MODEL_PATH) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not os.path.exists(path):
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


def make_detector(model_path: str = None, min_conf: float = 0.3):
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision
    opts = vision.HandLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=ensure_model(model_path or MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=min_conf,
    )
    return vision.HandLandmarker.create_from_options(opts)


def detect_box(detector, img_bgr, pad: float = BOX_PAD):
    """Square, padded hand box in pixels, or None. Square so the crop does not
    distort the hand's aspect ratio when resized."""
    import mediapipe as mp
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                      data=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    res = detector.detect(mp_img)
    if not res.hand_landmarks:
        return None
    lm = res.hand_landmarks[0]
    h, w = img_bgr.shape[:2]
    xs = [p.x * w for p in lm]
    ys = [p.y * h for p in lm]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    side = max(max(xs) - min(xs), max(ys) - min(ys)) * (1 + 2 * pad)
    return cx, cy, side


def crop_square(img_bgr, box, size: int = IMG_SIZE):
    """Grayscale square crop, edge-replicated where the box leaves the frame."""
    cx, cy, side = box
    h, w = img_bgr.shape[:2]
    half = side / 2
    x0, y0 = int(round(cx - half)), int(round(cy - half))
    x1, y1 = int(round(cx + half)), int(round(cy + half))

    pad_l, pad_t = max(0, -x0), max(0, -y0)
    pad_r, pad_b = max(0, x1 - w), max(0, y1 - h)
    if pad_l or pad_t or pad_r or pad_b:
        img_bgr = cv2.copyMakeBorder(img_bgr, pad_t, pad_b, pad_l, pad_r,
                                     cv2.BORDER_REPLICATE)
        x0 += pad_l; x1 += pad_l; y0 += pad_t; y1 += pad_t

    patch = img_bgr[y0:y1, x0:x1]
    if patch.size == 0:
        return None
    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    return cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)


def build(segments_csv="pseudo_label_segments.csv",
          clips_csv="pseudo_label_clips.csv",
          out_path="letter_crops.npz",
          size=IMG_SIZE, frames_per_hold=FRAMES_PER_HOLD,
          video_root=VIDEO_ROOT, verbose=True):
    """Extract crops for every pseudo-labelled hold. Returns a summary dict."""
    seg = pd.read_csv(segments_csv)
    clips = pd.read_csv(clips_csv)
    seg = seg.merge(clips[["signer_id", "video_id", "constraint_dominated"]],
                    on=["signer_id", "video_id"], how="left")

    detector = make_detector()

    images, letters, signers, videos, occ_ids, flags = [], [], [], [], [], []
    no_hand = no_frame = 0

    for (sg, vid), grp in seg.groupby(["signer_id", "video_id"], sort=True):
        cap = cv2.VideoCapture(os.path.join(video_root, sg, vid))
        if not cap.isOpened():
            no_frame += len(grp)
            continue
        for r in grp.itertuples():
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(r.minimum_frame))
            got, key_img = cap.read()
            if not got:
                no_frame += 1
                continue
            # detect once at the hold's centre; the hand barely moves during it
            box = detect_box(detector, key_img)
            if box is None:
                no_hand += 1
                continue

            span = np.linspace(r.start, r.end, frames_per_hold).round().astype(int)
            occ = f"{sg}|{vid}|{r.letter_index}"
            for f in np.unique(span):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
                got, img = cap.read()
                if not got:
                    continue
                patch = crop_square(img, box, size)
                if patch is None:
                    continue
                images.append(patch)
                letters.append(r.letter)
                signers.append(sg)
                videos.append(vid)
                occ_ids.append(occ)
                flags.append(bool(r.constraint_dominated))
        cap.release()

    X = np.stack(images) if images else np.empty((0, size, size), np.uint8)
    np.savez_compressed(
        out_path,
        images=X,
        letters=np.array(letters), signers=np.array(signers),
        videos=np.array(videos), occurrences=np.array(occ_ids),
        constraint_dominated=np.array(flags),
    )
    summary = {"images": len(X), "occurrences": len(set(occ_ids)),
               "letters": len(set(letters)), "no_hand": no_hand,
               "no_frame": no_frame, "path": out_path}
    if verbose:
        print(f"{summary['images']} crops from {summary['occurrences']} occurrences, "
              f"{summary['letters']} letters")
        print(f"holds where MediaPipe found no hand: {no_hand}; unreadable frames: {no_frame}")
        print(f"wrote {out_path}")
    return summary


if __name__ == "__main__":
    build()
