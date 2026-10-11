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

# Mask everything outside the hand before computing features.
#
# This is the single most important preprocessing step for live use. The
# dataset is shot against a plain green screen; a webcam sees a room. HOG is a
# gradient descriptor, so background edges inside the crop box compete directly
# with finger edges. Measured on the unmasked model: adding texture to the
# outer ring of a crop changes the predicted letter 66.7% of the time -- which
# is exactly the "same pose, random letter" behaviour seen live.
#
# The mask is the convex hull of the 21 landmarks, dilated to keep finger
# width and a little context, with everything outside set to a flat value so it
# contributes no gradients.
# The edge is FEATHERED rather than hard. A hard mask boundary is itself a
# strong gradient, so HOG keys on it -- and then a one-pixel difference in how
# the mask is rasterised flips the prediction. That made the browser port
# disagree with Python on 11 of 25 crops despite only 2.3% of pixels differing.
# Defining the mask by distance-to-hull with a soft falloff makes both
# implementations analytically identical and the boundary nearly gradient-free.
MASK_BACKGROUND = True
MASK_DILATE = 0.16      # fraction of the box side, hard radius
MASK_FEATHER = 0.08     # fraction of the box side, soft falloff beyond it
MASK_FILL = 128


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
    """(box, landmarks_px, handedness) or (None, None, None).

    The box is square so the crop does not distort the hand's aspect ratio when
    resized. Landmarks are returned in pixels so the caller can build a mask.
    """
    import mediapipe as mp
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                      data=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    res = detector.detect(mp_img)
    if not res.hand_landmarks:
        return None, None, None
    lm = res.hand_landmarks[0]
    h, w = img_bgr.shape[:2]
    pts = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)
    xs, ys = pts[:, 0], pts[:, 1]
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    side = max(xs.max() - xs.min(), ys.max() - ys.min()) * (1 + 2 * pad)
    hand = res.handedness[0][0].category_name if res.handedness else None
    return (cx, cy, side), pts, hand


def hand_mask(pts_px, box, size: int = IMG_SIZE, dilate: float = MASK_DILATE,
              feather: float = MASK_FEATHER):
    """Soft hand mask in [0,1], from signed distance to the landmark hull.

    alpha = 1 within `dilate` of the hull, falling linearly to 0 over the next
    `feather`. web/handmask.js computes exactly this.
    """
    cx, cy, side = box
    scale = size / side
    local = (pts_px - np.array([cx - side / 2, cy - side / 2])) * scale
    hull = cv2.convexHull(local.astype(np.float32))
    r = dilate * size
    f = max(1e-6, feather * size)

    alpha = np.zeros((size, size), np.float32)
    for y in range(size):
        for x in range(size):
            sd = cv2.pointPolygonTest(hull, (float(x), float(y)), True)  # + inside
            alpha[y, x] = min(1.0, max(0.0, (sd + r + f) / f))
    return alpha


def crop_square(img_bgr, box, size: int = IMG_SIZE, mask=None, fill=MASK_FILL):
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
    patch = cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)
    if mask is not None:
        blended = patch.astype(np.float32) * mask + float(fill) * (1.0 - mask)
        patch = np.clip(np.rint(blended), 0, 255).astype(np.uint8)
    return patch


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
    no_hand = no_frame = left_handed = 0

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
            box, pts, hand = detect_box(detector, key_img)
            if box is None:
                no_hand += 1
                continue
            # All five dataset signers sign right-handed (verified with
            # MediaPipe handedness). Mirroring a left hand would make it a
            # different letter, so a left-handed clip is excluded rather than
            # flipped.
            if hand == "Left":
                left_handed += 1
                continue
            mask = hand_mask(pts, box, size) if MASK_BACKGROUND else None

            span = np.linspace(r.start, r.end, frames_per_hold).round().astype(int)
            occ = f"{sg}|{vid}|{r.letter_index}"
            for f in np.unique(span):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
                got, img = cap.read()
                if not got:
                    continue
                patch = crop_square(img, box, size, mask=mask)
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
               "no_frame": no_frame, "left_handed": left_handed,
               "masked": MASK_BACKGROUND, "path": out_path}
    if verbose:
        print(f"{summary['images']} crops from {summary['occurrences']} occurrences, "
              f"{summary['letters']} letters")
        print(f"holds where MediaPipe found no hand: {no_hand}; unreadable frames: {no_frame}; "
              f"left-handed holds skipped: {left_handed}")
        print(f"background masking: {MASK_BACKGROUND}")
        print(f"wrote {out_path}")
    return summary


if __name__ == "__main__":
    build()
