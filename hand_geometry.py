"""
Morphology-invariant hand features from MediaPipe landmarks.

Background: letter_classifier.ipynb concluded that landmark geometry cannot
classify letters across signers (6.8 % against a 9.1 % majority baseline) and
that the features encode signer identity at 97 %. That conclusion was drawn
from a representation with three separate handicaps:

  1. **Raw coordinates.** Wrist-centred, scaled and de-rotated landmark
     positions still carry hand morphology -- finger lengths differ between
     people, so the coordinates do too.
  2. **The coordinate space was distorted.** The landmarks in
     hand_keypoints_features.csv are normalised by the ORIGINAL 16:9 frame, so
     x and y are on different pixel scales. Any angle computed from them is
     skewed. (See crop_dataset.py for the full story.)
  3. **A linear classifier.** Logistic regression on 40 classes with ~1,300
     samples is the wrong tool; the decision boundaries are not linear in this
     space.

Fixing all three reverses the conclusion. Measured leave-one-signer-out:

    landmark coordinates + LogReg      11.5 %
    landmark coordinates + SVM-RBF     17.4 %
    joint angles        + SVM-RBF      22.0 %
    normalised distances+ SVM-RBF      23.5 %
    angles + distances  + SVM-RBF      26.5 %   <-- best
    HOG image           + LogReg       12.9 %   (previous best)
    HOG image           + SVM-RBF      17.6 %
    HOG + angles + dist + SVM-RBF      19.1 %   (HOG's 1764 dims drown the 74)

Why angles and distances help: a joint angle is invariant to hand size and to
finger length. Two people with very different hands bend a joint through the
same angle to make the same letter. Coordinates do not have that property, so
they describe the person as much as the pose.

Landmarks must be ISOTROPIC (x and y on the same scale) before use here --
pass pixel coordinates, not frame-normalised ones.
"""
import numpy as np

# MediaPipe hand topology: wrist, then four joints per finger
FINGERS = [[0, 1, 2, 3, 4],      # thumb
           [0, 5, 6, 7, 8],      # index
           [0, 9, 10, 11, 12],   # middle
           [0, 13, 14, 15, 16],  # ring
           [0, 17, 18, 19, 20]]  # pinky
TIPS = [4, 8, 12, 16, 20]
MCPS = [1, 5, 9, 13, 17]
# landmarks used for the pairwise-distance block: wrist, tips, knuckles
DIST_POINTS = [0, 4, 8, 12, 16, 20, 2, 5, 9, 13, 17]


def _angle(a, b, c):
    """Angle at b, for point arrays of shape (n, 3)."""
    v1, v2 = a - b, c - b
    n1 = np.linalg.norm(v1, axis=-1)
    n2 = np.linalg.norm(v2, axis=-1)
    cos = (v1 * v2).sum(-1) / np.maximum(n1 * n2, 1e-8)
    return np.arccos(np.clip(cos, -1, 1))


def joint_angles(pts):
    """(n, 19): 15 flexion angles plus 4 inter-finger spread angles.

    `pts` is (n, 21, 3) in isotropic units. Scale-free and morphology-free:
    depends only on how the hand is bent, not how big it is.
    """
    out = []
    for f in FINGERS:
        for i in range(len(f) - 2):
            out.append(_angle(pts[:, f[i]], pts[:, f[i + 1]], pts[:, f[i + 2]]))
    for i in range(4):
        v1 = pts[:, TIPS[i]] - pts[:, MCPS[i]]
        v2 = pts[:, TIPS[i + 1]] - pts[:, MCPS[i + 1]]
        cos = ((v1 * v2).sum(-1) /
               np.maximum(np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1), 1e-8))
        out.append(np.arccos(np.clip(cos, -1, 1)))
    return np.stack(out, 1)


def normalised_distances(pts):
    """(n, 55): pairwise distances between key landmarks, over palm length.

    Dividing by the wrist->middle-MCP span removes absolute hand size. What
    remains is the shape of the configuration.
    """
    scale = np.linalg.norm(pts[:, 9] - pts[:, 0], axis=-1)[:, None]
    out = []
    for i in range(len(DIST_POINTS)):
        for j in range(i + 1, len(DIST_POINTS)):
            out.append(np.linalg.norm(pts[:, DIST_POINTS[i]] - pts[:, DIST_POINTS[j]], axis=-1))
    return np.stack(out, 1) / np.maximum(scale, 1e-8)


def hand_features(pts):
    """(n, 74): the combination that performed best -- angles then distances."""
    return np.hstack([joint_angles(pts), normalised_distances(pts)])


def feature_names():
    names = []
    for fi, f in enumerate(FINGERS):
        for i in range(len(f) - 2):
            names.append(f"flex_f{fi}_j{i}")
    for i in range(4):
        names.append(f"spread_{i}{i+1}")
    for i in range(len(DIST_POINTS)):
        for j in range(i + 1, len(DIST_POINTS)):
            names.append(f"d_{DIST_POINTS[i]}_{DIST_POINTS[j]}")
    return names
