"""
Small CNN over hand-crop images, for cross-signer letter classification.

Why a CNN at all: letter_classifier.ipynb showed landmark geometry predicts
signer identity at 97 % while predicting letters below the majority baseline,
and letter_reference.ipynb confirmed by eye that the labels are sound. So the
discriminative information is in the pixels, not the 21 coordinates.

Design choices that are about the *confound*, not about accuracy
----------------------------------------------------------------
* **Grayscale input** (done in crop_dataset.py). Colour is skin tone and
  clothing, i.e. signer identity -- the exact shortcut that sank the landmark
  features.
* **No horizontal flip** in augmentation. It is the obvious free 2x, and it is
  wrong here: flipping changes handedness, and the dataset mixes signers
  recorded from left and right angles. A flipped left hand is not a right hand
  performing the same letter.
* **Small model, heavy augmentation.** 1,359 independent occurrences is tiny.
  A large pretrained backbone would have ample capacity to memorise signers.
* **Occurrence-level scoring.** The ~5 frames sampled from one hold are near
  duplicates; scoring them independently measures memorisation of a hold. Frame
  softmaxes are averaged within an occurrence before the argmax.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def augment_batch(x):
    """Random affine + brightness/contrast, applied to a whole batch at once.

    Deliberately batched rather than per-sample in __getitem__: a per-image
    grid_sample on a 1x1x64x64 tensor is dominated by Python and kernel-launch
    overhead, and profiled ~10x slower overall on CPU. Each image still gets its
    own random transform -- the batching is purely how it is computed.

    No horizontal flip. It is the obvious free 2x and it is wrong here: flipping
    changes handedness, and the dataset mixes signers recorded from left and
    right angles.
    """
    n = x.shape[0]
    ang = (torch.empty(n).uniform_(-12, 12)) * np.pi / 180
    sc = torch.empty(n).uniform_(0.9, 1.12)
    tx = torch.empty(n).uniform_(-0.08, 0.08)
    ty = torch.empty(n).uniform_(-0.08, 0.08)
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc

    theta = torch.zeros(n, 2, 3)
    theta[:, 0, 0] = cos; theta[:, 0, 1] = -sin; theta[:, 0, 2] = tx
    theta[:, 1, 0] = sin; theta[:, 1, 1] = cos;  theta[:, 1, 2] = ty

    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="border", align_corners=False)

    x = x * torch.empty(n, 1, 1, 1).uniform_(0.85, 1.15) \
        + torch.empty(n, 1, 1, 1).uniform_(-0.08, 0.08)
    return x.clamp(0, 1)


class CropDataset(Dataset):
    """Grayscale crops. Augmentation happens batch-wise in the train loop."""

    def __init__(self, images, labels, train=True):
        self.x = torch.from_numpy(images.astype(np.float32) / 255.0)[:, None]
        self.y = torch.as_tensor(np.asarray(labels), dtype=torch.long)
        self.train = train

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class LetterCNN(nn.Module):
    """Four conv blocks -> global average pool -> linear.

    GAP rather than a flatten+FC: it removes most of the parameters and makes
    the model less able to key on absolute position within the crop (e.g. where
    a particular signer's sleeve sits).
    """

    def __init__(self, n_classes, width=32, p_drop=0.3):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        w = width
        self.features = nn.Sequential(
            block(1, w), block(w, w * 2), block(w * 2, w * 4), block(w * 4, w * 4),
        )
        self.head = nn.Sequential(
            nn.Dropout(p_drop),
            nn.Linear(w * 4, n_classes),
        )

    def forward(self, x):
        h = self.features(x)
        h = h.mean(dim=(2, 3))          # global average pool
        return self.head(h)


def train_fold(Xtr, ytr, Xte, n_classes, epochs=30, bs=64, lr=3e-3,
               class_weight=None, width=32, seed=0, device=DEVICE, verbose=False):
    """Train on one fold, return per-frame softmax probabilities for Xte."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = LetterCNN(n_classes, width=width).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * max(1, (len(Xtr) + bs - 1) // bs))
    w = None if class_weight is None else torch.tensor(class_weight, dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)

    loader = DataLoader(CropDataset(Xtr, ytr, train=True), batch_size=bs,
                        shuffle=True, num_workers=0, drop_last=False)

    model.train()
    for ep in range(epochs):
        tot = n = 0
        for xb, yb in loader:
            xb = ((augment_batch(xb) - 0.5) / 0.25).to(device)
            yb = yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss) * len(yb); n += len(yb)
        if verbose and (ep + 1) % 10 == 0:
            print(f"    epoch {ep+1:>3}  loss {tot/max(n,1):.3f}")

    model.eval()
    probs = []
    with torch.no_grad():
        for xb, _ in DataLoader(CropDataset(Xte, np.zeros(len(Xte), int), train=False),
                                batch_size=256):
            xb = ((xb - 0.5) / 0.25).to(device)
            probs.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(probs), model


def occurrence_predictions(probs, occurrences, classes):
    """Average frame softmaxes within each occurrence, then argmax.

    Returns (occurrence_ids, predicted_label) in a stable order.
    """
    import pandas as pd
    df = pd.DataFrame(probs)
    df["occ"] = occurrences
    agg = df.groupby("occ").mean()
    return agg.index.to_numpy(), np.asarray(classes)[agg.to_numpy().argmax(1)]
