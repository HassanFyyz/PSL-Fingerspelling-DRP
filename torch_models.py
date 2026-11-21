"""
Neural letter classifiers in PyTorch, with the scikit-learn interface the rest
of the repo expects: fit / predict / predict_proba / classes_.

  TorchMLP   a regularised MLP on a feature vector (dropout, weight decay,
             label smoothing, optional feature noise and mixup), averaged
             over several random initialisations.
  HandGCN    a graph convolution network over the 21 landmarks joined by the
             hand skeleton, on palm-frame coordinates.

Also here: `palm_frame` / `rich_features`, the wider feature set these models
and the tree models are compared on.

Everything runs on CPU in seconds; the data is ~1,300 rows.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from hand_geometry import FINGERS, joint_angles


# ─── features ────────────────────────────────────────────────────────────────
def palm_frame(P):
    """(n, 21, 3) landmarks in a frame attached to the hand.

    Wrist at the origin, wrist->middle-knuckle length 1, first axis along that
    bone, third axis the palm normal. Removes where the hand is, how big it is
    and how it is turned, and keeps how the fingers are arranged.
    """
    X = P - P[:, :1]
    a = X[:, 9]
    b = X[:, 17] - X[:, 5]
    e1 = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-8)
    n = np.cross(a, b)
    e3 = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-8)
    e2 = np.cross(e3, e1)
    R = np.stack([e1, e2, e3], 1)
    Xr = np.einsum("nij,nkj->nki", R, X)
    return Xr / np.maximum(np.linalg.norm(a, axis=1), 1e-8)[:, None, None]


def rich_features(P):
    """(n, 289): 19 joint angles, all 210 pairwise distances over palm length,
    and the 20 non-wrist landmarks in the palm frame (60 numbers)."""
    C = palm_frame(P)
    iu = np.triu_indices(21, 1)
    D = np.linalg.norm(C[:, iu[0]] - C[:, iu[1]], axis=-1)
    return np.hstack([joint_angles(P), D, C[:, 1:].reshape(len(P), -1)])


# ─── shared training loop ────────────────────────────────────────────────────
def _class_weights(yi, k):
    c = np.bincount(yi, minlength=k).astype(float)
    w = len(yi) / (k * np.maximum(c, 1))
    return torch.tensor(w, dtype=torch.float32)


def _train(net, X, yi, k, epochs, lr, wd, smooth, noise, mixup, balanced, seed, batch=64):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * int(np.ceil(len(X) / batch)))
    w = _class_weights(yi.numpy(), k) if balanced else None
    Y = Fn.one_hot(yi, k).float()
    if smooth:
        Y = Y * (1 - smooth) + smooth / k
    net.train()
    for _ in range(epochs):
        perm = torch.randperm(len(X), generator=g)
        for i in range(0, len(X), batch):
            b = perm[i:i + batch]
            if len(b) < 2:
                continue
            xb, yb = X[b], Y[b]
            wb = w[yi[b]] if w is not None else torch.ones(len(b))
            if noise:
                xb = xb + noise * torch.randn(xb.shape, generator=g)
            if mixup:
                lam = float(np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,), generator=g))).beta(mixup, mixup))
                j = torch.randperm(len(b), generator=g)
                xb = lam * xb + (1 - lam) * xb[j]
                yb = lam * yb + (1 - lam) * yb[j]
                wb = lam * wb + (1 - lam) * wb[j]
            loss = (-(yb * Fn.log_softmax(net(xb), 1)).sum(1) * wb).sum() / wb.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    net.eval()
    return net


class _Base:
    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


# ─── MLP on a feature vector ─────────────────────────────────────────────────
class TorchMLP(_Base):
    def __init__(self, hidden=(256, 256), dropout=0.4, wd=1e-2, smooth=0.1, noise=0.3, mixup=0.0,
                 balanced=True, epochs=150, lr=3e-3, n_models=5, seed=0):
        self.hidden, self.dropout, self.wd, self.smooth = hidden, dropout, wd, smooth
        self.noise, self.mixup, self.balanced = noise, mixup, balanced
        self.epochs, self.lr, self.n_models, self.seed = epochs, lr, n_models, seed

    def _net(self, d, k):
        layers, last = [], d
        for h in self.hidden:
            layers += [nn.Linear(last, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(self.dropout)]
            last = h
        return nn.Sequential(*layers, nn.Linear(last, k))

    def fit(self, X, y):
        torch.set_num_threads(1)
        self.classes_, yi = np.unique(y, return_inverse=True)
        self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-6
        Xt = torch.tensor((X - self.mu_) / self.sd_, dtype=torch.float32)
        yt = torch.tensor(yi, dtype=torch.long)
        self.nets_ = []
        for m in range(self.n_models):
            torch.manual_seed(self.seed * 1000 + m)
            net = self._net(X.shape[1], len(self.classes_))
            self.nets_.append(_train(net, Xt, yt, len(self.classes_), self.epochs, self.lr, self.wd,
                                     self.smooth, self.noise, self.mixup, self.balanced,
                                     self.seed * 1000 + m))
        return self

    def predict_proba(self, X):
        Xt = torch.tensor((X - self.mu_) / self.sd_, dtype=torch.float32)
        with torch.no_grad():
            return np.mean([Fn.softmax(n(Xt), 1).numpy() for n in self.nets_], axis=0)


# ─── graph convolution over the hand skeleton ────────────────────────────────
def _skeleton_adjacency():
    A = np.eye(21)
    for f in FINGERS:
        for a, b in zip(f[:-1], f[1:]):
            A[a, b] = A[b, a] = 1
    for a, b in [(5, 9), (9, 13), (13, 17)]:          # across the knuckles
        A[a, b] = A[b, a] = 1
    d = A.sum(1)
    return torch.tensor(A / np.sqrt(np.outer(d, d)), dtype=torch.float32)


class _GCNNet(nn.Module):
    def __init__(self, k, width, dropout):
        super().__init__()
        self.A = _skeleton_adjacency()
        self.l1, self.l2, self.l3 = nn.Linear(3, width), nn.Linear(width, width), nn.Linear(width, width)
        self.s1, self.s2, self.s3 = nn.Linear(3, width), nn.Linear(width, width), nn.Linear(width, width)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(21 * width, 256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, k))

    def forward(self, x):                              # x: (n, 63) flattened palm-frame coordinates
        h = x.view(len(x), 21, 3)
        for lin, self_lin in ((self.l1, self.s1), (self.l2, self.s2), (self.l3, self.s3)):
            h = Fn.gelu(torch.einsum("ij,njw->niw", self.A, lin(h)) + self_lin(h))
            h = self.drop(h)
        return self.head(h.flatten(1))


class HandGCN(_Base):
    """Takes the raw (n, 21, 3) landmarks; converts to the palm frame itself."""

    def __init__(self, width=48, dropout=0.3, wd=1e-2, smooth=0.1, noise=0.03, balanced=True,
                 epochs=150, lr=3e-3, n_models=5, seed=0):
        self.width, self.dropout, self.wd, self.smooth, self.noise = width, dropout, wd, smooth, noise
        self.balanced, self.epochs, self.lr, self.n_models, self.seed = balanced, epochs, lr, n_models, seed

    def fit(self, P, y):
        torch.set_num_threads(1)
        self.classes_, yi = np.unique(y, return_inverse=True)
        Xt = torch.tensor(palm_frame(P).reshape(len(P), -1), dtype=torch.float32)
        yt = torch.tensor(yi, dtype=torch.long)
        self.nets_ = []
        for m in range(self.n_models):
            torch.manual_seed(self.seed * 1000 + m)
            net = _GCNNet(len(self.classes_), self.width, self.dropout)
            self.nets_.append(_train(net, Xt, yt, len(self.classes_), self.epochs, self.lr, self.wd,
                                     self.smooth, self.noise, 0.0, self.balanced, self.seed * 1000 + m))
        return self

    def predict_proba(self, P):
        Xt = torch.tensor(palm_frame(P).reshape(len(P), -1), dtype=torch.float32)
        with torch.no_grad():
            return np.mean([Fn.softmax(n(Xt), 1).numpy() for n in self.nets_], axis=0)
